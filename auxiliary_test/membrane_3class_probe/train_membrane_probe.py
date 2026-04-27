"""
Linear probe: frozen ABACUST encoder → Linear head → 3-class membrane classification
Label: 0=inner(iof<-8), 1=TM(-8≤iof≤+2), 2=outer(iof>+2)
Also runs a random-init encoder as control group.
"""
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/chenty/abacust_mem/src')
from protein_utils.pdbtm_data_parser import Pdbtm_parser
from fasde.modules.design_utils import gather_nodes

CKPT        = '/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoints/checkpoint100.pt'
PDBTM_DIR   = '/home/chenty/abacust_mem/src/data/data_storage/tmdet_result/npys/'
MERGED_DIR  = '/home/chenty/abacust_mem/src/data/data_storage/merged_npys/all_npy/'
CLUSTER_NPY = '/home/chenty/abacust_mem/src/data/data_storage/merged_cluster_dict.npy'
OUT_DIR     = '/home/chenty/abacust_mem/auxiliary_test/membrane_3class_probe'
TM_INNER, TM_OUTER = -8.0, 2.0
SEED = 42

restypes = ['<unk>','<pad>','<cls>','<mask>',
            'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
restype_order = {r: i for i, r in enumerate(restypes)}

def seq_to_tokens(seq):
    return torch.tensor([restype_order.get(aa, 0) for aa in seq], dtype=torch.long)

# ── dataset ───────────────────────────────────────────────────────────────────
class ProbeDataset(Dataset):
    def __init__(self, split='train'):
        random.seed(SEED)
        np.random.seed(SEED)
        cluster_dict = np.load(CLUSTER_NPY, allow_pickle=True).item()
        self.split = split

        pdbtm_idx, merged_idx = {}, {}
        for f in os.listdir(PDBTM_DIR):
            pdbtm_idx.setdefault(f[:4].lower(), []).append(f)
        for f in os.listdir(MERGED_DIR):
            merged_idx.setdefault(f[:4].lower(), []).append(f)

        def member_to_samples(name):
            code = name[:4].lower()
            pf = pdbtm_idx.get(code, [])
            mf_list = merged_idx.get(code, [])
            out = []
            for m in mf_list:
                match = [f for f in pf if f.startswith(m[:12])]
                if not match:
                    continue
                out.append((os.path.join(MERGED_DIR, m),
                             os.path.join(PDBTM_DIR, match[0])))
            return out

        if split == 'train':
            self.cluster_centers = []
            self.cluster_to_samples = {}
            for center, members in cluster_dict['train'].items():
                pool = []
                for name in members:
                    pool.extend(member_to_samples(name))
                if pool:
                    self.cluster_centers.append(center)
                    self.cluster_to_samples[center] = pool
            print(f'  [train] clusters={len(self.cluster_centers)}, '
                  f'total_samples={sum(len(v) for v in self.cluster_to_samples.values())}')
        else:
            self.samples = []
            for members in cluster_dict['valid'].values():
                for name in members:
                    self.samples.extend(member_to_samples(name))
            print(f'  [valid] total_samples={len(self.samples)}')

        # class distribution check
        self._check_label_dist()

    def _check_label_dist(self):
        counts = [0, 0, 0]
        total = 0
        for i in range(min(200, len(self))):
            item = self[i]
            if item is None:
                continue
            for c in range(3):
                counts[c] += (item['labels'] == c).sum().item()
            total += (item['labels'] >= 0).sum().item()
        if total > 0:
            print(f'  label dist (first 200): '
                  f'inner={counts[0]/total:.3f} TM={counts[1]/total:.3f} outer={counts[2]/total:.3f}')

    def __len__(self):
        if self.split == 'train':
            return len(self.cluster_centers)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == 'train':
            center = self.cluster_centers[idx % len(self.cluster_centers)]
            merged_path, pdbtm_path = random.choice(self.cluster_to_samples[center])
        else:
            merged_path, pdbtm_path = self.samples[idx]
        try:
            item = Pdbtm_parser._init_from_filepath(pdbtm_path)
            G = torch.tensor(item._tmatrix, dtype=torch.float32)
            N = torch.tensor(item._normal,  dtype=torch.float32)
            if N.norm() < 1e-3:
                return None
            merged = np.load(merged_path, allow_pickle=True).item()
            seq = merged['prot']['sequence']
            bb  = merged['prot']['atom_bb']
            if len(seq) < 4 or len(seq) > 600:
                return None
            ca  = torch.tensor(bb[:, 1, :], dtype=torch.float32)
            gr, gt = G[:, :3], G[:, 3:]
            iof = ((gr @ (ca + gt.T).T).T * (N / N.norm())).sum(-1).abs() - N.norm()
            labels = torch.zeros(len(seq), dtype=torch.long)
            labels[iof < TM_INNER] = 0
            labels[(iof >= TM_INNER) & (iof <= TM_OUTER)] = 1
            labels[iof > TM_OUTER] = 2
            X      = torch.tensor(bb, dtype=torch.float32)
            tokens = seq_to_tokens(seq)
            mask   = torch.ones(len(seq), dtype=torch.float32)
            residx = torch.arange(len(seq), dtype=torch.long)
            return dict(X=X, tokens=tokens, mask=mask, residx=residx, labels=labels)
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_L = max(b['X'].shape[0] for b in batch)
    def pad(t, val=0):
        p = max_L - t.shape[0]
        if   t.dim() == 1: return F.pad(t, (0, p), value=val)
        elif t.dim() == 2: return F.pad(t, (0, 0, 0, p), value=val)
        elif t.dim() == 3: return F.pad(t, (0, 0, 0, 0, 0, p), value=val)
    out = {}
    for k, val in [('X',0),('tokens',1),('mask',0),('residx',0),('labels',-1)]:
        out[k] = torch.stack([pad(b[k], val) for b in batch])
    return out

# ── encoder-only forward ──────────────────────────────────────────────────────
class FrozenEncoder(nn.Module):
    def __init__(self, abacust):
        super().__init__()
        self.prot_encoder_pifold = abacust.prot_encoder_pifold
        self.features            = abacust.features
        self.W_e                 = abacust.W_e
        self.h_V_segment         = abacust.h_V_segment
        self.encoder_layers      = abacust.encoder_layers
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, X, tokens, mask, residx):
        B, L = mask.shape
        device = X.device
        chainidx = torch.zeros(B, L, dtype=torch.long, device=device)
        lig_mask = torch.zeros(B, L, device=device)
        prot_mask = mask

        pre_h_V = self.prot_encoder_pifold(X * prot_mask[..., None, None], prot_mask, tokens)
        E, E_idx = self.features(X, mask, residx, chainidx, lig_mask=lig_mask)
        h_E = self.W_e(E)
        h_V = prot_mask[..., None] * pre_h_V
        seg_token = prot_mask.long()
        h_V = h_V + self.h_V_segment(seg_token)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V

# ── probe head ────────────────────────────────────────────────────────────────
class ProbeHead(nn.Module):
    def __init__(self, in_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes)
    def forward(self, h):
        return self.net(h)

# ── metrics ───────────────────────────────────────────────────────────────────
def macro_f1(pred, labels, num_classes=3):
    """Returns (acc, per_class=[(prec,rec,f1), ...], macro_f1)."""
    valid = labels != -1
    pred, labels = pred[valid], labels[valid]
    per_class = []
    for c in range(num_classes):
        tp = ((pred == c) & (labels == c)).sum().item()
        fp = ((pred == c) & (labels != c)).sum().item()
        fn = ((pred != c) & (labels == c)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)
        per_class.append((prec, rec, f1))
    acc = (pred == labels).float().mean().item()
    return acc, per_class, sum(x[2] for x in per_class) / num_classes

# ── build model ───────────────────────────────────────────────────────────────
def _make_abacust():
    from fasde.modules.design_utils import ABACUST
    return ABACUST(
        num_letters=35, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=7, augment_eps=0.0,
        k_neighbors=48, vocab=35, nar=True, lig_neighbor_seq_mask=False,
        ligmpnn_init=False, pre_prot_mode='proteinMPNN', esm_embedder=False,
        max_iter_num=4, embed_unimol_reprs=False, encode_mpnn=False,
        tm_raw='raw', mem_between_mode='none', num_centers=20,
        depth_inject_method='none',
    )

def build_encoder(device, random_init=False):
    abacust = _make_abacust()
    if not random_init:
        state = torch.load(CKPT, map_location='cpu', weights_only=False)
        weights = {k[len('model.abacust.'):]: v
                   for k, v in state['model'].items()
                   if k.startswith('model.abacust.')}
        missing, unexpected = abacust.load_state_dict(weights, strict=False)
        print(f'  [pretrained] missing={len(missing)}, unexpected={len(unexpected)}')
    else:
        print('  [random init] skipping weight loading')
    enc = FrozenEncoder(abacust).to(device)
    enc.eval()
    return enc

# ── plot ──────────────────────────────────────────────────────────────────────
def _save_plots(tag, history, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history['train_loss']) + 1))
    class_names = ['inner', 'TM', 'outer']
    cls_colors  = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'3-class membrane probe [{tag}]', fontsize=14)

    # Row 0: loss / accuracy / macroF1
    axes[0, 0].plot(epochs, history['train_loss'], label='train')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].set_xlabel('epoch')

    axes[0, 1].plot(epochs, history['train_acc'], label='train')
    axes[0, 1].plot(epochs, history['valid_acc'], label='valid')
    axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylim(0, 1)

    axes[0, 2].plot(epochs, history['train_mf1'], label='train')
    axes[0, 2].plot(epochs, history['valid_mf1'], label='valid')
    axes[0, 2].set_title('Macro F1'); axes[0, 2].legend(); axes[0, 2].set_xlabel('epoch')
    axes[0, 2].set_ylim(0, 1)

    # Row 1: per-class precision
    for c in range(3):
        axes[1, c].plot(epochs, history['train_prec'][c], label='train', color=cls_colors[c])
        axes[1, c].plot(epochs, history['valid_prec'][c], label='valid',
                        color=cls_colors[c], linestyle='--')
        axes[1, c].set_title(f'{class_names[c]} Precision')
        axes[1, c].legend(); axes[1, c].set_xlabel('epoch')
        axes[1, c].set_ylim(0, 1)

    # Row 2: per-class recall
    for c in range(3):
        axes[2, c].plot(epochs, history['train_rec'][c], label='train', color=cls_colors[c])
        axes[2, c].plot(epochs, history['valid_rec'][c], label='valid',
                        color=cls_colors[c], linestyle='--')
        axes[2, c].set_title(f'{class_names[c]} Recall')
        axes[2, c].legend(); axes[2, c].set_xlabel('epoch')
        axes[2, c].set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'curves_{tag}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'[{tag}] Curves saved to {out_path}')

# ── training ──────────────────────────────────────────────────────────────────
def run_one(encoder, tag, epochs, batch_size, lr, device, train_loader, valid_loader):
    head = ProbeHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_vf1 = 0.0

    history = {
        'train_loss': [], 'train_acc': [], 'train_mf1': [],
        'train_prec': [[], [], []], 'train_rec': [[], [], []], 'train_f1': [[], [], []],
        'valid_acc':  [], 'valid_mf1':  [],
        'valid_prec': [[], [], []], 'valid_rec': [[], [], []], 'valid_f1': [[], [], []],
    }
    class_names = ['inner', 'TM', 'outer']

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0; all_pred = []; all_labels = []
        for batch in train_loader:
            if batch is None: continue
            X      = batch['X'].to(device)
            tokens = batch['tokens'].to(device)
            mask   = batch['mask'].to(device)
            residx = batch['residx'].to(device)
            labels = batch['labels'].to(device)
            h_V    = encoder(X, tokens, mask, residx)
            logits = head(h_V)
            B, L, _ = logits.shape
            loss = criterion(logits.view(B*L, 3), labels.view(B*L))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                all_pred.append(logits.argmax(-1).cpu())
                all_labels.append(labels.cpu())

        pred_t   = torch.cat([p.view(-1) for p in all_pred])
        labels_t = torch.cat([l.view(-1) for l in all_labels])
        acc, per_class, mf1 = macro_f1(pred_t, labels_t)
        avg_loss = total_loss / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(acc)
        history['train_mf1'].append(mf1)
        for c, (prec, rec, f1) in enumerate(per_class):
            history['train_prec'][c].append(prec)
            history['train_rec'][c].append(rec)
            history['train_f1'][c].append(f1)

        cstr = '  '.join(
            f'{class_names[c]} p={per_class[c][0]:.3f} r={per_class[c][1]:.3f} f1={per_class[c][2]:.3f}'
            for c in range(3))
        print(f'[{tag}][Ep {epoch:02d}] loss={avg_loss:.4f} acc={acc:.4f} macroF1={mf1:.4f}')
        print(f'[{tag}]         train  {cstr}')

        head.eval()
        vpred = []; vlabels = []
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None: continue
                X      = batch['X'].to(device)
                tokens = batch['tokens'].to(device)
                mask   = batch['mask'].to(device)
                residx = batch['residx'].to(device)
                labels = batch['labels'].to(device)
                h_V    = encoder(X, tokens, mask, residx)
                vpred.append(head(h_V).argmax(-1).cpu())
                vlabels.append(labels.cpu())

        pred_v   = torch.cat([p.view(-1) for p in vpred])
        labels_v = torch.cat([l.view(-1) for l in vlabels])
        vacc, vper_class, vmf1 = macro_f1(pred_v, labels_v)

        history['valid_acc'].append(vacc)
        history['valid_mf1'].append(vmf1)
        for c, (prec, rec, f1) in enumerate(vper_class):
            history['valid_prec'][c].append(prec)
            history['valid_rec'][c].append(rec)
            history['valid_f1'][c].append(f1)

        vcstr = '  '.join(
            f'{class_names[c]} p={vper_class[c][0]:.3f} r={vper_class[c][1]:.3f} f1={vper_class[c][2]:.3f}'
            for c in range(3))
        print(f'[{tag}]        valid  acc={vacc:.4f} macroF1={vmf1:.4f}')
        print(f'[{tag}]               {vcstr}')

        if vmf1 > best_vf1:
            best_vf1 = vmf1
            torch.save(head.state_dict(), os.path.join(OUT_DIR, f'probe_{tag}_best.pt'))
            print(f'[{tag}]        ★ best macroF1={best_vf1:.4f} saved')

    print(f'\n[{tag}] Done. Best valid macroF1 = {best_vf1:.4f}')
    _save_plots(tag, history, OUT_DIR)
    return best_vf1


def train(epochs=20, batch_size=8, lr=1e-3, gpu=0, group='pretrained'):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  group: {group}')

    train_ds = ProbeDataset('train')
    valid_ds = ProbeDataset('valid')
    print(f'train={len(train_ds)}, valid={len(valid_ds)}')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    if group in ('pretrained', 'both'):
        print('\n=== Group A: pretrained encoder ===')
        enc = build_encoder(device, random_init=False)
        f1_pre = run_one(enc, 'pretrained', epochs, batch_size, lr, device, train_loader, valid_loader)
        del enc; torch.cuda.empty_cache()

    if group in ('random', 'both'):
        print('\n=== Group B: random-init encoder (control) ===')
        enc = build_encoder(device, random_init=True)
        f1_rnd = run_one(enc, 'random', epochs, batch_size, lr, device, train_loader, valid_loader)

    if group == 'both':
        print(f'\n{"="*50}')
        print(f'pretrained macroF1  = {f1_pre:.4f}')
        print(f'random-init macroF1 = {f1_rnd:.4f}')
        print(f'delta               = {f1_pre - f1_rnd:+.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--gpu',        type=int,   default=1)
    p.add_argument('--group',      type=str,   default='pretrained',
                   choices=['pretrained', 'random', 'both'])
    a = p.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.gpu, a.group)
