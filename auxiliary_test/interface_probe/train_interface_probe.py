"""
Binary interface probe: frozen encoder → Linear → 2-class
Label: 1 = membrane interface (iof<0 AND relative_sasa>0.25), 0 = everything else
Requires sasa_cache.npy from precompute_sasa.py.
"""
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/chenty/abacust_mem/src')
sys.path.insert(0, '/home/chenty/abacust_mem/auxiliary_test/membrane_3class_probe')
from protein_utils.pdbtm_data_parser import Pdbtm_parser
from train_membrane_probe import FrozenEncoder, _make_abacust, collate_fn as _collate3

CKPT        = '/home/chenty/abacust_mem/src/experiments/abacust_mem_zero_with_0124_cluster_dict_b64_lr25/checkpoints/checkpoint100.pt'
PDBTM_DIR   = '/home/chenty/abacust_mem/src/data/data_storage/tmdet_result/npys/'
MERGED_DIR  = '/home/chenty/abacust_mem/src/data/data_storage/merged_npys/all_npy/'
CLUSTER_NPY = '/home/chenty/abacust_mem/src/data/data_storage/merged_cluster_dict.npy'
SASA_CACHE  = '/home/chenty/abacust_mem/auxiliary_test/interface_probe/sasa_cache.npy'
OUT_DIR     = '/home/chenty/abacust_mem/auxiliary_test/interface_probe'
SASA_THR    = 0.25
SEED        = 42

restypes = ['<unk>','<pad>','<cls>','<mask>',
            'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
restype_order = {r: i for i, r in enumerate(restypes)}

def seq_to_tokens(seq):
    return torch.tensor([restype_order.get(aa, 0) for aa in seq], dtype=torch.long)


class InterfaceDataset(Dataset):
    def __init__(self, split='train'):
        random.seed(SEED)
        np.random.seed(SEED)
        cluster_dict = np.load(CLUSTER_NPY, allow_pickle=True).item()
        sasa_cache   = np.load(SASA_CACHE, allow_pickle=True).item()
        self.split   = split

        pdbtm_idx, merged_idx = {}, {}
        for f in os.listdir(PDBTM_DIR):
            pdbtm_idx.setdefault(f[:4].lower(), []).append(f)
        for f in os.listdir(MERGED_DIR):
            merged_idx.setdefault(f[:4].lower(), []).append(f)

        def member_to_samples(name):
            """Return list of (merged_path, pdbtm_path, rel_sasa) for one member."""
            code = name[:4].lower()
            pf = pdbtm_idx.get(code, [])
            mf_list = merged_idx.get(code, [])
            out = []
            for mf in mf_list:
                match = [f for f in pf if f.startswith(mf[:12])]
                if not match:
                    continue
                if mf not in sasa_cache:
                    continue
                out.append((os.path.join(MERGED_DIR, mf),
                             os.path.join(PDBTM_DIR, match[0]),
                             sasa_cache[mf]))
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

        # label distribution check
        n_pos = n_tot = 0
        check_n = min(200, len(self))
        for i in range(check_n):
            item = self[i]
            if item is None:
                continue
            n_pos += (item['labels'] == 1).sum().item()
            n_tot += (item['labels'] >= 0).sum().item()
        if n_tot > 0:
            print(f'  [{split}] interface rate (first {check_n}): {n_pos}/{n_tot} = {n_pos/n_tot:.3f}')

    def __len__(self):
        if self.split == 'train':
            return len(self.cluster_centers)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == 'train':
            center = self.cluster_centers[idx % len(self.cluster_centers)]
            merged_path, pdbtm_path, rel_sasa = random.choice(self.cluster_to_samples[center])
        else:
            merged_path, pdbtm_path, rel_sasa = self.samples[idx]
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
            if len(rel_sasa) != len(seq):
                return None

            ca  = torch.tensor(bb[:, 1, :], dtype=torch.float32)
            gr, gt = G[:, :3], G[:, 3:]
            iof = ((gr @ (ca + gt.T).T).T * (N / N.norm())).sum(-1).abs() - N.norm()

            in_membrane  = (iof < 0)
            sasa_exposed = torch.tensor(rel_sasa, dtype=torch.float32) > SASA_THR

            labels = torch.zeros(len(seq), dtype=torch.long)
            labels[in_membrane & sasa_exposed] = 1

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


class ProbeHead(nn.Module):
    def __init__(self, in_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes)
    def forward(self, h):
        return self.net(h)


def binary_f1(pred, labels):
    valid = labels != -1
    pred, labels = pred[valid], labels[valid]
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-6)
    acc  = (pred == labels).float().mean().item()
    pos_rate = (labels == 1).float().mean().item()
    return acc, prec, rec, f1, pos_rate


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


def _save_plots(tag, history, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Binary interface probe [{tag}]', fontsize=14)

    # Row 0: loss / accuracy / F1
    axes[0, 0].plot(epochs, history['train_loss'], label='train')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].set_xlabel('epoch')

    axes[0, 1].plot(epochs, history['train_acc'], label='train')
    axes[0, 1].plot(epochs, history['valid_acc'], label='valid')
    axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylim(0, 1)

    axes[0, 2].plot(epochs, history['train_f1'], label='train')
    axes[0, 2].plot(epochs, history['valid_f1'], label='valid')
    axes[0, 2].set_title('F1 (interface class)'); axes[0, 2].legend(); axes[0, 2].set_xlabel('epoch')
    axes[0, 2].set_ylim(0, 1)

    # Row 1: precision / recall / pos_rate
    axes[1, 0].plot(epochs, history['train_prec'], label='train')
    axes[1, 0].plot(epochs, history['valid_prec'], label='valid')
    axes[1, 0].set_title('Precision (interface)'); axes[1, 0].legend(); axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(epochs, history['train_rec'], label='train')
    axes[1, 1].plot(epochs, history['valid_rec'], label='valid')
    axes[1, 1].set_title('Recall (interface)'); axes[1, 1].legend(); axes[1, 1].set_xlabel('epoch')
    axes[1, 1].set_ylim(0, 1)

    axes[1, 2].plot(epochs, history['valid_pos_rate'], label='valid pos rate')
    axes[1, 2].set_title('Positive Rate'); axes[1, 2].legend(); axes[1, 2].set_xlabel('epoch')
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'curves_{tag}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'[{tag}] Curves saved to {out_path}')


def run_one(encoder, tag, epochs, batch_size, lr, device, train_loader, valid_loader):
    head = ProbeHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1.0, 5.5], device=device))
    best_vf1 = 0.0

    history = {
        'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [],
        'valid_acc':  [], 'valid_prec': [], 'valid_rec':  [], 'valid_f1':  [],
        'valid_pos_rate': [],
    }

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
            loss = criterion(logits.view(B*L, 2), labels.view(B*L))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                all_pred.append(logits.argmax(-1).cpu())
                all_labels.append(labels.cpu())

        pred_t   = torch.cat([p.view(-1) for p in all_pred])
        labels_t = torch.cat([l.view(-1) for l in all_labels])
        acc, prec, rec, f1, pos = binary_f1(pred_t, labels_t)
        avg_loss = total_loss / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(acc)
        history['train_prec'].append(prec)
        history['train_rec'].append(rec)
        history['train_f1'].append(f1)

        print(f'[{tag}][Ep {epoch:02d}] loss={avg_loss:.4f} acc={acc:.4f} '
              f'F1={f1:.4f} prec={prec:.3f} rec={rec:.3f} pos_rate={pos:.3f}')

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
        vacc, vprec, vrec, vf1, vpos = binary_f1(pred_v, labels_v)

        history['valid_acc'].append(vacc)
        history['valid_prec'].append(vprec)
        history['valid_rec'].append(vrec)
        history['valid_f1'].append(vf1)
        history['valid_pos_rate'].append(vpos)

        print(f'[{tag}]        valid  acc={vacc:.4f} F1={vf1:.4f} '
              f'prec={vprec:.3f} rec={vrec:.3f} pos_rate={vpos:.3f}')

        if vf1 > best_vf1:
            best_vf1 = vf1
            torch.save(head.state_dict(), os.path.join(OUT_DIR, f'interface_probe_{tag}_best.pt'))
            print(f'[{tag}]        ★ best F1={best_vf1:.4f} saved')

    print(f'\n[{tag}] Done. Best valid F1 = {best_vf1:.4f}')
    _save_plots(tag, history, OUT_DIR)
    return best_vf1


def train(epochs=20, batch_size=8, lr=1e-3, gpu=0, group='pretrained'):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  group: {group}')

    train_ds = InterfaceDataset('train')
    valid_ds = InterfaceDataset('valid')
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
        print(f'pretrained F1  = {f1_pre:.4f}')
        print(f'random-init F1 = {f1_rnd:.4f}')
        print(f'delta          = {f1_pre - f1_rnd:+.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--gpu',        type=int,   default=0)
    p.add_argument('--group',      type=str,   default='pretrained',
                   choices=['pretrained', 'random', 'both'])
    a = p.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.gpu, a.group)
