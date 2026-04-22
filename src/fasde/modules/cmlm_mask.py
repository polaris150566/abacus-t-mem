import torch

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def _skeptical_unmasking(output_scores, output_masks, prob):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * prob[:, None]
    ).long()
    # `length * p`` positions with lowest scores get kept

    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def _skeptical_unmasking_all(output_scores, output_masks, prob):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores)) * prob[:, None]
    ).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def inject_noise(tokens, avail_unmask, noise=None, mask_by_unk=False, padding_idx=1, unk_idx=0):
    padding_idx = padding_idx
    mask_idx = unk_idx

    def _random_mask(target_tokens):
        # import pdb;pdb.set_trace()
        target_masks = avail_unmask

        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0) #所有输入的availunmask为false的位置都会替换成2
        target_length = target_masks.sum(1).float()# 输入为1的对应true，为零对应false
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        masked_target_tokens = target_tokens.masked_fill(target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx)
        return masked_target_tokens

    if noise == 'random_mask':
        masked_tokens = _random_mask(tokens)

    prev_tokens = masked_tokens

    return prev_tokens



# ---------------- 追加的测试函数 ----------------
def _test_inject_noise():

    # 构造一个 3×6 的小例子
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10],
                           [4, 5, 6, 7, 8, 9],
                           [3, 3, 3, 3, 3, 3]])
    avail = torch.tensor([[1, 0, 1, 1, 0, 0],
                          [1, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0]], dtype=torch.bool)

    masked = inject_noise(tokens, avail, noise='random_mask')
    print("原始 tokens:")
    print(tokens)
    print("掩码位(可扰动):")
    print(avail.int())
    print("扰动后 tokens:")
    print(masked)

    # 简单校验：被掩盖的位置必须是 0(<unk>)
    assert ((masked == 0) == avail).any(dim=1).all(), "掩码未生效！"
    print("✅ 掩码测试通过")


# 如果直接运行脚本，就执行测试
if __name__ == "__main__":
    _test_inject_noise()