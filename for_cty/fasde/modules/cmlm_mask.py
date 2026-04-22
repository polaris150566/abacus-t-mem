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

    # def _full_mask(target_tokens):
    #     target_mask = (
    #         target_tokens.ne(padding_idx)  # & mask
    #         & target_tokens.ne(cls_idx)
    #         & target_tokens.ne(eos_idx)
    #     )
    #     # masked_target_tokens = target_tokens.masked_fill(~target_mask, mask_idx)
    #     masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
    #     return masked_target_tokens

    def _random_mask(target_tokens):
        # target_masks = (
        #         target_tokens.ne(padding_idx) & coord_mask
        #     )
        target_masks = avail_unmask
        
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        masked_target_tokens = target_tokens.masked_fill(target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx)
        return masked_target_tokens 

    # def _selected_mask(target_tokens, sel_mask):
    #     masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
    #     return masked_target_tokens

    # def _adaptive_mask(target_tokens):
    #     raise NotImplementedError

    if noise == 'random_mask':
        masked_tokens = _random_mask(tokens)

    # if noise == 'full_mask':
    #     masked_tokens = _full_mask(tokens)
    # elif noise == 'random_mask':
    #     masked_tokens = _random_mask(tokens)
    # elif noise == 'selected_mask':
    #     masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
    # elif noise == 'no_noise':
    #     masked_tokens = tokens
    # else:
    #     raise ValueError(f"Noise type ({noise}) not defined.")

    prev_tokens = masked_tokens
    # prev_token_mask = torch.all(torch.stack([prev_tokens.eq(mask_idx), coord_mask], -1), -1)
    # target_mask = prev_token_mask & coord_mask

    return prev_tokens#, prev_token_mask  # , target_mask
