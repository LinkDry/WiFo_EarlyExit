import time

import torch

from mask_strategy import causal_masking, causal_restore, fre_masking, fre_restore


def _forward_encoder_trunc(model, x, mask_ratio, mask_strategy, enc_depth, scale):
    N, _, T, H, W = x.shape

    x = model.Embedding(x)
    _, L, C = x.shape

    T = T // model.args.t_patch_size
    H = H // model.patch_size
    W = W // model.patch_size

    if mask_strategy == "temporal":
        x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T)
    elif mask_strategy == "fre":
        x, mask, ids_restore, ids_keep = fre_masking(x, mask_ratio, T=T, H=H, W=W)
    else:
        raise ValueError(f"Unsupported mask strategy: {mask_strategy}")

    input_size = (T, H, W)
    if model.pos_emb == "SinCos_3D":
        pos_embed_sort = model.pos_embed_enc_3d(ids_keep, N, input_size, scale=scale)
        assert x.shape == pos_embed_sort.shape
        x_attn = x + pos_embed_sort
    else:
        raise ValueError(f"Unsupported positional embedding: {model.pos_emb}")

    for index, blk in enumerate(model.blocks):
        if index >= enc_depth:
            break
        x_attn = blk(x_attn)

    x_attn = model.norm(x_attn)
    return x_attn, mask, ids_restore, input_size


def _forward_decoder_trunc(model, x, ids_restore, mask_strategy, input_size, dec_depth, scale):
    N = x.shape[0]
    T, H, W = input_size

    x = model.decoder_embed(x)
    C = x.shape[-1]

    if mask_strategy == "temporal":
        x = causal_restore(x, ids_restore, N, T, H, W, C, model.mask_token)
    elif mask_strategy == "fre":
        x = fre_restore(x, ids_restore, N, T, H, W, C, model.mask_token)
    else:
        raise ValueError(f"Unsupported mask strategy: {mask_strategy}")

    if model.pos_emb == "SinCos_3D":
        decoder_pos_embed = model.pos_embed_dec_3d(ids_restore, N, input_size, scale=scale)
        assert x.shape == decoder_pos_embed.shape
        x_attn = x + decoder_pos_embed
    else:
        raise ValueError(f"Unsupported positional embedding: {model.pos_emb}")

    for index, blk in enumerate(model.decoder_blocks):
        if index >= dec_depth:
            break
        x_attn = blk(x_attn)
    x_attn = model.decoder_norm(x_attn)
    return x_attn


def forward_trunc_from_noisy(model, imgs_n, mask_ratio, mask_strategy, enc_depth, dec_depth):
    if not 1 <= enc_depth <= len(model.blocks):
        raise ValueError(f"enc_depth must be in 1..{len(model.blocks)}, got {enc_depth}")
    if not 0 <= dec_depth <= len(model.decoder_blocks):
        raise ValueError(f"dec_depth must be in 0..{len(model.decoder_blocks)}, got {dec_depth}")

    scale = [1, 1, 1]
    start_time = time.time()
    T, H, W = imgs_n.shape[2:]
    latent, mask, ids_restore, input_size = _forward_encoder_trunc(
        model, imgs_n, mask_ratio, mask_strategy, enc_depth, scale
    )
    pred = _forward_decoder_trunc(
        model, latent, ids_restore, mask_strategy, input_size, dec_depth, scale
    )
    pred = model.decoder_pred(pred)

    Len = model.t_patch_size * model.patch_size**2
    pred_complex = pred[:, :, :Len] + 1j * pred[:, :, Len:]
    loss1, loss2, target = model.forward_loss(imgs_n, pred_complex, mask)
    return loss1, loss2, pred_complex, target, mask


def forward_trunc(model, imgs_list, mask_ratio, mask_strategy, enc_depth, dec_depth):
    imgs = torch.stack(imgs_list).squeeze(1)

    snr_db = 20
    noise_power = torch.mean(imgs**2) * 10 ** (-snr_db / 10)
    noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
    imgs_n = imgs + noise

    return forward_trunc_from_noisy(
        model, imgs_n, mask_ratio, mask_strategy, enc_depth, dec_depth
    )
