import copy

import torch
from torch import nn

from mask_strategy import causal_restore, fre_restore
from wifo_trunc import _forward_encoder_trunc


class ExitHead(nn.Module):
    def __init__(self, backbone, kind):
        super().__init__()
        self.kind = kind
        self.norm = copy.deepcopy(backbone.decoder_norm)
        self.pred = copy.deepcopy(backbone.decoder_pred)
        channels = self.pred.in_features
        self.adapter = None
        if kind == "h1":
            bottleneck = channels // 4
            self.adapter = nn.Sequential(
                nn.Linear(channels, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, channels),
            )
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        elif kind != "h0":
            raise ValueError(kind)
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, hidden):
        if self.adapter is not None:
            hidden = hidden + self.adapter(hidden)
        return self.pred(self.norm(hidden))


def add_noise(imgs):
    noise_power = torch.mean(imgs**2) * 10 ** (-20 / 10)
    noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
    return imgs + noise


def exit_hidden_from_noisy(model, imgs_n, mask_ratio, mask_strategy, enc_depth, dec_depth, interface_adapter=None):
    latent, mask, ids_restore, input_size = _forward_encoder_trunc(
        model, imgs_n, mask_ratio, mask_strategy, enc_depth, [1, 1, 1]
    )
    if interface_adapter is not None:
        latent = interface_adapter(latent)
    x = model.decoder_embed(latent)
    channels = x.shape[-1]
    T, H, W = input_size
    if mask_strategy == "temporal":
        x = causal_restore(x, ids_restore, x.shape[0], T, H, W, channels, model.mask_token)
    elif mask_strategy == "fre":
        x = fre_restore(x, ids_restore, x.shape[0], T, H, W, channels, model.mask_token)
    else:
        raise ValueError(mask_strategy)
    position = model.pos_embed_dec_3d(ids_restore, x.shape[0], input_size, scale=[1, 1, 1])
    hidden = x + position
    for index, block in enumerate(model.decoder_blocks):
        if index >= dec_depth:
            break
        hidden = block(hidden)
    return hidden, mask


def exit_loss_and_outputs(model, head, hidden, imgs_n, mask):
    if head is None:
        prediction = model.decoder_pred(model.decoder_norm(hidden))
    else:
        prediction = head(hidden)
    length = model.t_patch_size * model.patch_size**2
    pred_complex = prediction[:, :, :length] + 1j * prediction[:, :, length:]
    loss1, _, target = model.forward_loss(imgs_n, pred_complex, mask)
    return loss1, pred_complex, target


def evaluate_exit(model, head, data, device, task, enc_depth, dec_depth, seed=100, batch_size=64, interface_adapter=None, mask_ratio=0.5):
    import numpy as np

    values = []
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.inference_mode():
        for start in range(0, data.shape[0], batch_size):
            array = np.array(data[start:start + batch_size], copy=True)
            imgs = torch.from_numpy(array).to(device)
            imgs_n = add_noise(imgs)
            hidden, mask = exit_hidden_from_noisy(model, imgs_n, mask_ratio, task, enc_depth, dec_depth, interface_adapter)
            _, pred, target = exit_loss_and_outputs(model, head, hidden, imgs_n, mask)
            expanded = mask.reshape(pred.shape[0], -1).unsqueeze(-1)
            numerator = (torch.abs(pred - target) ** 2 * expanded).sum(dim=(1, 2))
            denominator = (torch.abs(target) ** 2 * expanded).sum(dim=(1, 2))
            values.append((numerator / denominator).float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)
