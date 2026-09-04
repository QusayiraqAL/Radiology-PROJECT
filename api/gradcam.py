# -*- coding: utf-8 -*-
"""Grad-CAM for the ResNet image models — the teaching feature that shows WHERE the network
looks. Returns a base64 PNG overlaying a class-activation heatmap on the original image, so a
student can see which region drove the prediction (consolidation, a lesion, a cell, an organ).

Works with any torchvision ResNet here (target layer = layer4[-1]). Kept dependency-light:
numpy + PIL only for the colormap/overlay (no matplotlib at serve time).
"""
import io
import base64
import numpy as np
import torch
from PIL import Image

# a compact "jet-like" colormap (blue -> cyan -> green -> yellow -> red) as RGB stops
_CMAP = np.array([
    [0, 0, 128], [0, 0, 255], [0, 128, 255], [0, 255, 255],
    [0, 255, 128], [128, 255, 0], [255, 255, 0], [255, 128, 0], [255, 0, 0],
], dtype=np.float32)


def _colorize(gray01):
    """Map a HxW array in [0,1] to an RGB heatmap via the jet-like colormap."""
    x = np.clip(gray01, 0, 1) * (len(_CMAP) - 1)
    lo = np.floor(x).astype(int); hi = np.minimum(lo + 1, len(_CMAP) - 1)
    frac = (x - lo)[..., None]
    return (_CMAP[lo] * (1 - frac) + _CMAP[hi] * frac).astype(np.uint8)


def gradcam_overlay(net, x, target_idx, base_pil, target_layer=None, alpha=0.45):
    """net: torchvision ResNet; x: (1,3,H,W) normalized tensor on the model's device;
    target_idx: class channel; base_pil: original RGB PIL to overlay on.
    Returns 'data:image/png;base64,...' or None on failure (never raises into serving)."""
    try:
        layer = target_layer if target_layer is not None else net.layer4[-1]
        acts, grads = {}, {}

        def fwd_hook(_m, _i, o): acts["v"] = o.detach()
        def bwd_hook(_m, gi, go): grads["v"] = go[0].detach()

        h1 = layer.register_forward_hook(fwd_hook)
        h2 = layer.register_full_backward_hook(bwd_hook)
        try:
            was_training = net.training
            net.eval()
            x = x.clone().requires_grad_(True)
            logits = net(x)
            net.zero_grad(set_to_none=True)
            logits[0, target_idx].backward()
            A = acts["v"][0]                       # (C,h,w)
            G = grads["v"][0]                       # (C,h,w)
        finally:
            h1.remove(); h2.remove()

        # Grad-CAM++ weighting: sharper, better-localized maps than vanilla Grad-CAM.
        # alpha_k = G^2 / (2 G^2 + sum_hw(A * G^3)); w_k = sum_hw(alpha_k * relu(G))
        G2 = G * G
        G3 = G2 * G
        denom = 2 * G2 + (A * G3).sum(dim=(1, 2), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha_pp = G2 / denom                                 # renamed: don't shadow blend `alpha`
        weights = (alpha_pp * torch.relu(G)).sum(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * A).sum(0)).cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        W, H = base_pil.size
        cam_img = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
        heat = _colorize(cam_img)
        base = np.asarray(base_pil.convert("RGB"), dtype=np.float32)
        # only tint the hot regions so anatomy stays visible (mask by activation strength)
        m = (cam_img[..., None] ** 0.8) * alpha
        blend = (base * (1 - m) + heat.astype(np.float32) * m).astype(np.uint8)

        buf = io.BytesIO(); Image.fromarray(blend).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None
