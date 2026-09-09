# -*- coding: utf-8 -*-
"""
Image preprocessing shared by training AND serving — single source of truth.

This file exists in this shape on purpose. Session 6 spent hours on three bugs whose common
shape was "the training/evaluation path and the serving path do different things" (TRAINING_LOG
steps 54, 57, 59). A preprocessing step added to the trainer and not to the API would recreate
exactly that, and would be harder to spot than the greyscale bug because the model would still
produce plausible-looking output. So: one function, one name, written into the checkpoint, read
back by everything that touches the model.

The techniques come from the Kaggle Diabetic Retinopathy notebooks in `code benifit/`:

  ben_graham   The winning preprocessing of the 2015 Kaggle DR competition, still standard.
               out = 4*img - 4*gaussian_blur(img, sigma=size/30) + 128, clipped.
               It is an unsharp mask: subtracting a heavy blur removes the slow illumination
               gradient that varies with camera and pupil, and amplifies the fine structure -
               microaneurysms, haemorrhages, exudates - that the grade actually depends on.
               Two fundus photos of the same eye from different cameras look very different
               before this and nearly identical after.

  clahe        Contrast-limited adaptive histogram equalisation, per channel. Boosts local
               contrast without the global blow-out of plain equalisation. Used as a light
               augmentation (p=0.3) in the reference notebook rather than a fixed transform,
               so it is offered here as a fixed preprocessing to be *measured*, not assumed.

Implemented on numpy + PIL + skimage because cv2/albumentations are not installed in this
project's environments and the arithmetic is identical either way - cv2.addWeighted(a, 4, b,
-4, 128) is elementwise 4a - 4b + 128 with saturation, which is what this does.
"""
import numpy as np
from PIL import Image, ImageFilter

MODES = ("none", "ben_graham", "clahe")


def _to_uint8_rgb(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def ben_graham(arr, sigma_div=30.0):
    """4*img - 4*blur + 128, clipped. `arr` is HxWx3 (or HxW) uint8; returns HxWx3 uint8.

    sigma is tied to image size (size/30) exactly as in the reference notebook, so the filter
    covers the same fraction of the field of view at any resolution.
    """
    a = _to_uint8_rgb(arr)
    h, w = a.shape[:2]
    sigma = max(1.0, float(max(h, w)) / sigma_div)
    blur = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=sigma)),
                      dtype=np.float32)
    out = 4.0 * a.astype(np.float32) - 4.0 * blur + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(arr, clip_limit=0.02):
    """Per-channel CLAHE. skimage works in float [0,1] and returns the same, so scale back."""
    from skimage.exposure import equalize_adapthist
    a = _to_uint8_rgb(arr)
    out = equalize_adapthist(a.astype(np.float32) / 255.0, clip_limit=clip_limit)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def apply(arr, mode):
    """Dispatch. `mode` None or "none" is the identity, so an untouched checkpoint behaves
    exactly as it did before this module existed."""
    if not mode or mode == "none":
        return _to_uint8_rgb(arr)
    if mode == "ben_graham":
        return ben_graham(arr)
    if mode == "clahe":
        return clahe(arr)
    raise ValueError("unknown preprocessing mode %r; expected one of %s" % (mode, ", ".join(MODES)))


def apply_pil(pil_img, mode):
    """Same thing for a PIL image in, PIL image out - what the serving path holds."""
    if not mode or mode == "none":
        return pil_img
    return Image.fromarray(apply(np.asarray(pil_img.convert("RGB")), mode))


# --- test-time augmentation views ---------------------------------------------------------
# The project has always used a single hflip average. The Kaggle DR notebook in code benifit/
# uses five views (identity, hflip, +10 deg, -10 deg, centre-crop 0.9). More views average out
# more of the model's sensitivity to framing, at a linear cost in inference time.
#
# But session 6 step 67 measured what happens when an augmentation contradicts the label:
# organc's classes include kidney-left / kidney-right, and an hflip average asks the model to
# agree with its own mirror image - which for those classes is the OTHER class. So the view
# set is not a constant. It is chosen from what the label is allowed to be invariant to, and
# that is a property of the dataset, not of the technique.
#
#   "flip"     identity + hflip                     - the historical default
#   "multi"    identity + hflip + rot(+10,-10) + centre-crop 0.9
#   "safe"     identity + rot(+7,-7) + centre-crop 0.95   - no mirroring, small rotations
#   "none"     identity only
TTA_VIEWS = {
    "none":  ["id"],
    "flip":  ["id", "hflip"],
    "multi": ["id", "hflip", "rot+10", "rot-10", "crop0.9"],
    "safe":  ["id", "rot+7", "rot-7", "crop0.95"],
}


def tta_batch(x, view):
    """Apply one named view to a normalised NCHW tensor. Returns a tensor of the same shape.

    Rotation and cropping are done with torch so this runs on whatever device the batch is
    already on - no host round-trip per view.
    """
    import torch
    import torch.nn.functional as F
    if view == "id":
        return x
    if view == "hflip":
        return torch.flip(x, dims=[3])
    if view.startswith("rot"):
        deg = float(view[3:])
        th = torch.tensor(deg * 3.141592653589793 / 180.0, device=x.device, dtype=x.dtype)
        cos, sin = torch.cos(th), torch.sin(th)
        m = torch.zeros(x.size(0), 2, 3, device=x.device, dtype=x.dtype)
        m[:, 0, 0], m[:, 0, 1] = cos, -sin
        m[:, 1, 0], m[:, 1, 1] = sin, cos
        grid = F.affine_grid(m, list(x.shape), align_corners=False)
        # zeros padding, matching how a rotated PIL image would come out on a black border
        return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    if view.startswith("crop"):
        f = float(view[4:])
        h, w = x.shape[2], x.shape[3]
        ch, cw = int(round(h * f)), int(round(w * f))
        top, left = (h - ch) // 2, (w - cw) // 2
        return F.interpolate(x[:, :, top:top + ch, left:left + cw], size=(h, w),
                             mode="bilinear", align_corners=False)
    raise ValueError("unknown TTA view %r" % view)


def views_for(ck):
    """The TTA view list a checkpoint should be served with.

    One place for the backward-compatibility logic, so six call sites do not each invent it.
    Checkpoints written before 2026-09-08 carry only a boolean `tta`, which meant exactly
    "average with the horizontal flip" - so False -> ["id"] and True -> ["id", "hflip"], which
    is byte-for-byte what those models were measured with. Newer ones carry `tta_views`, a key
    into TTA_VIEWS.
    """
    name = ck.get("tta_views")
    if name:
        if name not in TTA_VIEWS:
            raise ValueError("checkpoint asks for unknown TTA view set %r" % name)
        return TTA_VIEWS[name], name
    return (TTA_VIEWS["flip"], "flip") if ck.get("tta") else (TTA_VIEWS["none"], "none")


def tta_average(net, x, views):
    """Mean softmax over the named views. `views` is a list of view names."""
    import torch
    acc = None
    for v in views:
        p = torch.softmax(net(tta_batch(x, v)), 1)
        acc = p if acc is None else acc + p
    return acc / len(views)
