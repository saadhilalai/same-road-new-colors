"""
Step 4: what the AI looks at. Two ways to make a heat map of how much the AI's decision (steering and pedal)
depends on each part of its 64x64 view:

1. Blur test (the main method; Greydanus et al. 2018, "Visualizing and Understanding Atari Agents"): blur one small
   round patch of the image at a time (in both stacked images) and measure how much the decision changes.
   A big change means the AI relies on what is in that patch.
2. Gradient map (the check): how strongly a tiny change to each pixel would move the decision (the derivative).

Both use the AI's best action (the Beta mean, as in every evaluation).
"""

import numpy as np
import torch as th
import torch.nn.functional as F

import config

BLUR_SIGMA = 3.0        # how strongly the patch is blurred (pixels of the 64x64 view)
MASK_SIGMA = 4.0        # size of the round patch (Greydanus: 5 on 84x84 images, scaled to 64x64)
MASK_STRIDE = 4         # one patch every 4 pixels: 16 x 16 = 256 patches per image
CHUNK = 1024            # images per pass through the network


def _gaussian_kernel(sigma):
    r = int(np.ceil(3 * sigma))
    x = th.arange(-r, r + 1, dtype=th.float32)
    k = th.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(x, sigma=BLUR_SIGMA):
    """Gaussian blur of every channel of (N, C, H, W)."""
    k = _gaussian_kernel(sigma).to(x.device)
    c, r = x.shape[1], len(k) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def patch_masks(size=config.OBS_SIZE, stride=MASK_STRIDE, sigma=MASK_SIGMA, device="cpu"):
    """(n_patches, size, size): one round patch per grid point (peak 1)."""
    centers = th.arange(stride // 2, size, stride, dtype=th.float32)
    yy, xx = th.meshgrid(th.arange(size, dtype=th.float32), th.arange(size, dtype=th.float32), indexing="ij")
    masks = [th.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)) for cy in centers for cx in centers]
    return th.stack(masks).to(device)


def best_action(policy, obs):
    """The AI's best action [steer, pedal] for (N, 6, 64, 64) images with values 0-255 (differentiable)."""
    return policy.get_distribution(obs).mode()


def to_tensor(stacks, device):
    """(N, 64, 64, 6) uint8 stacks -> (N, 6, 64, 64) float tensor, the layout the AI's network reads."""
    return th.as_tensor(np.ascontiguousarray(stacks.transpose(0, 3, 1, 2)), device=device).float()


@th.no_grad()
def blur_maps(policy, stacks, masks):
    """(N, 64, 64) blur-test heat maps: 0.5 * squared change of [steer, pedal] when each patch is blurred."""
    obs = to_tensor(stacks, masks.device)
    blurred = blur(obs)
    m = masks[:, None]                                          # (P, 1, H, W)
    out = []
    for i in range(len(obs)):
        base = best_action(policy, obs[i:i + 1])                # (1, 2)
        perturbed = obs[i][None] * (1 - m) + blurred[i][None] * m
        change = th.cat([best_action(policy, perturbed[j:j + CHUNK]) for j in range(0, len(perturbed), CHUNK)])
        score = 0.5 * ((change - base) ** 2).sum(dim=1)        # (P,)
        out.append((score[:, None, None] * masks).sum(0) / masks.sum(0))
    return th.stack(out).cpu().numpy()


def gradient_maps(policy, stacks, device):
    """(N, 64, 64) gradient heat maps: |d steer / d pixel| + |d pedal / d pixel|, added up over the 6 channels."""
    obs = to_tensor(stacks, device).requires_grad_(True)
    action = best_action(policy, obs)
    total = th.zeros_like(obs)
    for k in range(action.shape[1]):
        (g,) = th.autograd.grad(action[:, k].sum(), obs, retain_graph=True)
        total += g.abs()
    return total.sum(dim=1).detach().cpu().numpy()


def region_shares(maps, fractions):
    """
    maps (N, 64, 64), fractions (N, R, 64, 64) -> (N, R): share of each image's heat that falls in each region.
    Images with no heat at all get NaN (left out of averages).
    """
    fractions = fractions.astype(np.float32)
    heat = (maps[:, None] * fractions).sum(axis=(2, 3))
    total = maps.sum(axis=(1, 2))[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, heat / total, np.nan)
