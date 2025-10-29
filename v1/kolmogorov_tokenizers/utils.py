import torch

# imgs: [B, C, H, W] (your denoised_reconstructed_imgs)
# diffusion_t: [B] (Long), the timesteps you set
def group_images_by_timestep(imgs, timesteps, name_tag):
    assert imgs.shape[0] == timesteps.shape[0]
    unique_t = torch.unique(timesteps, sorted=True)
    groups = {}
    for t in unique_t:
        mask = (timesteps == t)
        groups[f"{name_tag}_{int(t.item())}"] = imgs[mask]
    return groups

def make_stratified_timesteps(B, T, groups=16, device="cuda"):
    """
    Return diffusion_t of shape [B] split into `groups` contiguous intervals.
    First group = low noise (small t), last = high noise (large t).
    Each group gets either ⌊B/groups⌋ or ⌈B/groups⌉ samples.
    """
    # pick one representative t per group (midpoint of each interval)
    # [0, T-1] split into `groups` bins, take mid t of each bin
    edges = torch.linspace(0, T, steps=groups+1, device=device)
    mids  = ((edges[:-1] + edges[1:]) * 0.5).round().clamp(0, T-1).long()  # [groups]

    # how many items per group (distribute the remainder to early groups)
    counts = torch.full((groups,), B // groups, device=device, dtype=torch.long)
    counts[: B % groups] += 1  # handle remainder

    diffusion_t = mids.repeat_interleave(counts)  # [B]
    return diffusion_t

def get_1d_sincos_pos_embed_torch(K: int, D: int, device=None, dtype=None) -> torch.Tensor:
    """
    Fixed 1D sin/cos positional embedding.
    Returns [1, K, D] (broadcastable across batch).
    """
    # support odd D by padding then truncating
    De = D if D % 2 == 0 else D + 1
    half = De // 2

    pos = torch.arange(K, device=device, dtype=torch.float32)             # [K]
    omega = torch.arange(half, device=device, dtype=torch.float32) / half # [half]
    omega = 1.0 / (10000 ** omega)                                        # [half]

    angles = pos[:, None] * omega[None, :]                                # [K, half]
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)                  # [K, De]
    emb = emb[:, :D]                                                      # [K, D]
    return emb.unsqueeze(0).to(dtype=dtype)                               # [1, K, D]
