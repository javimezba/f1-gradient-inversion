"""
fl_reconstruction.py

Reconstruct private client images from the gradients shared in federated
learning, given white-box access to the global models (some of them
initialised with malicious "trap weights").

12 global models, 128 images each (64x64 RGB). Attack per architecture:

  MLP  (first layer Linear 12288 -> 1024, the image goes straight in)
       Analytic attack: for a neuron activated by a single sample,
       dL/dW_k / dL/db_k = x exactly.
         ReLU models          -> near-perfect recovery
         sigmoid / tanh models -> recovery is mixed across samples

  CNN  (conv 3->8, 3x3, stride S, then fc1 with trap weights)
       1. Analytic attack on fc1 recovers act(conv(x)), the feature map.
       2. The conv is white-box, so it is inverted by optimisation.
         stride 1  -> the system is overdetermined: near-exact image
         stride 4/8 -> lossy downsampling: low-frequency approximation

  ViT  (patch embedding + transformer blocks)
       No clean analytic attack (the patch-embedding bias mixes all
       patches). Gradient matching on a rebuilt ViT forward pass
       (Geiping et al.). Best-effort.

References:
  [1] Boenisch et al., "When the Curious Abandon Honesty: Federated
      Learning Is Not Private", EuroS&P 2023 (trap weights)
  [2] Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019
  [3] Geiping et al., "Inverting Gradients", NeurIPS 2020
"""

import datetime
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- CONFIG --------------------------------
N_MODELS = 12
BATCH = 128
OUT_SIZE = (64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR, GRADIENTS_DIR = "models", "gradients"
OUTPUT_PATH = "reconstructions.pt"

EPS_BIAS = 1e-9
RESID_DEDUP = 0.85        # residual-similarity threshold for de-duplication
CONV_INV_STEPS = 600
CONV_INV_LR = 0.1
VIT_STEPS, VIT_LR, TV_WEIGHT = 2500, 0.1, 1e-2

ACT_FN = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "gelu": F.gelu}


# --------------------------- UTILITIES -------------------------------
def robust_normalize(x):
    """Per-image min-max to [0, 1] after clipping the 1% / 99% quantiles."""
    x = x.float()
    n = x.shape[0]
    f = x.reshape(n, -1)
    lo = torch.quantile(f, 0.01, 1, keepdim=True)
    hi = torch.quantile(f, 0.99, 1, keepdim=True)
    f = torch.max(torch.min(f, hi), lo)
    mn = f.min(1, keepdim=True).values
    mx = f.max(1, keepdim=True).values
    return ((f - mn) / (mx - mn).clamp_min(1e-8)).view_as(x).clamp(0, 1)


def tv_per_image(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean((1, 2, 3)) + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean((1, 2, 3))


def tv_loss(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def adapt_to_output(x):
    """Force shape (N, 3, 64, 64), float32, values in [0, 1]."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    _, c, h, w = x.shape
    if c >= 3:
        x = x[:, :3]
    elif c == 1:
        x = x.repeat(1, 3, 1, 1)
    else:
        x = torch.cat([x, x[:, -1:]], 1)
    if (h, w) != OUT_SIZE:
        x = F.interpolate(x, size=OUT_SIZE, mode="bilinear", align_corners=False)
    return x.clamp(0, 1).float().contiguous().cpu()


def validate_output(sub):
    assert set(sub) == {f"model{i}" for i in range(1, N_MODELS + 1)}, "missing keys"
    for k, v in sub.items():
        assert tuple(v.shape) == (BATCH, 3, *OUT_SIZE), f"{k}: shape {tuple(v.shape)}"
        assert v.dtype == torch.float32 and torch.isfinite(v).all(), f"{k}: dtype / NaN"
        assert v.min() >= 0 and v.max() <= 1, f"{k}: values outside [0, 1]"


# ------------- DIVERSE SELECTION OF ANALYTIC CANDIDATES --------------
def select_diverse(cand, reliab, n=BATCH, dedup=RESID_DEDUP):
    """
    Pick n candidates, ranked by reliability (|db|) and structure (low TV),
    skipping near-duplicates. Similarity is measured on the RESIDUAL (after
    removing the common mean), which keeps the selection diverse: the score
    matches each guess to a DIFFERENT ground-truth image.
    """
    M = cand.shape[0]
    if M <= n:
        return torch.arange(M, device=cand.device)
    flat = cand.reshape(M, -1)
    tv = tv_per_image(robust_normalize(cand)) if cand.dim() == 4 else -flat.std(1)
    r_rel = torch.argsort(torch.argsort(-reliab)).float()
    r_tv = torch.argsort(torch.argsort(tv)).float()
    order = torch.argsort(r_rel + r_tv)
    res = F.normalize(flat - flat.mean(0, keepdim=True), dim=1, eps=1e-8)

    keep, used = [], torch.zeros(M, dtype=torch.bool, device=cand.device)
    for i in order.tolist():
        if used[i]:
            continue
        keep.append(i)
        used[i] = True
        used |= (res[i] @ res.T) > dedup
        if len(keep) >= n:
            break
    if len(keep) < n:  # not enough distinct ones: fill with the best remaining
        for i in order.tolist():
            if i not in keep:
                keep.append(i)
            if len(keep) >= n:
                break
    return torch.tensor(keep[:n], device=cand.device)


# ------------------------ ANALYTIC FC ATTACK -------------------------
def analytic_fc(grad_w, grad_b):
    """For every neuron with non-zero bias gradient: x_k = dW_k / db_k.
    Returns (candidates (M, D), reliability |db| (M,))."""
    gw = grad_w.to(DEVICE).float()
    gb = grad_b.to(DEVICE).float()
    idx = torch.nonzero(gb.abs() > EPS_BIAS, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return None
    return gw[idx] / gb[idx].unsqueeze(1), gb[idx].abs()


# ------------------------- CONV INVERSION ----------------------------
def invert_conv(feat, w_conv, b_conv, activation, stride, in_hw):
    """
    Given feat = act(conv(x)) of shape (N, Cout, Hf, Wf), recover x by
    minimising ||act(conv(x)) - feat||^2 + small TV prior.
    3x3 conv, padding 1, known (white-box) weights. Batched on GPU.
    """
    N, Cin = feat.shape[0], w_conv.shape[1]
    H, W = in_hw
    wc = w_conv.to(DEVICE).float()
    bc = b_conv.to(DEVICE).float() if b_conv is not None else None
    feat = feat.to(DEVICE).float()
    act = ACT_FN[activation]

    x = torch.rand(N, Cin, H, W, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x], lr=CONV_INV_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONV_INV_STEPS)
    for step in range(CONV_INV_STEPS):
        opt.zero_grad()
        pred = act(F.conv2d(x, wc, bc, stride=stride, padding=1))
        loss = F.mse_loss(pred, feat) + 1e-3 * tv_loss(x)
        loss.backward()
        opt.step()
        sch.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if step % 200 == 0:
            print(f"      conv inversion {step}/{CONV_INV_STEPS}  mse={loss.item():.5f}", flush=True)
    return x.detach()


# --------------------------- ViT FORWARD -----------------------------
class ViTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.heads
        qkv = self.qkv(self.norm1(x)).reshape(B, T, 3, h, D // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(D // h)).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ViT(nn.Module):
    """Minimal ViT rebuilt from a timm-style state dict."""

    def __init__(self, sd):
        super().__init__()
        pe = sd["patch_embed.proj.weight"]
        dim, ps = pe.shape[0], pe.shape[2]
        self.proj = nn.Conv2d(3, dim, ps, stride=ps)
        n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        heads = {192: 3, 256: 4}.get(dim, max(1, dim // 64))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads) for _ in range(n_blocks)])
        self.has_cls = "cls_token" in sd
        self.cls = nn.Parameter(sd["cls_token"].clone()) if self.has_cls else None
        self.pos = nn.Parameter(sd["pos_embed"].clone())
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, sd["head.weight"].shape[0])
        self._load(sd)

    def _load(self, sd):
        with torch.no_grad():
            self.proj.weight.copy_(sd["patch_embed.proj.weight"])
            self.proj.bias.copy_(sd["patch_embed.proj.bias"])
            for i, blk in enumerate(self.blocks):
                p = f"blocks.{i}."
                for mod, key in ((blk.norm1, "norm1"), (blk.norm2, "norm2"),
                                 (blk.qkv, "attn.qkv"), (blk.proj, "attn.proj"),
                                 (blk.fc1, "mlp.fc1"), (blk.fc2, "mlp.fc2")):
                    mod.weight.copy_(sd[p + key + ".weight"])
                    mod.bias.copy_(sd[p + key + ".bias"])
            pre = "norm" if "norm.weight" in sd else "fc_norm"
            self.norm.weight.copy_(sd[pre + ".weight"])
            self.norm.bias.copy_(sd[pre + ".bias"])
            self.head.weight.copy_(sd["head.weight"])
            self.head.bias.copy_(sd["head.bias"])

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.has_cls:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0] if self.has_cls else x.mean(1))


# ------------------- GRADIENT MATCHING (ViT) -------------------------
def recover_labels(grads, last_name):
    """iDLG-style label guess: most negative output-bias gradients."""
    gb = grads.get(f"{last_name}.bias")
    score = gb if gb is not None else grads[f"{last_name}.weight"].sum(1)
    lbl = torch.argsort(score)[:BATCH]
    if lbl.numel() < BATCH:
        lbl = lbl.repeat((BATCH + lbl.numel() - 1) // lbl.numel())[:BATCH]
    return lbl.to(DEVICE)


def grad_cos_loss(pred, tgt):
    num = sum((p.flatten() * t.flatten()).sum() for p, t in zip(pred, tgt))
    dp = sum((p.flatten() ** 2).sum() for p in pred)
    dt = sum((t.flatten() ** 2).sum() for t in tgt)
    return 1 - num / (dp.sqrt() * dt.sqrt() + 1e-10)


def invert_gradients_vit(model, grads, labels, steps=VIT_STEPS):
    model = model.to(DEVICE).eval()
    names = [n for n, _ in model.named_parameters() if n in grads]
    params = [p for n, p in model.named_parameters() if n in grads]
    tgt = [grads[n].to(DEVICE).float() for n in names]

    d = torch.rand(BATCH, 3, 64, 64, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([d], lr=VIT_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for step in range(steps):
        opt.zero_grad()
        g = torch.autograd.grad(F.cross_entropy(model(d), labels), params, create_graph=True)
        loss = grad_cos_loss(g, tgt) + TV_WEIGHT * tv_loss(d)
        loss.backward()
        opt.step()
        sch.step()
        with torch.no_grad():
            d.clamp_(0, 1)
        if step % 250 == 0:
            print(f"    gradient matching {step}/{steps}  loss={loss.item():.5f}", flush=True)
    return d.detach()


# ---------------------------- DISPATCH -------------------------------
def reconstruct(m):
    name = f"model{m}"
    gd = torch.load(os.path.join(GRADIENTS_DIR, f"{name}.pt"), weights_only=False, map_location="cpu")
    sd = torch.load(os.path.join(MODELS_DIR, f"{name}.pt"), weights_only=False, map_location="cpu")
    fam, act, fs, grads = gd["family"], gd["activation"], gd["feature_shape"], gd["gradients"]
    print(f"  family={fam}  activation={act}  feature_shape={fs}", flush=True)

    if fam == "mlp":
        res = analytic_fc(grads["net.0.weight"], grads["net.0.bias"])
        if res is None:
            raise RuntimeError("no active neurons")
        cand, reliab = res
        imgs = cand.view(-1, 3, 64, 64)
        sel = select_diverse(imgs, reliab)
        print(f"    [mlp] {cand.shape[0]} candidates -> {len(sel)} selected", flush=True)
        return adapt_to_output(robust_normalize(imgs[sel]))

    if fam == "cnn":
        cf, hf, wf = fs
        res = analytic_fc(grads["fc1.weight"], grads["fc1.bias"])
        if res is None:
            raise RuntimeError("no active neurons in fc1")
        cand, reliab = res
        feats = cand.view(-1, cf, hf, wf)
        sel = select_diverse(feats, reliab)
        stride = max(1, round(64 / hf))
        print(f"    [cnn] {cand.shape[0]} feature maps -> {len(sel)} selected; "
              f"inverting conv (stride={stride})", flush=True)
        # Invert with the model's conv WEIGHTS (white-box), not its gradients.
        imgs = invert_conv(feats[sel], sd["conv.weight"], sd.get("conv.bias"),
                           act, stride, in_hw=(64, 64))
        return adapt_to_output(robust_normalize(imgs))

    if fam == "vit":
        try:
            out = invert_gradients_vit(ViT(sd), grads, recover_labels(grads, "head"))
            return adapt_to_output(robust_normalize(out))
        except Exception as e:
            print(f"    [vit] forward failed ({e}) -> analytic patch fallback", flush=True)
            pe, pb = grads["patch_embed.proj.weight"], grads.get("patch_embed.proj.bias")
            r = analytic_fc(pe.reshape(pe.shape[0], -1), pb) if pb is not None else None
            if r is None:
                raise
            cand, reliab = r
            patches = cand.view(-1, 3, pe.shape[2], pe.shape[3])
            sel = select_diverse(patches, reliab)
            return adapt_to_output(robust_normalize(patches[sel]))

    raise RuntimeError(f"unknown model family: {fam}")


# ---------------------------- MAIN -----------------------------------
def main():
    out = {}
    for m in range(1, N_MODELS + 1):
        print(f"\n>>> model{m}", flush=True)
        try:
            r = reconstruct(m)
            assert r.shape == (BATCH, 3, *OUT_SIZE)
            out[f"model{m}"] = r
        except Exception as e:
            print(f"  [ERROR] model{m}: {e} -> random fallback", flush=True)
            out[f"model{m}"] = torch.rand(BATCH, 3, *OUT_SIZE, dtype=torch.float32)
        torch.save(out, OUTPUT_PATH)  # incremental save (safe on preemptible jobs)
        print(f"  saved {len(out)}/{N_MODELS}  ({datetime.datetime.now():%H:%M:%S})", flush=True)

    validate_output(out)
    torch.save(out, OUTPUT_PATH)
    print(f"\nDone -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
