import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model.video_tokenizer import vidTokenizer
from model.latent_action_model import LAM
from model.dynamic_model import DynamicsModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, W, N_CODES = 30, 128, 3e-4, 8, 4
(PROJECT_ROOT / "stats").mkdir(exist_ok=True)
(PROJECT_ROOT / "checkpoints").mkdir(exist_ok=True)

tok = vidTokenizer().to(dev)
tok.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/tokenizer_frozen.pt", map_location=dev))
tok.eval().requires_grad_(False)
lam = LAM(latent_dim=128, d_model=128, n_codes=N_CODES).to(dev)
lam.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/lam_frozen.pt", map_location=dev))
lam.eval().requires_grad_(False)

d = np.load(PROJECT_ROOT / "data/pong_data.npz")
frames, dones = d["frames"], d["dones"]
T, E = frames.shape[:2]
flat = torch.from_numpy(frames.reshape(-1, 64, 64)).float().div(255).unsqueeze(1)
with torch.no_grad():
    z = torch.cat([tok.encoder(flat[i:i + 1024].to(dev)).cpu()
                   for i in range(0, len(flat), 1024)]).reshape(T, E, -1)
    a, b = z[:-1].reshape(-1, 128), z[1:].reshape(-1, 128)
    c = torch.cat([lam(a[i:i + 4096].to(dev), b[i:i + 4096].to(dev))[1].cpu()
                   for i in range(0, len(a), 4096)]).reshape(T - 1, E)

def make_windows(envs):
    s = np.arange(T - W)
    s, e = np.repeat(s, len(envs)), np.tile(envs, len(s))
    idx = s[:, None] + np.arange(W)
    ok = ~dones[idx, e[:, None]].any(1)
    s, e = s[ok], e[ok]
    zi = s[:, None] + np.arange(W + 1)
    zz = z[torch.from_numpy(zi), torch.from_numpy(e)[:, None]]
    cc = c[torch.from_numpy(zi[:, :W]), torch.from_numpy(e)[:, None]]
    return zz[:, :W], cc, zz[:, 1:]

xtr, ctr, ytr = make_windows(np.arange(E - 1))
xva, cva, yva = make_windows(np.array([E - 1]))
print(f"train {len(xtr)} windows, val {len(xva)}, baseline mse (copy z_t): "
      f"{F.mse_loss(xva, yva):.6f}")
loader = DataLoader(TensorDataset(xtr, ctr, ytr), batch_size=BATCH, shuffle=True)

model = DynamicsModel(latent_dim=128, n_codes=N_CODES, context=W).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
NOISE = 0.1 * z.std().item()
train_hist, val_hist = [], []
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x, cd, y in loader:
        x, cd, y = x.to(dev), cd.to(dev), y.to(dev)
        pred = model(x + NOISE * torch.randn_like(x), cd)
        loss = F.mse_loss(pred, y)
        x2 = torch.cat([x[:, :1], pred.detach()[:, :-1]], dim=1)
        loss = loss + F.mse_loss(model(x2, cd)[:, 1:], y[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(x)
    model.eval()
    with torch.no_grad():
        val = F.mse_loss(model(xva.to(dev), cva.to(dev)), yva.to(dev)).item()
    train_hist.append(total / len(xtr)); val_hist.append(val)
    print(f"Epoch {epoch + 1}/{EPOCHS}, train {train_hist[-1]:.6f}, val {val:.6f}")

plt.figure(figsize=(6, 4))
plt.plot(train_hist, label="train"); plt.plot(val_hist, label="val (unseen env)")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (latent)"); plt.legend()
plt.title("Dynamics training loss"); plt.tight_layout()
plt.savefig(PROJECT_ROOT / "stats" / "dynamics_loss.png", dpi=150)

with torch.no_grad():
    pick = torch.linspace(0, len(xva) - 1, 6).long()
    pred = model(xva[pick].to(dev), cva[pick].to(dev))[:, -1]
    real_img = tok.decoder(yva[pick][:, -1].to(dev))[:, 0].cpu()
    pred_img = tok.decoder(pred)[:, 0].cpu()
fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
for i in range(6):
    ax[0, i].imshow(real_img[i], cmap="gray", vmin=0, vmax=1); ax[0, i].axis("off")
    ax[1, i].imshow(pred_img[i], cmap="gray", vmin=0, vmax=1); ax[1, i].axis("off")
ax[0, 0].set_title("real next frame", loc="left"); ax[1, 0].set_title("predicted", loc="left")
plt.tight_layout(); plt.savefig(PROJECT_ROOT / "stats" / "dynamics_examples.png", dpi=150)

model.eval().requires_grad_(False)
torch.save(model.state_dict(), PROJECT_ROOT / "checkpoints" / "dynamics_frozen.pt")