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

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device used: {dev}")
EPOCHS, BATCH, LR, N_CODES = 30, 256, 3e-4, 4
(PROJECT_ROOT / "stats").mkdir(exist_ok=True)
(PROJECT_ROOT / "checkpoints").mkdir(exist_ok=True)

tok = vidTokenizer().to(dev)
tok.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/tokenizer_frozen.pt", map_location=dev))
tok.eval().requires_grad_(False)

d = np.load(PROJECT_ROOT / "data/pong_data.npz")
frames, dones = d["frames"], d["dones"]
T, E = frames.shape[:2]
flat = torch.from_numpy(frames.reshape(-1, 64, 64)).float().div(255).unsqueeze(1)
with torch.no_grad():
    z = torch.cat([tok.encoder(flat[i:i + 1024].to(dev)).cpu()
                   for i in range(0, len(flat), 1024)])
z = z.reshape(T, E, -1)

valid = ~dones[:-1]
zt, zt1 = z[:-1][torch.from_numpy(valid)], z[1:][torch.from_numpy(valid)]
print(f"{len(zt)} pairs, baseline mse (predict z_t+1 = z_t): {F.mse_loss(zt, zt1):.6f}")
loader = DataLoader(TensorDataset(zt, zt1), batch_size=BATCH, shuffle=True)

model = LAM(latent_dim=z.shape[-1], d_model=128, n_codes=N_CODES).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
history = []
for epoch in range(EPOCHS):
    model.train()
    total, counts = 0.0, torch.zeros(N_CODES)
    for a, b in loader:
        a, b = a.to(dev), b.to(dev)
        pred, idx = model(a, b)
        loss = F.mse_loss(pred, b)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(a)
        counts += torch.bincount(idx.cpu(), minlength=N_CODES)
    history.append(total / len(zt))
    print(f"Epoch {epoch + 1}/{EPOCHS}, loss {history[-1]:.6f}, code usage {counts.int().tolist()}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), history)
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (latent)"); plt.title("LAM training loss")
plt.tight_layout(); plt.savefig(PROJECT_ROOT / "stats" / "lam_loss.png", dpi=150)

model.eval().requires_grad_(False)
torch.save(model.state_dict(), PROJECT_ROOT / "checkpoints" / "lam_frozen.pt")