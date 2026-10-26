import sys
from pathlib import Path

import numpy as np, torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.video_tokenizer import vidTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compute device is {dev}")
stats_dir = PROJECT_ROOT / "stats"
stats_dir.mkdir(exist_ok=True)

frames = np.load(PROJECT_ROOT / "data" / "pong_data.npz")["frames"].reshape(-1, 64, 64)
X = torch.from_numpy(frames).float().div(255).unsqueeze(1).to(dev)

model = vidTokenizer(learning_rate=1e-3).to(dev)

idx = torch.linspace(0, len(X) - 1, 6).long() 
examples = X[idx]

def reconstruct():
    model.eval()
    with torch.no_grad():
        p = model(examples)
    return p.cpu().numpy()[:, 0]

def psnr(p):
    mse = ((p - examples.cpu().numpy()[:, 0]) ** 2).mean()
    return 10 * np.log10(1 / mse)

before = reconstruct()
history = model.fit(X, epochs=30, batch_size=64)
after = reconstruct()
real = examples.cpu().numpy()[:, 0]

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(history) + 1), history)
plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("train MSE (log scale)"); plt.title("Tokenizer training loss")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(stats_dir / "loss_curve.png", dpi=150); plt.close()

fig, axes = plt.subplots(3, len(real), figsize=(2 * len(real), 6.4))
for j in range(len(real)):
    for i, (img, name) in enumerate([(real, "real"), (before, f"before ({psnr(before):.1f} dB)"),
                                     (after, f"after ({psnr(after):.1f} dB)")]):
        axes[i, j].imshow(img[j], cmap="gray", vmin=0, vmax=1)
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(name, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(stats_dir / "before_after.png", dpi=150); plt.close()

print(f"final loss {history[-1]:.5f}, psnr {psnr(before):.1f} -> {psnr(after):.1f} dB")

model.eval()
model.requires_grad_(False)  # freeze
torch.save(model.state_dict(), PROJECT_ROOT / "tokenizer_frozen.pt")