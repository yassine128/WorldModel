import sys, time, ctypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, torch, cv2
from model.video_tokenizer import vidTokenizer
from model.dynamic_model import DynamicsModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
W, N_ACTIONS, FPS, SCALE = 8, 3, 15, 8
STAY, UP, DOWN = 0, 1, 2
SNAP = True 

tok = vidTokenizer().to(dev)
tok.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/tokenizer_frozen.pt", map_location=dev))
dyn = DynamicsModel(latent_dim=128, n_codes=N_ACTIONS, context=W).to(dev)
dyn.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/dynamics_frozen.pt", map_location=dev))
for m in (tok, dyn):
    m.eval().requires_grad_(False)

d = np.load(PROJECT_ROOT / "data/pong_data.npz")
frames, dones, actions = d["frames"], d["dones"], d["actions"]
T, E = frames.shape[:2]
env = E - 1
start = next(s for s in range(T // 2, T - W - 1) if not dones[s:s + W, env].any())
x = torch.from_numpy(frames[start:start + W + 1, env]).float().div(255).unsqueeze(1).to(dev)
with torch.no_grad():
    z_all = tok.encoder(x)
init_codes = torch.from_numpy(actions[start:start + W, env]).long().to(dev)
INIT_Z, INIT_C = z_all[:W].unsqueeze(0).clone(), init_codes.unsqueeze(0).clone()

def held(key):
    return bool(ctypes.windll.user32.GetAsyncKeyState(ord(key)) & 0x8000)

def to_image(frame):
    with torch.no_grad():
        f = frame[0, 0].clamp(0, 1).cpu().numpy()
    return cv2.resize((f * 255).astype(np.uint8), None, fx=SCALE, fy=SCALE,
                      interpolation=cv2.INTER_NEAREST)

z, codes = INIT_Z.clone(), INIT_C.clone()
print("hold w: up, hold s: down, nothing: stay, r: reset, q: quit")

while True:
    t0 = time.time()
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if key == ord("r"): z, codes = INIT_Z.clone(), INIT_C.clone()
    code = UP if held("W") else DOWN if held("S") else STAY

    with torch.no_grad():
        codes[0, -1] = code
        z_new = dyn(z, codes)[:, -1]
        frame = tok.decoder(z_new)
        if SNAP:
            z_new = tok.encoder(frame) 
        z = torch.cat([z[:, 1:], z_new.unsqueeze(1)], dim=1)
        codes = torch.cat([codes[:, 1:], torch.full((1, 1), STAY, device=dev)], dim=1)

    img = to_image(frame)
    cv2.putText(img, ["stay", "up", "down"][code], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.imshow("World model", img)
    time.sleep(max(0, 1 / FPS - (time.time() - t0)))

cv2.destroyAllWindows()