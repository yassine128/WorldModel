import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np, torch, cv2
from model.video_tokenizer import vidTokenizer
from model.latent_action_model import LAM
from model.dynamic_model import DynamicsModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
W, N_CODES, FPS, SCALE = 8, 4, 15, 8

KEY_TO_CODE = {ord("w"): 1, ord("s"): 2, ord("0"): 0, ord("1"): 1, ord("2"): 2, ord("3"): 3}
IDLE_CODE = 0 
KEY_HOLD_SECONDS = 0.15
SNAP = True 

tok = vidTokenizer().to(dev)
tok.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/tokenizer_frozen.pt", map_location=dev))
lam = LAM(latent_dim=128, d_model=128, n_codes=N_CODES).to(dev)
lam.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/lam_frozen.pt", map_location=dev))
dyn = DynamicsModel(latent_dim=128, n_codes=N_CODES, context=W).to(dev)
dyn.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints/dynamics_frozen.pt", map_location=dev))
for m in (tok, lam, dyn):
    m.eval().requires_grad_(False)

d = np.load(PROJECT_ROOT / "data/pong_data.npz")
frames, dones = d["frames"], d["dones"]
T, E = frames.shape[:2]
env = E - 1
start = next(s for s in range(T // 2, T - W - 1) if not dones[s:s + W, env].any())
x = torch.from_numpy(frames[start:start + W + 1, env]).float().div(255).unsqueeze(1).to(dev)
with torch.no_grad():
    z_all = tok.encoder(x)                                   
    init_codes = lam(z_all[:-1], z_all[1:])[1]              
INIT_Z, INIT_C = z_all[:W].unsqueeze(0).clone(), init_codes.unsqueeze(0).clone()

def to_image(frame):
    with torch.no_grad():
        f = frame[0, 0].clamp(0, 1).cpu().numpy()
    return cv2.resize((f * 255).astype(np.uint8), None, fx=SCALE, fy=SCALE,
                      interpolation=cv2.INTER_NEAREST)

z, codes = INIT_Z.clone(), INIT_C.clone()
last_key, last_key_time = None, 0.0
print("w / s or 0-3: choose code, r: reset to the starting frames, q: quit")

while True:
    t0 = time.time()
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if key == ord("r"): z, codes = INIT_Z.clone(), INIT_C.clone()
    if key in KEY_TO_CODE: last_key, last_key_time = key, time.time()
    held = last_key is not None and time.time() - last_key_time < KEY_HOLD_SECONDS
    code = KEY_TO_CODE[last_key] if held else IDLE_CODE

    with torch.no_grad():
        codes[0, -1] = code
        z_new = dyn(z, codes)[:, -1]
        frame = tok.decoder(z_new)
        if SNAP:
            z_new = tok.encoder(frame) 
        z = torch.cat([z[:, 1:], z_new.unsqueeze(1)], dim=1)
        codes = torch.cat([codes[:, 1:], torch.full((1, 1), IDLE_CODE, device=dev)], dim=1)

    img = to_image(frame)
    cv2.putText(img, f"code {code}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.imshow("World model", img)
    time.sleep(max(0, 1 / FPS - (time.time() - t0)))

cv2.destroyAllWindows()