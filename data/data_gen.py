import numpy as np, cv2
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

N_ENVS, STEPS = 8, 5000 
ACTION_MAP = np.array([0, 2, 3])
HOLD_MIN, HOLD_MAX = 4, 16

def prep(frame):
    f = cv2.resize(frame[34:194], (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

def main():
    rng = np.random.default_rng(0)
    envs = gym.make_vec("ALE/Pong-v5", num_envs=N_ENVS, vectorization_mode="async",
                        frameskip=4, repeat_action_probability=0.0)
    obs, _ = envs.reset(seed=0)

    frames = np.zeros((STEPS, N_ENVS, 64, 64), np.uint8)
    dones = np.zeros((STEPS, N_ENVS), bool)

    a = rng.integers(0, 3, N_ENVS)
    hold = rng.integers(HOLD_MIN, HOLD_MAX + 1, N_ENVS)

    for t in range(STEPS):
        frames[t] = [prep(o) for o in obs]
        obs, r, term, trunc, _ = envs.step(ACTION_MAP[a])
        dones[t] = term | trunc

        hold -= 1
        expired = hold <= 0
        a[expired] = rng.integers(0, 3, expired.sum())
        hold[expired] = rng.integers(HOLD_MIN, HOLD_MAX + 1, expired.sum())

    np.savez_compressed("pong_data.npz", frames=frames, dones=dones)
    envs.close()

if __name__ == "__main__":
    main()