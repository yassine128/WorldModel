import gymnasium as gym
import ale_py
from gymnasium.utils.play import play

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
play(env, keys_to_action={"w": 2, "s": 3}, zoom=4, fps=24)