from gym3.interactive import Interactive
from procgen import ProcgenGym3Env


env = ProcgenGym3Env(
    num=1,
    env_name="coinrun",
    start_level=0,
    num_levels=1,
    render_mode="rgb_array",
)

try:
    Interactive(env, info_key="rgb").run()
finally:
    env.close()