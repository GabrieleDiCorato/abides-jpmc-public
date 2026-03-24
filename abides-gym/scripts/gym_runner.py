# Import to register environments
import gymnasium as gym
from tqdm import tqdm

if __name__ == "__main__":

    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
    )

    state, info = env.reset(seed=0)
    for _i in tqdm(range(5)):
        state, reward, terminated, truncated, info = env.step(0)
