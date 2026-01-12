# Import to register environments
import abides_gym  # noqa: F401 - Import needed to register environments
import gymnasium as gym


def test_gym_runner_markets_execution():

    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
    )

    state, info = env.reset(seed=0)
    for i in range(5):
        state, reward, terminated, truncated, info = env.step(0)
    env.step(1)
    env.step(2)
    env.reset()
    env.close()


def test_gym_runner_markets_daily_investor():

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
    )

    state, info = env.reset(seed=0)
    for i in range(5):
        state, reward, terminated, truncated, info = env.step(0)
    env.step(1)
    env.step(2)
    env.reset()
    env.close()
