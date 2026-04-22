# test_rl_agent.py

from platoon_split_env_mock import PlatoonSplitEnv
from stable_baselines3 import PPO

# Load environment and trained PPO model
env = PlatoonSplitEnv()
model = PPO.load("/home/zzha/PycharmProjects/RampMerging4_250208/PlatoonSplit_RL_model/rl_model/best_model")

# Run one episode to evaluate the agent's behavior
obs, _ = env.reset()
for i in range(10):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    print(f"\nStep {i+1}")
    print(f"Action (score): {action}")
    print(f"Reward: {reward}")
    print(f"New state: {obs}")