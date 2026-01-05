# train_rl_agent.py

import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from platoon_split_env_mock import PlatoonSplitEnv

# === Configurations ===
MODEL_DIR = "rl_model"
LOG_DIR = "rl_logs"
TOTAL_TIMESTEPS = 200_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Initialize custom env ===
env = PlatoonSplitEnv()  # your custom gymnasium.Env

# === Wrap with SB3 compatibility if needed ===
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env(lambda: env, n_envs=1)

# === Load and train PPO agent ===
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_platoon"
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

model.save(f"{MODEL_DIR}/best_model")
print("Model saved to:", f"{MODEL_DIR}/best_model.zip")
