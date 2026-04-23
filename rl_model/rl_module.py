# rl_module.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Required for loss plotting
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from rl_model.state_builder import StateBuilder

class SimpleMLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for regression (predicting score).
    """
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super(SimpleMLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1] range
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RLScoringAgent:
    """
    Reinforcement learning scoring agent.
    This module predicts the score of inserting a candidate AV,
    and learns to regress the expected reward based on observed outcomes.
    """
    def __init__(self, traci, data_recorder, exp_name, model_path=None, lr=5e-4, gamma=0.99):
        """
        Initialize the scoring model and training components.

        Args:
            traci: SUMO traci connection.
            model_path: Optional path to load a pretrained model.
            lr: Learning rate for optimizer. 1e-3 => 0.001; 5e-4 => 0.0005; 1e-4
            gamma: Discount factor (currently unused, but kept for future use).
        """
        self.traci = traci
        self.data_recorder = data_recorder
        self.state_builder = StateBuilder(traci, data_recorder)
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = SimpleMLP(input_dim=8).to(self.device)
        self.model = SimpleMLP(input_dim=10).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # for auto optimising parameters
        # self.loss_fn = nn.MSELoss() # Mean Squared Error Loss
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = []  # Buffer to store (state, reward) tuples
        self.loss_history = []  # Track training loss over time

        # **** Define project root (rl_model folder) ****
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Root directory for logs and models (HPC-aware)
        run_root = os.environ.get("RUN_DIR", os.path.join(self.base_dir, "rl_logs")) # environ (environment)

        # Create a unique folder name for this exp (e.g. CA_LR0.0005_B16_E5_S21_20260422_1830)
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        suffix = f"_task{array_id}" if array_id else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_id = f"{exp_name}_{timestamp}_{suffix}"
        # All-in-one run directory for logs AND models
        # self.run_dir = os.path.join(self.base_dir, "rl_logs", self.run_id)
        self.run_dir = os.path.join(run_root, self.run_id)

        os.makedirs(self.run_dir, exist_ok=True)
        # Model saving directory (optional: same as logs or a subfolder)
        self.model_save_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        # Initialise TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=self.run_dir)
        print(f"[TensorBoard] Logging to {self.run_dir}")

        if model_path: # predict mode, load the model
            self.load_model(model_path)

    def predict_score(self, state: np.ndarray) -> float:
        """
        Predict a scalar score for a given AV insertion state.
        Args:
            state: 8-dimensional normalized state vector.
        Returns:
            float: predicted score (higher means more suitable for insertion).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            score = self.model(state_tensor).item()
        return score

    def record_transition(self, state: np.ndarray, reward: float):
        """
        Record one transition (state and observed reward).
        Args:
            state: state vector used in the scoring decision.
            reward: actual reward observed after AV insertion outcome.
        """
        self.memory.append((state, reward))

    def train_on_recorded(self, current_step, epochs=1, batch_size=32):
        """
        Train the model using all recorded transitions (state, reward pairs).
        Args:
            epochs: number of training epochs per batch.
            batch_size: size of each training mini-batch.
        """
        if not self.memory:
            return

        states, rewards = zip(*self.memory)
        X_all = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        y_all = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset_size = len(X_all)
        indices = np.arange(dataset_size)

        for _ in range(epochs):
            np.random.shuffle(indices) # Shuffle the data at each epoch
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = X_all[batch_idx]
                y_batch = y_all[batch_idx]

                preds = self.model(X_batch)
                loss = self.loss_fn(preds, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.loss_history.append(loss.item()) # Record current loss

        # --- TensorBoard Logging ---
        self.writer.add_scalar('Loss/Training_Loss', loss.item(), current_step)

        print(f"[Train] Fitted on {len(self.memory)} samples, final loss = {loss.item():.4f}")
        self.memory.clear()

    def log_training_metrics(self, current_step):
        """
        SAVE (Records) training performance data to a CSV file.
        Triggered every time a training session is initiated.
        """
        if len(self.memory) < 32:
            return

        # Extract the last 32 rewards to calculate current model performance
        # before the weights are updated by the upcoming training session.
        recent_samples = self.memory[-32:]
        recent_rewards = [m[1] for m in recent_samples]

        avg_reward = np.mean(recent_rewards)
        max_reward = np.max(recent_rewards)
        min_reward = np.min(recent_rewards)

        # --- [NEW] TensorBoard Logging ---
        self.writer.add_scalar('Reward/Average', avg_reward, current_step)
        self.writer.add_scalar('Reward/Max', max_reward, current_step)
        self.writer.add_scalar('Reward/Min', min_reward, current_step)

        # Construct the data entry for academic plotting
        log_entry = {
            'session_id': len(self.loss_history),  # Index of the training iteration
            'sim_step': current_step,  # Current simulation time-step
            'avg_reward': round(float(avg_reward), 4),
            'max_reward': round(float(max_reward), 4),
            'min_reward': round(float(min_reward), 4),
            'sample_size': 32}  # Number of transitions in this batch

        # Save to CSV using append mode to ensure data persistence
        # file_path = os.path.join(self.log_dir, "training_reward_log.csv")
        file_path = os.path.join(self.run_dir, "reward_log.csv")
        file_exists = os.path.isfile(file_path)

        try:
            # Use pandas for structured data logging
            df = pd.DataFrame([log_entry])
            df.to_csv(file_path, mode='a', header=not file_exists, index=False)
            # Consistent console logging for real-time monitoring
            print(f"[Log] Session {log_entry['session_id']} at Step {current_step}: "
                  f"Mean Reward = {avg_reward:.3f}")

        except Exception as e:
            print(f"[Error] Failed to log training metrics: {e}")

    def plot_loss_curve(self):
        """
        SAVE loss history CSV
        Plot the loss curve based on recorded training history.
        """
        if not self.loss_history:
            print("[Plot] No loss history to show.")
            return
        # save loss data
        # loss_csv_path = os.path.join(self.log_dir, "loss_history.csv")
        loss_csv_path = os.path.join(self.run_dir, "loss_log.csv")
        df_loss = pd.DataFrame({'step': list(range(len(self.loss_history))), 'loss': self.loss_history})
        df_loss.to_csv(loss_csv_path, index=False)
        print(f"[Plot] Loss data saved to {loss_csv_path}")

        plt.plot(self.loss_history, label="Training Loss", color='blue', linewidth=1, alpha=0.3)
        # Smoothed loss line (moving average)
        smoothed = pd.Series(self.loss_history).rolling(window=10).mean()
        plt.plot(smoothed, label="Smoothed Loss (window=15)", color='red', linewidth=2)

        plt.xlabel("Training Step")
        plt.ylabel("MSE Loss")
        plt.title("Loss Curve of Scoring Model")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_score_scatter(self, ls_score):
        """
        save score history CSV
        """
        # Save data
        score_csv_path = os.path.join(self.run_dir, "score_log.csv")
        df_scores = pd.DataFrame({'index': list(range(len(ls_score))), 'score': ls_score})
        df_scores.to_csv(score_csv_path, index=False)
        # plot
        plt.figure(figsize=(8, 4))
        plt.scatter(range(len(ls_score)), ls_score, s=3, c='blue', label='Score', alpha=0.5)
        plt.title('Score Distribution per Decision')
        plt.xlabel('Decision Index')
        plt.ylabel('Predicted Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, filename):
        """
        Save the model parameters to a file.
        """
        full_path = os.path.join(self.model_save_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        print(f"[Model] Saved model to {full_path}")

    def load_model(self, path):
        """
        Load model parameters from a file.

        Args:
            path: path to load .pt model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)
        print(f"[Model] Loaded model from {path}")