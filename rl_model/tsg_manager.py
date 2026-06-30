# rl_model/tsg_manager.py
import os
from pathlib import Path
from rl_model.rl_module import SelfGateAgent


class TSGManager:
    def __init__(self, tsg_mode="off", exp_name="default_run",
                 lr=5e-4, train_interval=32):
        self.tsg_mode = tsg_mode
        self.train_interval = train_interval
        self.next_save_step = 10000

        # default_gate_model_path = (
        #         Path(__file__).resolve().parent
        #         / "saved_models"
        #         / "task_self_gate_latest.pt"
        # )

        self.run_root = Path(os.environ.get(
            "RUN_DIR",
            Path(__file__).resolve().parent / "rl_logs"
        ))
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.train_latest_model_path = self.run_root / "task_self_gate_latest.pt"
        self.predict_model_path = (
                Path(__file__).resolve().parent
                / "saved_models"
                / "task_self_gate_latest.pt" # task_self_gate_step10900.pt; task_self_gate_latest.pt
        )

        self.gate_agent = None
        if tsg_mode in ("train", "predict"):
            if tsg_mode == 'train':
                model_path = self.train_latest_model_path if self.train_latest_model_path.exists() else None
            elif tsg_mode == 'predict':
                model_path = self.predict_model_path if self.predict_model_path.exists() else None
            self.gate_agent = SelfGateAgent(
                exp_name=f"SHARED_TSG_{exp_name}",
                model_path=model_path,
                input_dim=6,
                lr=lr
            )

    def train_if_needed(self, step, st):
        if self.tsg_mode != "train":
            return
        if self.gate_agent is None:
            raise ValueError("[TSG] tsg_mode='train' requires gate_agent")

        if len(self.gate_agent.memory) < self.train_interval:
            return

        self.gate_agent.train_on_recorded(
            current_step=step,
            epochs=5,
            batch_size=max(1, int(self.train_interval / 2))
        )

        if step > self.next_save_step or step == st * 10 - 1:
            self.gate_agent.save_model_to_path(
                self.run_root / f"task_self_gate_step{step}.pt"
            )
            self.save_latest()
            self.next_save_step += 30000

    def save_latest(self):
        if self.gate_agent is None:
            return
        self.gate_agent.save_model_to_path(self.train_latest_model_path)