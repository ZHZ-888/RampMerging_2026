'''
v2x_disturbance.py
This module implements a lightweight communication disturbance simulator
for V2X-based control systems. It simulates stochastic packet loss and
randomised communication latency, allowing evaluation of algorithm
robustness under imperfect communication conditions.
'''

from functions import print_control as prc
from collections import deque
import random

class UpdateDelayBuffer:
    def __init__(self, loss_rate=0.2, sim_step=0.1):
        self.loss_rate = loss_rate
        self.sim_step = sim_step
        self.buffer = deque()  # store (payload, release_step)

    def push(self, current_step, payload):
        """
        Decide delay (release_step) and store payload if not dropped
            latency (delay): 0.05~0.2 s

        self.buffer: (payload, release_step)
        """
        if payload:
            prc.print_message(f"\n[SEND] Step {current_step}: Generated command → {payload}")
            if random.random() < self.loss_rate:
                # print(f"[DROP] Payload lost: {payload}")
                prc.print_message("[DROP] Payload lost")
                return
            min_steps = int(0.05 / self.sim_step)
            max_steps = int(0.2 / self.sim_step)
            delay = random.randint(min_steps, max_steps)
            prc.print_message(f"[DELAY] {delay} for command → {payload}")
            release_step = current_step + delay
            self.buffer.append((payload, release_step))

    def push2(self, current_step, payload): # only in jam_control
        """Decide delay (release_step) and store payload if not dropped
        latency (delay): 0.05~0.2 s
        """
        if payload:
            min_steps = int(0.05 / self.sim_step)
            max_steps = int(0.2 / self.sim_step)
            delay = random.randint(min_steps, max_steps)
            release_step = current_step + delay

            if (payload, release_step) in self.buffer:
                prc.print_message(f"[SKIP] Duplicate entry: ({payload}, {release_step})")
                return

            prc.print_message(f"\n[SEND] Step {current_step}: Generated command → {payload}")
            if random.random() < self.loss_rate:
                # print(f"[DROP] Payload lost: {payload}")
                prc.print_message("[DROP] Payload lost")
                return

            prc.print_message(f"[DELAY] {delay} for command → {payload}")
            self.buffer.append((payload, release_step))

    def maybe_release(self, current_step):
        """Check if any payload is ready to be executed
           release commands at the release_step
           self.buffer[0][1] => release_step
        """
        if self.buffer and self.buffer[0][1] == current_step:
            payload = self.buffer.popleft()[0]
            prc.print_message(f"[EXECUTE] Step {current_step}: AV executes → {payload}")
            return payload
        return None


if __name__ == '__main__':
    prc.PRINT_ENABLED = True
    delay_buffer = UpdateDelayBuffer(loss_rate=0.2)

    dic_command = {"command1": 2, "command2": 10, "command3": 13}
    dic_command = {"command1": 2, "command2": 10, "command3": 13, "command4": 21,
                   "command5": 23, "command6": 26, "command7": 27, "command8": 28,
                   "command9": 34, "command10": 45, "command11": 56, "command12": 57}
    dic_steps = {v: k for k, v in dic_command.items()}
    for step in range(100):
        cmd = dic_steps.get(step)
        delay_buffer.push(step, cmd)
        pay_load = delay_buffer.maybe_release(step)
        if pay_load:
            pass
