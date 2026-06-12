"""
ML Arena — Pacman Agent (Stable Baselines 3)
Environment: ALE/MsPacman-v5  obs_type="ram"

Observation (observation):
    numpy.ndarray, shape=(128,), dtype=uint8 — Atari RAM state

Action space (action_space):
    gymnasium.spaces.Discrete(9)
    0=NOOP  1=UP  2=RIGHT  3=LEFT  4=DOWN
    5=UPRIGHT  6=UPLEFT  7=DOWNRIGHT  8=DOWNLEFT

Score: average total reward across episodes, higher is better.
"""

import os

import numpy as np

from model import ALGORITHM  # ✅ Algorithm defined in model.py — do not edit here

# ┌─────────────────────────────────────────────────────────────────┐
# │  What to change                                                  │
# │  ✅ Change: model.py (algorithm, policy, network architecture)   │
# │  ❌ Fixed:  class Agent, act() signature, return type int        │
# │            weights_path (system mounts to this location)         │
# └─────────────────────────────────────────────────────────────────┘

class Agent:
    def __init__(self):
        weights_path = os.path.join(os.path.dirname(__file__), "model.zip")
        self.model = ALGORITHM.load(weights_path, device="cpu")
        self._state = None  # for recurrent policies (LSTM etc.)

    def reset(self, _observation: np.ndarray):
        """✅ Optional: reset recurrent state at the start of each episode."""
        self._state = None

    def act(self, observation: np.ndarray, _action_space) -> int:
        """
        ❌ Fixed: method name, signature, return type
        ✅ Change: pass deterministic=False for stochastic policies

        Args:
            observation: shape (128,) uint8 — Atari RAM state
            action_space: gymnasium Discrete(9)
        Returns:
            int 0~8
        """
        action, self._state = self.model.predict(
            observation, state=self._state, deterministic=True
        )
        return int(action)
