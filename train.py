"""
ML Arena — Pacman SB3 Training Script
Environment: ALE/MsPacman-v5, obs_type="ram" (128-byte RAM state)

Install dependencies:
    pip install -r requirements.txt

Train then upload:
    python run.py

Switching algorithms:
    Edit model.py — change ALGORITHM to PPO, A2C, etc.
    Re-run train.py and run.py.
"""

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from model import ALGORITHM, POLICY, POLICY_KWARGS, SAVE_PATH

gym.register_envs(ale_py)

# ═══ ✅ Tune freely: training hyperparameters ══════════════════════
TOTAL_TIMESTEPS = 3_000_000   # recommended: 1M+ for meaningful performance
N_ENVS          = 8           # parallel environments for faster sampling
# ══════════════════════════════════════════════════════════════════


# ═══ ✅ 進階技巧：取消注釋以啟用，可大幅提升 Agent 強度 ════════════════
# ── 技巧 1：自訂 Reward Wrapper（在每步加入中間獎勵）─────────────────
class PacmanOptimalWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. 軽い生存ペナルティ（立ち止まり防止）
        reward -= 0.01  
        
        # 2. 死亡ペナルティ（重すぎない適度な値に戻し、臆病化を防ぐ）
        if terminated and info.get("lives", 1) == 0:
            reward -= 5.0
            
        # 3. ゴーストやパワーエサのポジティブ報酬を素直に強化！
        if reward >= 2.0:
            reward *= 1.5   # 1550点出た時の強みをさらに1.5倍に伸ばす
            
        return obs, reward, terminated, truncated, info

def main():
    env = make_vec_env(
        "ALE/MsPacman-v5",
        n_envs=N_ENVS,
        env_kwargs={"obs_type": "ram"},
        wrapper_class=PacmanOptimalWrapper,  # 最適化されたラッパーを適用
    )

    model = ALGORITHM(
        POLICY,
        env,
        policy_kwargs=POLICY_KWARGS or None,
        verbose=1,
        learning_rate        = 1e-4,     # 学習率は固定にして安定させる
        buffer_size          = 200000,
        learning_starts      = 50000,
        batch_size           = 64,       # Atariで一番安定しやすい64に戻す
        exploration_fraction = 0.15,
        exploration_final_eps= 0.02,     # 2%のブレを残す
        train_freq           = 4,
        gamma                = 0.98,
        target_update_interval=10000,    # ネットワークの更新頻度を適正化
    )

    print(f"Training {ALGORITHM.__name__} for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)
    print(f"\nModel saved as {SAVE_PATH}.zip — ready to upload with run.py")
    env.close()


if __name__ == "__main__":
    main()
