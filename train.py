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
import gymnasium as gym
class PacmanRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # RAM[14] 約為剩餘生命數，可用來偵測被鬼吃到
        # RAM[123] 約為目前關卡（越高越好）
        reward -= 0.01   # 每步小懲罰，鼓勵快速得分
        if terminated and info.get("lives", 1) == 0:
            reward -= 5.0   # 死亡大懲罰
        if reward >= 2.0:
            reward *= 1.2
        return obs, reward, terminated, truncated, info
# 啟用方式：在 make_vec_env 加入 wrapper_class=PacmanRewardWrapper
# ─────────────────────────────────────────────────────────────────────
#
# ── 技巧 2：Frame Stacking（疊多幀以提供時序資訊）──────────────────
# from stable_baselines3.common.vec_env import VecFrameStack
# 在 make_vec_env 後加入：env = VecFrameStack(env, n_stack=4)
# 注意：啟用後觀測維度從 (128,) 變為 (512,)，網路容量需對應調整
# ─────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════

def main():
    # obs_type="ram" → shape (128,) uint8, consistent with env_runner.py
    env = make_vec_env(
        "ALE/MsPacman-v5",
        n_envs=N_ENVS,
        env_kwargs={"obs_type": "ram"},
    )

    model = ALGORITHM(
        POLICY,
        env,
        policy_kwargs=POLICY_KWARGS or None,
        verbose=1,
        learning_rate        = 1e-4,    # 建議：5e-5 ~ 1e-3
        buffer_size          = 200000,  # 經驗回放緩衝區；記憶體足夠時可試更大
        learning_starts      = 50000,   # 開始學習前先收集的步數（預熱期）
        batch_size           = 64,      # 建議：32 ~ 256
        exploration_fraction = 0.15,    # ε 衰減到最小值所用比例；越大→探索越久
        exploration_final_eps = 0.01,   # ε 最終值；0.05 表示 5% 隨機探索
        train_freq           = 4,       # 每幾步更新一次
        gamma                = 0.98,    # 折扣因子；建議 0.95 ~ 0.99
        target_update_interval=20000,   # 目標網路更新頻率；建議 10000 ~ 50000
    )

    print(f"Training {ALGORITHM.__name__} for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)
    print(f"\nModel saved as {SAVE_PATH}.zip — ready to upload with run.py")
    env.close()


if __name__ == "__main__":
    main()
