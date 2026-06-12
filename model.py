"""
✅ The ONLY file you need to change to swap algorithms or network architecture.
Both train.py and agent.py import from here — change once, sync everywhere.
"""

from stable_baselines3 import A2C  # ✅ Change to: PPO, A2C, SAC ...

# ═══ Algorithm choice (change freely) ════════════════════════════
ALGORITHM     = A2C  # SB3 algorithm class
POLICY        = "MlpPolicy"   # RAM obs = (128,); MlpPolicy is the right fit
POLICY_KWARGS = dict(
    net_arch=[256, 256],          
)
SAVE_PATH     = "model"        # SB3 appends .zip; upload model.zip via run.py
# ══════════════════════════════════════════════════════════════════

# ═══ 進階：超參數調整參考 ══════════════════════════════════════════
# 以下參數可傳入 ALGORITHM(...) 呼叫（在 train.py 的 main() 中設定）：
#
# DQN 特有（預設演算法）：
#   learning_rate        = 1e-4    # 建議：5e-5 ~ 1e-3
#   buffer_size          = 1000000 # 經驗回放緩衝區；記憶體足夠時可試更大
#   learning_starts      = 50000   # 開始學習前先收集的步數（預熱期）
#   batch_size           = 32      # 建議：32 ~ 256
#   exploration_fraction = 0.1     # ε 衰減到最小值所用比例；越大→探索越久
#   exploration_final_eps = 0.05   # ε 最終值；0.05 表示 5% 隨機探索
#   train_freq           = 4       # 每幾步更新一次
#
# PPO / A2C 特有（若改用這些演算法）：
#   learning_rate = 3e-4   # 建議：1e-4 ~ 1e-3
#   n_steps       = 128    # Atari 建議較小的 n_steps
#   ent_coef      = 0.01   # 熵正則化
#
# net_arch 建議（RAM obs 128 維；Frame Stacking 後為 512 維）：
#   [128, 128]   ← 較小選項，記憶體有限時可試
#   [256, 256]   ← 預設；啟用 Frame Stacking 後可維持或調大
# 注意：RAM obs 使用 MlpPolicy；改用 image obs（obs_type="rgb"）才需 CnnPolicy。
# ══════════════════════════════════════════════════════════════════
