# 學生端專案檔案說明 — Pacman RL 競賽（Stable Baselines 3）

本目錄包含參加 Pacman RL 競賽所需的所有基礎檔案，使用 **Stable Baselines 3** 框架：

## 1. `model.py`（演算法設定，**主要修改點**）
- **用途**：定義 SB3 演算法類別、Policy 與網路結構。
- **設計**：`train.py` 和 `agent.py` 都從這裡 import，**只需改一次即自動同步**。
- **換演算法**：修改 `ALGORITHM = DQN` 那行，例如改成 `PPO`、`A2C`。
  - 務必重新執行 `train.py` 產生對應的 `model.zip`。
- **調整網路**：修改 `POLICY_KWARGS` 中的 `net_arch`。

## 2. `agent.py`（競賽介面，**必須保留**）
- **用途**：定義系統載入與呼叫你模型的標準介面。
- **規則**：`class Agent` 名稱、`act()` 方法名稱與參數簽名**不可修改**。
- **可改**：將 `deterministic=True` 改為 `False` 以啟用隨機探索策略。

## 3. `train.py`（訓練腳本，**可自由調整**）
- **用途**：使用 SB3 在本地訓練 Agent，產生 `model.zip` 權重檔。
- **可改**：超參數（`TOTAL_TIMESTEPS`、`N_ENVS`）。
- **執行**：
  ```bash
  python train.py
  ```

## 4. `run.py`（提交腳本）
- **用途**：將 `agent.py`、`model.py`、`model.zip` 上傳至 MLArena 伺服器。
- **自動嵌入**：`model.py` 會自動被嵌入 `agent.py`，沙箱可直接執行。
- **執行前**：修改 `STUDENT_ID`、`SLOT_INDEX`、`SLOT_NAME`、`DESCRIPTION`。
- **執行**：
  ```bash
  python run.py
  ```

## 5. `environment.yml` 與 `requirements.txt`（Conda 環境與套件設定檔）
請確保先使用終端機進入作業包目錄下（即 `student` 資料夾），再執行以下環境建立指令：
```bash
conda env create -f environment.yml
conda activate mlarena
pip install -r requirements.txt
```

## 6. `arena_client.py`（系統客戶端，通常不需修改）

---

## 執行流程

```
1. 開啟終端機，並使用 cd 命令進入解壓後的作業包目錄（即含有 environment.yml 的資料夾）：
   cd <作業包目錄路徑>

   # 建立 Conda 環境、啟用環境並安裝依賴套件
   conda env create -f environment.yml
   conda activate mlarena
   pip install -r requirements.txt

2. python train.py          → 產生 model.zip
3. 修改 run.py 的 STUDENT_ID
4. python run.py            → 上傳至競賽平台
```

## 觀測值與動作說明

| 項目 | 說明 |
|------|------|
| 觀測值 | `numpy.ndarray` shape `(128,)` uint8 — Atari RAM 狀態 |
| 動作空間 | `Discrete(9)`：0=停止, 1=上, 2=右, 3=左, 4=下, 5~8=斜向 |
| 評分 | 多局平均總分，越高越好 |
| 演算法 | 預設 DQN；可換成 PPO、A2C（修改 `model.py`） |

---

## 如何訓練更強的 Agent

### 1. Reward Shaping（獎勵塑形）

Atari Pacman 的原始分數（吃豆、吃鬼）作為 reward 已有基本引導，但可加入更精細的中間步訊號：

| 策略 | 實作位置 | 說明 |
|------|----------|------|
| 死亡懲罰 | `train.py` 技巧 1（Wrapper） | 偵測生命數下降給予大懲罰，強化存活意識 |
| 每步小懲罰 | `train.py` 技巧 1（Wrapper） | `-0.01` 每步，鼓勵快速吃豆而非原地停留 |

取消 `train.py` 中 `PacmanRewardWrapper` 注釋，並在 `make_vec_env` 加入 `wrapper_class=PacmanRewardWrapper` 即可啟用。

### 2. Frame Stacking（時序資訊）

Pacman 的鬼魂移動方向需要時序資訊才能判斷。Frame Stacking 將最近 N 幀疊在一起當作觀測：

取消 `train.py` 中 Frame Stacking 注釋（技巧 2），觀測維度從 128 變為 512（4 幀 × 128）。同步在 `model.py` 調大 `net_arch`（如 `[256, 256]`）。

### 3. 超參數調整

| 參數 | 預設值（DQN） | 效果 | 建議嘗試值 |
|------|--------------|------|-----------|
| `learning_rate` | `1e-4` | 過大→震盪，過小→收斂慢 | `5e-5` ~ `5e-4` |
| `buffer_size` | `1_000_000` | 越大→樣本多樣性越高 | 依記憶體調整 |
| `exploration_fraction` | `0.1` | 越大→探索期越長 | `0.1` ~ `0.2` |
| `TOTAL_TIMESTEPS` | `100_000` | Pacman 需大量步數 | `1_000_000`+ |

> ⚠️ Pacman 對 `TOTAL_TIMESTEPS` 特別敏感，100k 步幾乎看不到有意義的策略，建議從 **500k** 起步。

詳細參數說明見 `model.py` 末尾注釋。

### 4. 演算法選擇與網路結構

- **DQN**（預設）：適合動作空間小、離散的問題；Pacman 是經典應用場景
- **PPO / A2C**：策略梯度方法，在某些 Atari 遊戲表現更好，可試著切換
- 修改 `model.py` 的 `ALGORITHM` 即可切換，詳見該檔案末尾注釋

> 改演算法後務必重新執行 `train.py`，不同演算法產生的 `model.zip` 不能混用。
