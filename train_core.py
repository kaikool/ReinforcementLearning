import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
import traceback
import sys
from typing import Tuple, Dict, Any, Optional

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Các Module cốt lõi
from core.feature_factory import QuantFeatureFactory
from core.regime import MarketRegime
from core.trading_env import AdvancedTradingEnv, EnvConfig

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# --- CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS) ---

def linear_schedule(initial_value: float):
    """Lập lịch thay đổi giá trị theo đường thẳng."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class EntropyCoefDecayCallback(BaseCallback):
    """Callback để giảm dần hệ số Entropy."""
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        progress = self.step_count / self.total_timesteps
        current_ent_coef = self.initial_ent_coef + (self.final_ent_coef - self.initial_ent_coef) * progress
        current_ent_coef = max(current_ent_coef, self.final_ent_coef)
        self.model.ent_coef = current_ent_coef
        if self.step_count % 10000 == 0:
            self.logger.record("train/ent_coef", current_ent_coef)
        return True

class CustomLoggingCallback(BaseCallback):
    """Hệ thống Logging Đa tầng (Quant-Grade)."""
    def __init__(self, base_path: str, save_freq: int = 2048):
        super().__init__()
        self.base_path = base_path
        self.save_freq = save_freq
        self.paths = {
            "episode": os.path.join(base_path, "episode_stats.csv"),
            "step": os.path.join(base_path, "step_stats.csv"),
            "health": os.path.join(base_path, "rl_health.csv"),
            "reward": os.path.join(base_path, "reward_decomp.csv")
        }
        self.episode_count = self._get_last_episode_id()
        
    def _get_last_episode_id(self) -> int:
        try:
            if os.path.exists(self.paths["episode"]):
                df = pd.read_csv(self.paths["episode"])
                if not df.empty: return int(df["episode"].max())
        except Exception: pass
        return 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if self.num_timesteps % 100 == 0:
            step_fact = {
                "global_step": self.num_timesteps,
                "episode_step": info.get('episode_step', 0),
                "price": info.get('price', 0),
                "target_weight": info.get('weight', 0),
                "net_worth": info.get('net_worth', 0),
                "vol_ann": info.get('regime_stats', {}).get('vol', 0),
                "hurst": info.get('regime_stats', {}).get('hurst', 0.5),
                "efficiency": info.get('regime_stats', {}).get('efficiency', 0),
                "persistence": info.get('regime_stats', {}).get('persistence', 0),
                "step_reward": info.get('reward_decomposition', {}).get('total', 0),
                "action_logit": info.get('action_logit', 0), 
                "win_rate": info.get('win_rate', 0),
                "num_trades": info.get('num_trades', 0),
                "weight_delta": info.get('weight_delta', 0),
                "max_drawdown": info.get('max_drawdown', 0),
                "episode": self.episode_count
            }
            rd = info.get('reward_decomposition', {})
            rd_fact = {
                "global_step": self.num_timesteps,
                "pnl_reward": rd.get('pnl_reward', 0),
                "dd_penalty": rd.get('dd_penalty', 0),
                "vol_scaling": rd.get('vol_scaling', 1.0),
                "edge_score": info.get('edge_score', 0)
            }
            self._flush("reward", [rd_fact])
            self._flush("step", [step_fact])

        if 'episode' in info:
            self.episode_count += 1
            ep_info = info['episode']
            ep_fact = {
                "episode": self.episode_count,
                "episode_length": ep_info['l'],
                "total_reward": ep_info['r'],
                "final_equity": info.get('net_worth', 0),
                "max_drawdown": info.get('max_drawdown', 0),
                "win_rate": info.get('win_rate', 0),
                "num_trades": info.get('num_trades', 0)
            }
            self._flush("episode", [ep_fact])

        if self.num_timesteps % 512 == 0:
            logs = getattr(self.model.logger, 'name_to_value', {})
            health_fact = {
                "global_step": self.num_timesteps,
                "entropy": abs(logs.get("train/entropy", logs.get("train/entropy_loss", 0))),
                "kl": logs.get("train/approx_kl", 0),
                "v_loss": logs.get("train/value_loss", logs.get("train/loss", 0)),
                "lr": logs.get("train/learning_rate", 0),
                "explained_var": logs.get("train/explained_variance", 0)
            }
            self._flush("health", [health_fact])
        return True

    def _flush(self, key, data_list):
        if not data_list: return
        file_path = self.paths[key]
        header = not os.path.exists(file_path)
        pd.DataFrame(data_list).to_csv(file_path, mode='a', index=False, header=header)

def get_next_version_dir(base_dir="artifacts"):
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    existing_dirs = glob.glob(os.path.join(base_dir, "version*"))
    max_version = 0
    for d in existing_dirs:
        match = re.search(r"version(\d+)", os.path.basename(d))
        if match: max_version = max(max_version, int(match.group(1)))
    new_dir = os.path.join(base_dir, f"version{max_version + 1}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def select_file(pattern: str):
    files = glob.glob(pattern)
    files.sort(reverse=True)
    if not files: return None
    return files[0]

# --- PIPELINE SUB-FUNCTIONS (MAIN LOGIC) ---

def prepare_data(train_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tải dữ liệu, tính toán đặc trưng và chia tách Train/Val/Test (70/15/15)."""
    df_raw = pd.read_csv(train_file)
    df_head = pd.read_csv(train_file, nrows=1)
    time_col = "Time (EET)" if "Time (EET)" in df_head.columns else "Gmt time"
    df_raw.rename(columns={time_col: 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    df_raw['date'] = pd.to_datetime(df_raw['date'], dayfirst=True, format='mixed')
    df_raw.set_index('date', inplace=True)
    df_raw.sort_index(inplace=True)
    
    factory = QuantFeatureFactory()
    all_features = factory.process_data(df_raw, normalize=False)
    df_raw = df_raw.loc[all_features.index].copy()
    
    # Split: 70% Train, 15% Val, 15% Test
    n = len(all_features)
    val_idx = int(n * 0.70)
    test_idx = int(n * 0.85)
    
    train_df = df_raw.iloc[:val_idx].copy().reset_index(drop=True)
    val_df = df_raw.iloc[val_idx:test_idx].copy().reset_index(drop=True)
    test_df = df_raw.iloc[test_idx:].copy().reset_index(drop=True)
    
    train_features = all_features.iloc[:val_idx].copy().reset_index(drop=True)
    val_features = all_features.iloc[val_idx:test_idx].copy().reset_index(drop=True)
    test_features = all_features.iloc[test_idx:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df, train_features, val_features, test_features, factory

def prepare_regime_model(train_features, artifact_dir) -> Tuple[MarketRegime, np.ndarray]:
    """Huấn luyện mô hình Regime với cơ chế Warmup được refactor chuẩn xác."""
    regime_model_path = os.path.join(artifact_dir, "regime_model.pkl")
    regime_model = MarketRegime(n_components=3, n_iter=100, model_path=regime_model_path)
    
    # Cấu hình đặc trưng cho HMM (Regime Detection)
    # log_ret: Hướng giá, realized_vol: Độ biến động, entropy_raw: Độ nhiễu
    regime_cols = ['log_ret', 'realized_vol', 'entropy_raw'] 
    
    # abs_log_ret (Cường độ nến): Dù có tương quan với realized_vol nhưng giúp HMM phân tách 
    # các trạng thái 'Shock' (biến động đột ngột) tốt hơn so với SMA-Vol.
    regime_cols_with_abs = regime_cols + ['abs_log_ret']
    
    for df in [train_features]: 
        if 'abs_log_ret' not in df.columns: 
            df['abs_log_ret'] = df['log_ret'].abs()
    
    # Warmup Logic Refactored: Fit on warmup ONLY if available, otherwise fit on all
    warmup_size = min(3000, int(len(train_features) * 0.3))
    
    if len(train_features) > 1000 and warmup_size > 500:
        print(f">> Khởi động HMM trên {warmup_size} nến Warm-up...")
        full_regime_data = train_features[regime_cols_with_abs].fillna(0.0).values
        warmup_data = full_regime_data[:warmup_size]
        regime_model.fit(warmup_data)
        
        # Remove warmup from future training steps
        train_features_sliced = train_features.iloc[warmup_size:].reset_index(drop=True)
        train_regime_data = train_features_sliced[regime_cols_with_abs].fillna(0.0).values
    else:
        print(">> No warmup: Training Regime Model on Full Data...")
        train_regime_data = train_features[regime_cols_with_abs].fillna(0.0).values
        regime_model.fit(train_regime_data)
        train_features_sliced = train_features

    train_probs = regime_model.predict_proba_causal(train_regime_data)
    return regime_model, train_probs, train_features_sliced

def run_train():
    print("=== NEXT-GEN TRAINING ENGINE V9.1 (OPTIMIZED) ===")
    artifact_dir = get_next_version_dir()
    train_file = select_file("data/train_*.csv")
    if not train_file: return
    
    # 1. Config & Data
    symbol_conf = {"lot_size": 100.0, "spread_usd": 0.30, "commission_usd": 0.0}
    train_df, val_df, test_df, train_f, val_f, test_f, factory = prepare_data(train_file)
    factory.save_stats(os.path.join(artifact_dir, "factory_stats.pkl"))
    
    # 2. Regime Model (HMM)
    regime_model, train_probs, train_f_final = prepare_regime_model(train_f, artifact_dir)
    
    # Prep Val/Test Probs
    val_probs = regime_model.predict_proba_causal(val_f.assign(abs_log_ret=val_f['log_ret'].abs())[['log_ret', 'realized_vol', 'entropy_raw', 'abs_log_ret']].fillna(0.0).values)
    test_probs = regime_model.predict_proba_causal(test_f.assign(abs_log_ret=test_f['log_ret'].abs())[['log_ret', 'realized_vol', 'entropy_raw', 'abs_log_ret']].fillna(0.0).values)
    
    # 3. Environments
    env_config = EnvConfig() # Default V9 config
    
    def make_env(df, feat, probs, is_train):
        return AdvancedTradingEnv(df=df, feature_factory=factory, regime_model=regime_model, config=env_config,
                                 lot_size=symbol_conf["lot_size"], spread_usd=symbol_conf["spread_usd"],
                                 df_features=feat, regime_probs=probs, is_training=is_train)

    # Increased to 4 Envs for better diversity
    print(">> Khởi tạo 4 Môi trường huấn luyện...")
    train_vec = DummyVecEnv([lambda: make_env(train_df.iloc[-len(train_f_final):].reset_index(drop=True), train_f_final, train_probs, True) for _ in range(4)])
    train_vec = VecNormalize(train_vec, norm_obs=False, norm_reward=False)
    train_vec = VecMonitor(train_vec, filename=os.path.join(artifact_dir, "monitor.csv"))
    
    val_vec = DummyVecEnv([lambda: make_env(val_df, val_f, val_probs, False)])
    val_vec = VecNormalize(val_vec, norm_obs=False, norm_reward=False, training=False)
    
    # 4. Model (RecurrentPPO)
    print(">> Khởi tạo mô hình RecurrentPPO...")
    total_timesteps = 3_000_000
    model = RecurrentPPO("MlpLstmPolicy", train_vec, verbose=1, learning_rate=linear_schedule(3e-4),
                        n_steps=1024, batch_size=512, ent_coef=0.02, policy_kwargs={"enable_critic_lstm": True, "lstm_hidden_size": 256},
                        tensorboard_log=os.path.join(artifact_dir, "logs"))
    
    # 5. Train
    dash_dir = os.path.join("dashboard", "live_stats")
    for f in glob.glob(os.path.join(dash_dir, "*.csv")): 
        try: os.remove(f)
        except: pass
    os.makedirs(dash_dir, exist_ok=True)
    
    callbacks = [
        CheckpointCallback(save_freq=50000, save_path=os.path.join(artifact_dir, "checkpoints"), name_prefix="core_agent"),
        EntropyCoefDecayCallback(0.02, 0.001, total_timesteps), 
        CustomLoggingCallback(dash_dir, save_freq=1024) 
    ]
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\nInterrupt. Saving...")
    
    model.save(os.path.join(artifact_dir, "final_model"))
    print(f"Bản lưu cuối: {artifact_dir}")

if __name__ == "__main__":
    try: run_train()
    except Exception: traceback.print_exc(); sys.exit(1)
