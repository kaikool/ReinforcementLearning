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

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Các Module cốt lõi
from core.feature_factory import QuantFeatureFactory
from core.regime import MarketRegime
from core.trading_env import AdvancedTradingEnv

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# --- CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS) ---

def linear_schedule(initial_value: float):
    """Lập lịch thay đổi giá trị theo đường thẳng (Ví dụ: Giảm dần Learning Rate)."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class EntropyCoefDecayCallback(BaseCallback):
    """
    Callback để giảm dần hệ số Entropy (Entropy Coefficient).
    Giúp Agent khám phá mạnh ở đầu và ổn định dần ở cuối quá trình huấn luyện.
    """
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
    """
    Hệ thống Logging Đa tầng (Quant-Grade).
    Tách biệt: Episode Stats, Step Stats (Sparse), RL Health, Reward Decomposition.
    Tuân thủ nguyên tắc: CSV = FACT, HTML = VIEW.
    """
    def __init__(self, base_path: str, save_freq: int = 2048):
        super().__init__()
        self.base_path = base_path
        self.save_freq = save_freq
        
        # Paths
        self.paths = {
            "episode": os.path.join(base_path, "episode_stats.csv"),
            "step": os.path.join(base_path, "step_stats.csv"),
            "health": os.path.join(base_path, "rl_health.csv"),
            "reward": os.path.join(base_path, "reward_decomp.csv")
        }
        
        # Khởi tạo Episode ID (Duy trì khi Resume)
        self.episode_count = self._get_last_episode_id()
        
    def _get_last_episode_id(self) -> int:
        """Đọc episode rốt cuộc từ CSV để không bị reset khi Resume."""
        try:
            if os.path.exists(self.paths["episode"]):
                df = pd.read_csv(self.paths["episode"])
                if not df.empty:
                    return int(df["episode"].max())
        except Exception: pass
        return 0

    def _on_step(self) -> bool:
        # 1. Step-level Core (Sparse logging)
        info = self.locals['infos'][0]
        if self.num_timesteps % 100 == 0:
            step_fact = {
                "global_step": self.num_timesteps,
                "episode_step": info.get('episode_step', 0),
                "price": info.get('price', 0),
                "target_weight": info.get('weight', 0),
                "weight_delta": info.get('weight_delta', 0),
                "net_worth": info.get('net_worth', 0),
                "vol_ann": info.get('regime_stats', {}).get('vol', 0),
                "hurst": info.get('regime_stats', {}).get('hurst', 0.5),
                "efficiency": info.get('regime_stats', {}).get('efficiency', 0),
                "persistence": info.get('regime_stats', {}).get('persistence', 0),
                "step_reward": info.get('reward_decomposition', {}).get('total', 0), # Total Clipped Reward
                "action_logit": info.get('action_logit', 0), 
                "win_rate": info.get('win_rate', 0),
                "profit_factor": info.get('profit_factor', 0),
                "max_drawdown": info.get('max_drawdown', 0),
                "episode": self.episode_count # Thêm để Dashboard phân tách các Episode
            }
            # self._flush("step", [step_fact]) -> Moved below
            
            # Record Reward Decomp
            rd = info.get('reward_decomposition', {})
            rd_fact = {
                "global_step": self.num_timesteps,
                "pnl_reward": rd.get('pnl_reward', 0),
                "dd_penalty": rd.get('dd_penalty', 0),
                "vol_scaling": rd.get('vol_scaling', 1.0),
                "edge_score": info.get('edge_score', 0)
            }
            self._flush("reward", [rd_fact])

            step_fact["step_reward"] = rd.get('total', 0)
            self._flush("step", [step_fact])

        # 2. Episode-level Metrics
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
                "profit_factor": info.get('profit_factor', 0),
                "avg_exposure": info.get('avg_exposure', 0),
                "num_trades": info.get('num_trades', 0)
            }
            self._flush("episode", [ep_fact])

        # 3. RL Health (Frequency: every 512 steps for better granularity)
        if self.num_timesteps % 512 == 0:
            # FIX: SB3 Logger không có get_log_dict(), dùng name_to_value
            logs = getattr(self.model.logger, 'name_to_value', {})
            health_fact = {
                "global_step": self.num_timesteps,
                "entropy": abs(logs.get("train/entropy", logs.get("train/entropy_loss", 0))),
                "kl": logs.get("train/approx_kl", 0),
                "v_loss": logs.get("train/value_loss", logs.get("train/loss", 0)),
                "lr": logs.get("train/learning_rate", 0),
                "clip": logs.get("train/clip_fraction", 0),
                "explained_var": logs.get("train/explained_variance", 0)
            }
            self._flush("health", [health_fact])
            
        return True

    def _flush(self, key, data_list):
        """Append mode logging (FACT persistence)."""
        if not data_list: return
        file_path = self.paths[key]
        header = not os.path.exists(file_path)
        pd.DataFrame(data_list).to_csv(file_path, mode='a', index=False, header=header)

def get_next_version_dir(base_dir="artifacts"):
    """Tự động tạo và trả về thư mục phiên bản tiếp theo (ví dụ: version1, version2...)."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing_dirs = glob.glob(os.path.join(base_dir, "version*"))
    max_version = 0
    for d in existing_dirs:
        match = re.search(r"version(\d+)", os.path.basename(d))
        if match:
            v = int(match.group(1))
            if v > max_version: max_version = v
    new_dir = os.path.join(base_dir, f"version{max_version + 1}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def calculate_metrics(equity_curve):
    """Tính toán các chỉ số cơ bản từ đường cong vốn (Equity Curve)."""
    if not equity_curve: return {}
    equity = np.array(equity_curve)
    initial = equity[0]
    final = equity[-1]
    total_return_pct = (final - initial) / initial * 100
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd_pct = np.max(drawdown) * 100
    return {
        "Lợi nhuận (%)": total_return_pct,
        "Sụt giảm tối đa (%)": max_dd_pct,
        "Vốn cuối cùng": final
    }

def select_file(pattern: str):
    """Tự động chọn file dữ liệu phù hợp nhất."""
    files = glob.glob(pattern)
    files.sort(reverse=True) # Ưu tiên các file mới nhất hoặc có tên dài nhất
    if not files:
        print(f"❌ Không tìm thấy file nào khớp với mẫu: {pattern}")
        return None
    print(f"   [Auto] Phát hiện {len(files)} file, tự động chọn: {files[0]}")
    return files[0]

# --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---

def main():
    print("=== NEXT-GEN TRAINING ENGINE (HỆ THỐNG HUẤN LUYỆN THẾ HỆ MỚI) ===")
    
    # 1. Thiết lập thư mục lưu trữ
    artifact_dir = get_next_version_dir()
    
    # 2. Chọn dữ liệu huấn luyện
    train_file = select_file("data/train_*.csv")
    if not train_file: return
    
    # Tự động cấu hình theo cặp tài sản (Symbol Config)
    # Tự động cấu hình theo cặp tài sản (Symbol Config)
    symbol_conf = {
        "spread_usd": 0.30, "commission_usd": 0.0, "lot_size": 100000.0
    }
    if "XAUUSD" in train_file:
        print(">> Cấu hình nhận diện: XAUUSD (Vàng)")
        symbol_conf = {
            "lot_size": 100.0, 
            # Fixed Spread USD for Gold ($0.30 per oz)
            "spread_usd": 0.30, "commission_usd": 0.0
        }
        
    # 3. Tải và Chia tách dữ liệu
    print(f"Đang tải dữ liệu: {train_file}")
    df_raw = pd.read_csv(train_file)
    
    # Chuẩn hóa tiêu đề các cột (Normalize Headers)
    df_head = pd.read_csv(train_file, nrows=1)
    time_col = "Time (EET)" if "Time (EET)" in df_head.columns else "Gmt time"
    rename_map = {time_col: 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df_raw.rename(columns=rename_map, inplace=True)
    
    # 4. Feature Engineering
    print(">> Tính toán chỉ báo kỹ thuật liên tục...")
    factory = QuantFeatureFactory()
    
    # [LỖI FIX] Chuyển 'date' thành index trước để khớp với all_features sau burn-in
    df_raw['date'] = pd.to_datetime(df_raw['date'], dayfirst=True, format='mixed')
    df_raw.set_index('date', inplace=True)
    df_raw.sort_index(inplace=True)
    
    all_features = factory.process_data(df_raw)
    
    # Chia lại tập dữ liệu sau Burn-in (Tận dụng index để đồng bộ hoàn toàn)
    df_raw = df_raw.loc[all_features.index].copy()
    
    # Re-calculate split based on processed data length
    new_split_idx = int(len(all_features) * 0.8)
    
    train_df = df_raw.iloc[:new_split_idx].copy().reset_index(drop=True)
    test_df = df_raw.iloc[new_split_idx:].copy().reset_index(drop=True)
    
    train_features = all_features.iloc[:new_split_idx].copy().reset_index(drop=True)
    test_features = all_features.iloc[new_split_idx:].copy().reset_index(drop=True)
    
    train_factory = factory 
    test_factory = factory
    
    # Lưu thống kê chuẩn hóa cho Backtest
    factory_stats_path = os.path.join(artifact_dir, "factory_stats.pkl")
    factory.save_stats(factory_stats_path)
    
    # Lấy Multi-variate Data để train Regime Model (Risk Aware)
    # Cần: log_ret (Direction), realized_vol (Magnitude), entropy_raw (Structure)
    # FIX: Add abs(log_ret) for Expectancy Pass (Distinguish direction strength)
    train_features['abs_log_ret'] = train_features['log_ret'].abs()
    test_features['abs_log_ret'] = test_features['log_ret'].abs()
    
    regime_cols = ['log_ret', 'realized_vol', 'entropy_raw', 'abs_log_ret']
    train_regime_data = train_features[regime_cols].fillna(0.0).values
    test_regime_data = test_features[regime_cols].fillna(0.0).values

    # Huấn luyện mô hình trạng thái thị trường (Regime)
    regime_model_path = os.path.join(artifact_dir, "regime_model.pkl")
    regime_model = MarketRegime(n_components=3, n_iter=100, model_path=regime_model_path)
    
    # Cấu hình Warm-up cho HMM (Chống Leakage)
    warmup_size = min(3000, int(len(train_regime_data) * 0.3))
    
    if len(train_regime_data) > 500 and warmup_size > 200:
        print(f">> Khởi động HMM & Scaler trên {warmup_size} nến Warm-up...")
        warmup_data = train_regime_data[:warmup_size]
        regime_model.fit(warmup_data)
        
        # Agent sẽ học trên dữ liệu sau Warm-up
        train_features = train_features.iloc[warmup_size:].reset_index(drop=True)
        train_df = train_df.iloc[warmup_size:].reset_index(drop=True)

        

            
    if len(train_regime_data) > 1000:
        # NOTE: Already fitted on Warmup or Full Train Analysis
        # If not fitted yet (no warmup), fit now
        if not regime_model.is_fitted:
             print(">> Training Regime Model (Offline Loop) on Full Train Data...")
             regime_model.fit(train_regime_data)
             
        # FIX: Use Causal Filter for Training Data too (Avoid Observation Leakage)
        train_probs = regime_model.predict_proba_causal(train_regime_data)
    else:
        print(">> Warning: Không đủ dữ liệu để train HMM.")
        train_probs = np.ones((len(train_regime_data), 3)) / 3
        regime_model.is_fitted = False
    
    # Test: Predict OOS (Dùng model cuối cùng từ tập train)
    if regime_model.is_fitted:
        # FIX: Use Causal Filter for Test Data
        test_probs = regime_model.predict_proba_causal(test_regime_data)
    else:
        test_probs = np.ones((len(test_regime_data), 3)) / 3
    
    print(">> Khởi tạo Môi trường & Mạng thần kinh...")

    # 5. Nhà máy tạo môi trường (Environment Factory)
    def make_env_train():
        return AdvancedTradingEnv(
            df=train_df, 
            feature_factory=train_factory,  
            regime_model=regime_model,
            window_size=64,
            initial_equity=10000,
            spread_usd=symbol_conf["spread_usd"],
            commission_usd=symbol_conf["commission_usd"],
            lot_size=symbol_conf["lot_size"],
            # Tối ưu hóa: Truyền dữ liệu đã tính toán trước
            df_features=train_features,
            regime_probs=train_probs,
            is_training=True # Bật Noise
        )
        
    def make_env_test():
        return AdvancedTradingEnv(
            df=test_df,
            feature_factory=test_factory,
            regime_model=regime_model,
            window_size=64,
            initial_equity=10000,
            spread_usd=symbol_conf["spread_usd"],
            commission_usd=symbol_conf["commission_usd"],
            lot_size=symbol_conf["lot_size"],
            # Tối ưu hóa: Truyền dữ liệu đã tính toán trước
            df_features=test_features,
            regime_probs=test_probs,
            is_training=False # Tắt Noise
        )

    # Khởi tạo Vectorized Env & Normalization
    print(">> Đang khởi tạo môi trường (2 Envs) với VecNormalize...")
    train_vec = DummyVecEnv([make_env_train for _ in range(2)])
    
    train_vec = VecNormalize(
        train_vec, 
        norm_obs=True, 
        norm_reward=False, # Tắt để giữ tính ổn định cho Log-Return
        clip_obs=10.0, 
        clip_reward=10.0,
        gamma=0.995
    )
    
    train_vec = VecMonitor(
        train_vec, 
        filename=os.path.join(artifact_dir, "monitor.csv"),
        info_keywords=("net_worth", "pnl")
    )
    
    eval_vec = DummyVecEnv([make_env_test])
    # Eval dùng cùng tham số chuẩn hóa (nhưng không cập nhật stats)
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, training=False)
    
    # 7. Thiết lập Mô hình RL (Recurrent PPO - New Train)
    print(">> Đang khởi tạo mô hình RecurrentPPO (New Train - Optimized Params)...")
    # [XUẤT PHÁT THẤP - XÁC THỰC DAMPING]
    total_timesteps = 3_000_000
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_vec,
        verbose=1, 
        learning_rate=linear_schedule(3e-4), # Giảm nhẹ LR để học kỹ hơn
        n_steps=1024, # Tăng lên để LSTM nhìn xa hơn
        batch_size=512, 
        gamma=0.99, # Thay đổi nhẹ để cân bằng hiện tại/tương lai
        gae_lambda=0.95,
        clip_range=0.2, 
        ent_coef=0.02, # Tăng nhẹ để tránh kẹt vào local optima sớm
        policy_kwargs={
            "enable_critic_lstm": True,
            "lstm_hidden_size": 256,
            "n_lstm_layers": 2,
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        },
        tensorboard_log=os.path.join(artifact_dir, "logs")
    )
    
    # 8. Huấn luyện (Train)
    dash_dir = os.path.join("dashboard", "live_stats")
    # Xóa dữ liệu cũ nếu có để biểu đồ sạch sẽ
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
        print(f">> BẮT ĐẦU HUẤN LUYỆN NEW TRAIN (Mục tiêu: {total_timesteps/1e6:.1f} triệu bước)...")
        print(f">> Dashboard FACT logs: {dash_dir}")
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
    except KeyboardInterrupt:
        print("\nTiến trình bị gián đoạn bởi Người dùng.")
        
    # 9. Lưu trữ và Đánh giá (Save & Evaluate)
    final_path = os.path.join(artifact_dir, "final_model")
    model.save(final_path)
    print(f"Đã lưu mô hình cuối cùng tại: {final_path}")
    
    print(">> Đang chạy đánh giá chi tiết trên tập Test...")
    # Logic đánh giá nhanh
    obs = eval_vec.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    equity = []
    
    # Chạy trọn vẹn một episode test
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, infos = eval_vec.step(action)
        equity.append(infos[0]['net_worth'])
        if dones[0]: break
        
    metrics = calculate_metrics(equity)
    print("Kết quả đánh giá:", metrics)
    
    # Vẽ biểu đồ kết quả
    plt.figure(figsize=(10, 6))
    plt.plot(equity)
    plt.title(f"Evaluation: Return {metrics.get('Lợi nhuận (%)', 0):.2f}%")
    plt.savefig(os.path.join(artifact_dir, "eval_curve.png"))
    print(f"Biểu đồ đánh giá đã được lưu tại: {os.path.join(artifact_dir, 'eval_curve.png')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n\nLỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH THI HÀNH CHÍNH:")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)
        sys.exit(1)
