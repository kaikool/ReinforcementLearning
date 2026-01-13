import numpy as np
from collections import deque
from core.feature_factory import QuantFeatureFactory

# Single Source of Truth for Feature Selection
# Đảm bảo nhất quán giữa Training (Batch) và Inference (Online)
FEATURE_COLUMNS = [
    'log_ret',
    'realized_vol',    # Độ lớn biến động (Magnitude)
    'rsi_raw',
    'macd_raw',
    'cci_raw',
    'atr_rel',         # Phạm vi biến động (Range)
    'bb_pct',          # Vị trí giá tương đối (Location)
    'vol_rel',         # Đại diện thanh khoản (Liquidity Proxy)
    'is_active_hour',
    'entropy_raw',
    'efficiency_raw',
    'hurst',                 # Chế độ thị trường liên tục (Continuous Regime)
    'entropy_delta',         # Proxy sự kiện (Confusion Shock)
    'price_shock',           # Proxy sự kiện (Biến động giá bất ngờ)
    'interaction_trend_vol',
    'trend_efficiency',      # Alpha Proxy (New)
    'directional_persistence', # Win-rate Proxy (New)
    'vol_scalar',      # Hệ số scale vị thế
    'vol_ratio',       # Tỷ lệ biến động tương đối (Current/Target)
    'last_cost_pct',   # Chi phí giao dịch trước đó (% Equity)
    'last_vol_scalar',  # Context quy mô vị thế trước đó
]

class StateBuilder:
    """
    Module lắp ráp Vector Quan sát (Observation Tensor) cuối cùng.
    Refactored: Single Source of Truth, Consistent Logic, No Hidden Calculations.
    """
    
    def __init__(self, window_size=64, n_features=None):
        self.window_size = window_size
        self.n_features = n_features
        
        # Online Mode utilities
        self.ff = QuantFeatureFactory()
        self.history = deque(maxlen=window_size)
    
    @property
    def feature_names(self):
        """Trả về danh sách tên các đặc trưng theo đúng thứ tự trong vector observation."""
        names = list(FEATURE_COLUMNS)
        # Macro Regime (3 components)
        names.extend(['regime_0', 'regime_1', 'regime_2'])
        # Portfolio / Context States
        names.extend(['unrealized_pnl_norm', 'time_in_trade_norm', 'position_weight'])
        return names
    
    def reset(self):
        """Đặt lại trạng thái cho Chế độ thực thi."""
        self.ff.reset()
        self.history.clear()
        
    def process_one_step(self, 
                         price: float, 
                         volume: float, 
                         indicators: dict, 
                         regime_probs: np.ndarray, 
                         portfolio_state: dict,
                         perform_normalization: bool = True,
                         frozen: bool = False) -> np.ndarray:
        """
        [Chế độ Online/Step-by-Step] Tạo vector trạng thái cho 1 bước.
        """
        features = []
        
        # 2. Xây dựng Feature Vector dựa trên FEATURE_COLUMNS
        for col in FEATURE_COLUMNS:
            # Ưu tiên lấy từ Market Data, sau đó đến Portfolio State (Tránh trùng lặp logic)
            val = indicators.get(col)
            if val is None:
                val = portfolio_state.get(col)
                
            if val is None:
                 raise RuntimeError(f"CRITICAL: Feature '{col}' missing! Check Env.")

            if perform_normalization:
                # Online: Chuẩn hóa trực tiếp
                norm_val = self.ff.update_and_normalize(col, val, frozen=frozen)
                features.append(norm_val)
            else:
                # Batch: Value đã được normalize trước đó
                features.append(val)
                
        # 3. Macro Regime (Giữ nguyên xác suất thô -> Scale up)
        # FIX: Scale Regime Probs to match Z-Score magnitude (~O(2.0))
        features.extend(regime_probs * 2.0)
        
        # 4. Trạng thái Danh mục (Portfolio State)
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        equity = portfolio_state.get('equity', 10000.0)
        norm_pnl = np.clip(unrealized_pnl / (equity + 1e-9), -1.0, 1.0)
        features.append(norm_pnl) 
        
        # FIX: Single Source Normalization (User Request)
        # Value passed in is already normalized in Env (min(steps/1000, 1.0))
        norm_time = portfolio_state.get('time_in_trade', 0.0)
        features.append(norm_time)
        
        pos = portfolio_state.get('position', 0)
        features.append(float(pos))
        
        # Các tính năng mới (Action Threshold) đã được xử lý trong FEATURE_COLUMNS nếu có

        
        obs_vector = np.array(features, dtype=np.float32)
        
        # Cập nhật bộ đệm lịch sử
        # Lưu ý: Việc Frame Stacking được xử lý bên ngoài bởi SB3 Wrapper.
        self.history.append(obs_vector)
        
        return obs_vector

    def build_state(self, market_history, regime_probs, account_state):
        """
        [Chế độ Batch - Experimental] Dùng để assemble state nhanh khi training.
        Tuy nhiên, với LSTM và Gym Env, ta thường dùng Step-based (process_one_step).
        Hàm này giữ lại để tham khảo hoặc dùng cho Feed-forward models.
        """
        # Select đúng các cột Features
        try:
            market_tensor = market_history[FEATURE_COLUMNS].values
        except KeyError as e:
            # Fallback nếu thiếu cột (ví dụ environment cũ)
            print(f"KeyError in build_state: {e}. Using all values.")
            market_tensor = market_history.values
        
        # Meta features
        norm_pnl = np.clip(account_state['unrealized_pnl'] / (account_state['equity'] + 1e-9), -1.0, 1.0)
        norm_time = min(account_state['time_in_trade'] / 100.0, 1.0)
        pos_weight = float(account_state['position'])
        
        meta_vec = np.concatenate([regime_probs, [norm_pnl, norm_time, pos_weight]])
        
        # Broadcast
        meta_tensor = np.tile(meta_vec, (len(market_tensor), 1))
        
        full_state = np.hstack([market_tensor, meta_tensor])
        
        return full_state.astype(np.float32)
