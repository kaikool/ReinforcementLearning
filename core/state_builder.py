import numpy as np
from collections import deque
from core.feature_factory import QuantFeatureFactory

# Single Source of Truth for Feature Selection
FEATURE_COLUMNS = [
    'log_ret',
    'realized_vol',    # Magnitude
    'rsi_raw',
    'macd_raw',
    'cci_raw',
    'atr_rel',         # Range
    'bb_pct',          # Location
    'vol_rel',         # Liquidity Proxy
    'is_active_hour',
    'entropy_raw',
    'efficiency_raw',
    'hurst',                 # Continuous Regime
    'entropy_delta',         # Confusion Shock
    'price_shock',           # Proxy sự kiện
    'interaction_trend_vol',
    'trend_efficiency',      # Alpha Proxy
    'directional_persistence', # Win-rate Proxy
    'vol_scalar',      # Hệ số scale vị thế
    'vol_ratio',       # Tỷ lệ biến động tương đối
    'last_cost_pct',   # Chi phí giao dịch trước đó
    'last_vol_scalar',  # Context quy mô vị thế trước đó
]

class StateBuilder:
    """
    Module lắp ráp Vector Quan sát (StateBuilder V8 - Continuous Scales).
    Đảm bảo tính nhất quán về dải giá trị giữa Market Features và Portfolio Features.
    """
    
    # --- Operational Constants ---
    REGIME_SCALE_FACTOR = 2.0
    PNL_CLIP_MIN = -1.0
    PNL_CLIP_MAX = 1.0
    EPSILON = 1e-9
    
    # Scale Portfolio Time (Uniform [0, 1] -> Z-score approximate)
    TIME_MEAN = 0.5
    TIME_STD = 0.29 # Standard deviation of Uniform(0, 1)
    
    def __init__(self, window_size=64, n_features=None):
        self.window_size = window_size
        self.n_features = n_features
        
        # Online Mode utilities
        self.ff = QuantFeatureFactory()
        self.history = deque(maxlen=window_size)
    
    @property
    def feature_names(self):
        """Trả về danh sách tên các đặc trưng."""
        names = list(FEATURE_COLUMNS)
        # Macro Regime (3 components)
        names.extend(['regime_0', 'regime_1', 'regime_2'])
        # Portfolio / Context States
        names.extend(['unrealized_pnl_z', 'time_in_trade_z', 'position_weight'])
        return names
    
    def reset(self):
        """Đặt lại trạng thái."""
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
        [Chế độ Online] Tạo vector trạng thái cho 1 bước.
        """
        features = []
        
        # 1. Market Features (Z-scored)
        for col in FEATURE_COLUMNS:
            val = indicators.get(col)
            if val is None:
                val = portfolio_state.get(col)
                
            if val is None:
                 raise RuntimeError(f"CRITICAL: Feature '{col}' missing! Check Env.")

            if perform_normalization:
                norm_val = self.ff.update_and_normalize(col, val, frozen=frozen)
                features.append(norm_val)
            else:
                # Validation in Batch Mode
                if not np.isfinite(val):
                    raise ValueError(f"Feature {col} not finite: {val}")
                features.append(val)
                
        # 2. Macro Regime (3 components)
        if regime_probs.shape != (3,):
            raise ValueError(f"Expected regime_probs shape (3,), got {regime_probs.shape}")
            
        # Scale to match Z-score magnitude
        features.extend(regime_probs * self.REGIME_SCALE_FACTOR)
        
        # 3. Portfolio State (Unify to Z-score scales)
        # Unrealized PnL: [% Equity] -> Limited to [-1, 1] then Z-scored if possible
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        equity = portfolio_state.get('equity', 10000.0)
        pnl_pct = np.clip(unrealized_pnl / (equity + self.EPSILON), self.PNL_CLIP_MIN, self.PNL_CLIP_MAX)
        
        if perform_normalization:
            # Dùng EMA Factory để chuẩn hóa PnL - giúp Agent hiểu quy mô PnL hiện tại so với lịch sử
            norm_pnl_z = self.ff.update_and_normalize('unrealized_pnl_z', pnl_pct, frozen=frozen)
            features.append(norm_pnl_z)
        else:
            features.append(pnl_pct) # Fallback to clipped pct if skipping norm
        
        # Time in Trade: [0, 1] -> Z-score (Mean=0.5, Std=0.29)
        raw_time = portfolio_state.get('time_in_trade', 0.0)
        norm_time_z = (raw_time - self.TIME_MEAN) / self.TIME_STD
        features.append(norm_time_z)
        
        # Current Position (Weight already reflects scale)
        pos = portfolio_state.get('position', 0.0)
        features.append(float(pos))
        
        obs_vector = np.array(features, dtype=np.float32)
        self.history.append(obs_vector)
        
        return obs_vector

    def build_state(self, market_history, regime_probs, account_state):
        """
        [Chế độ Batch] Dùng cho training offline hoặc feed-forward.
        """
        # Validate input
        if regime_probs.shape != (3,):
            # Nếu truyền vào mảng nhiều chiều (n_steps, 3)
            if regime_probs.ndim == 2 and regime_probs.shape[1] == 3:
                pass
            else:
                raise ValueError(f"Expected regime_probs shape (3,) or (N,3), got {regime_probs.shape}")

        # Market Data
        market_tensor = market_history[FEATURE_COLUMNS].values
        
        # Portfolio Features
        equity = account_state.get('equity', 10000.0)
        unrealized_pnl = account_state.get('unrealized_pnl', 0.0)
        pnl_pct = np.clip(unrealized_pnl / (equity + self.EPSILON), self.PNL_CLIP_MIN, self.PNL_CLIP_MAX)
        
        raw_time = account_state.get('time_in_trade', 0.0)
        norm_time_z = (raw_time - self.TIME_MEAN) / self.TIME_STD
        
        pos_weight = float(account_state.get('position', 0.0))
        
        # Regime scaling
        scaled_regime = regime_probs * self.REGIME_SCALE_FACTOR
        
        # Meta vector
        # Lưu ý: Trong batch mode chúng ta không thể dễ dàng áp dụng EMA normalization 
        # cho PnL mà không lặp qua từng bước. Ở đây dùng pnl_pct tạm thời.
        if scaled_regime.ndim == 1:
            meta_vec = np.concatenate([scaled_regime, [pnl_pct, norm_time_z, pos_weight]])
            meta_tensor = np.tile(meta_vec, (len(market_tensor), 1))
        else:
            # Batch of regime probs
            meta_vecs = []
            for i in range(len(scaled_regime)):
                 meta_vecs.append(np.concatenate([scaled_regime[i], [pnl_pct, norm_time_z, pos_weight]]))
            meta_tensor = np.array(meta_vecs)

        full_state = np.hstack([market_tensor, meta_tensor])
        return full_state.astype(np.float32)
