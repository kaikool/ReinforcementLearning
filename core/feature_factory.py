import numpy as np
import pandas as pd
from numba import jit
from core.indicator import (
    calculate_rsi, calculate_macd, calculate_bollinger, 
    calculate_atr, calculate_cci
)
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")



@jit(nopython=True)
def calc_hurst_numba(ts):
    """Tính số mũ Hurst (Hurst Exponent) tối ưu bằng Numba."""
    lags = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20])
    tau = np.zeros(len(lags))
    
    for i in range(len(lags)):
        lag = lags[i]
        if len(ts) <= lag: return 0.5
        diff = ts[lag:] - ts[:-lag]
        mean_val = np.mean(diff)
        sq_diff = (diff - mean_val) ** 2
        std = np.sqrt(np.sum(sq_diff) / len(diff))
        
        # [LỖI 41 FIX] Tránh gây nhiễu khi dữ liệu phẳng
        if std < 1e-8:
            return 0.5 
            
        tau[i] = std
        
    # Polyfit (Hồi quy tuyến tính trên không gian log-log)
    x = np.log(lags)
    y = np.log(tau)
    
    # OLS để tính độ dốc (H)
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if abs(denominator) < 1e-12:
        return 0.5
        
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    
    return slope * 2.0

@jit(nopython=True)
def calc_entropy_numba(x, bins=10):
    """Tính Entropy Shannon đã chuẩn hóa (Tối ưu bằng Numba)."""
    n = len(x)
    if n <= 1: return 0.0
    
    min_val = np.min(x)
    max_val = np.max(x)
    
    # Xử lý trường hợp biên: Min bằng Max (dữ liệu phẳng)
    if max_val - min_val < 1e-9:
        return 0.0
        
    hist = np.zeros(bins)
    bin_width = (max_val - min_val) / bins + 1e-9 # Thêm epsilon
    
    for val in x:
        idx = int((val - min_val) / bin_width)
        if idx >= bins: idx = bins - 1
        hist[idx] += 1
        
    probs = hist / n
    
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * np.log(p)
            
    # Chuẩn hóa Entropy về khoảng [0, 1]
    return ent / np.log(bins)



@dataclass
class EMAState:
    """Lưu trữ trạng thái cho thuật toán EMA Normalization."""
    mean: float = 0.0
    var: float = 1.0
    alpha: float = 0.001  # Alpha nhỏ (~2000 steps) để stable
    initialized: bool = False

class QuantFeatureFactory:
    """
    Nhà máy sản xuất đặc trưng Quant (Quantitative Features).
    Đảm bảo tính nhân quả (Causality) và chuẩn hóa Online (EMA).
    """
    
    def __init__(self, 
 
                 regime_window=100,
                 clip_range: float = 10.0, # Nới lỏng clip range
                 ema_alpha: float = 0.001): 
        self.regime_window = regime_window
        
        self.stats = {} 
        self.clip_range = clip_range
        self.ema_alpha = ema_alpha

    def update_and_normalize(self, feature_name: str, value: float, frozen: bool = False) -> float:
        """
        Cập nhật thống kê EMA và trả về Z-Score online. 
        Nếu frozen=True: Chỉ normalize bằng Mean/Var hiện tại, KHÔNG cập nhật state.
        """
        # Xử lý nan/inf
        if not np.isfinite(value):
            return 0.0 
            
        if feature_name not in self.stats:
            self.stats[feature_name] = EMAState(alpha=self.ema_alpha)
            
        state = self.stats[feature_name]
        
        if not state.initialized:
            if frozen: return 0.0 # Cannot normalize if not initialized and frozen
            state.mean = value
            state.var = 1.0 
            state.initialized = True
            return 0.0
        
        if not frozen:
            # Incremental EMA Update (Training Mode only)
            diff = value - state.mean
            incr = state.alpha * diff
            state.mean += incr
            state.var = (1 - state.alpha) * (state.var + diff * incr)
        
        std_dev = np.sqrt(max(1e-8, state.var))
        z_score = (value - state.mean) / std_dev
        
        return np.clip(z_score, -self.clip_range, self.clip_range)

    def reset(self):
        """Reset internal state (Does NOT clear EMA stats by default for persistence)."""
        pass
        
    def clear_stats(self):
        """Explicitly clear normalization statistics."""
        self.stats = {}

    def save_stats(self, path: str):
        """Save EMA statistics to a file for later use in Backtesting/Live."""
        import joblib
        joblib.dump(self.stats, path)
        print(f"   [Factory] Stats saved to: {path}")

    def load_stats(self, path: str):
        """Load EMA statistics from a file."""
        import joblib
        import os
        if os.path.exists(path):
            self.stats = joblib.load(path)
            print(f"   [Factory] Stats loaded from: {path}")
        else:
            print(f"   [Factory] Warning: Stats file not found at {path}")

    def process_data(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """Pipeline xử lý dữ liệu."""
        df = df.copy()
        if 'date' in df.columns:
            # Xử lý định dạng ngày tháng hỗn hợp
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True) 
        
        # 1. Temporal Features (Chỉ dùng Cyclic Encoding để tránh Discontinuity)
        # Bỏ feature 'hour' tuyến tính vì 23h và 0h cách xa nhau về giá trị nhưng gần về thời gian
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
        
        # Đánh dấu phiên hoạt động mạnh (Active Session)
        df['is_active_hour'] = ((df.index.hour >= 8) & (df.index.hour <= 22)).astype(float)
        
        # 2. Technical Features (Raw)
        df = self._add_technical_indicators_raw(df)
        
        # 3. Regime Features (Raw)
        df = self._add_regime_features_raw(df)
        
        # 4. Xử lý Cold Start (Drop Burn-in TRƯỚC khi Normalize)
        # Cần drop khoảng 2/alpha mẫu đầu tiên để Mean/Var hội tụ
        # Hoặc ít nhất bằng regime_window + 100 để đảm bảo Hurst ổn định
        burn_in = max(int(2.0 / self.ema_alpha), self.regime_window + 100)
        if len(df) > burn_in + 100: # Chỉ drop nếu đủ dữ liệu
            df = df.iloc[burn_in:]
        
        if normalize:
            # 5. Normalize (EWM Online Simulation) - Apply AFTER drop to avoid initial noise skew
            features_to_norm = [
                'log_ret', 'realized_vol', 
                'macd_raw', 'rsi_raw', 'cci_raw', 'atr_rel', 
                'bb_pct', 'vol_rel', 
                'entropy_raw', 'efficiency_raw',
                'hurst', 'entropy_delta', 'price_shock', 
                'interaction_trend_vol', 'trend_efficiency', 'directional_persistence'
            ]
            
            for col in features_to_norm:
                if col in df.columns:
                    ewm_mean = df[col].ewm(alpha=self.ema_alpha, adjust=False).mean()
                    ewm_std = df[col].ewm(alpha=self.ema_alpha, adjust=False).std()
                    
                    df[col] = (df[col] - ewm_mean) / (ewm_std + 1e-8)
                    
                    if col in ['price_shock', 'entropy_delta']:
                        df[col] = df[col].clip(-5.0, 5.0).fillna(0.0)
                    else:
                        df[col] = df[col].clip(-self.clip_range, self.clip_range).fillna(0.0)
                    
                    # Lưu trạng thái EMA cuối cùng để duy trì tính liên tục khi Online
                    final_mean = ewm_mean.iloc[-1]
                    final_std = ewm_std.iloc[-1]
                    final_var = (final_std ** 2)
                    
                    if col not in self.stats:
                        self.stats[col] = EMAState(alpha=self.ema_alpha)
                    
                    self.stats[col].mean = final_mean
                    self.stats[col].var = max(1e-8, final_var)
                    self.stats[col].initialized = True

        df.dropna(inplace=True)
        return df

    def _add_technical_indicators_raw(self, df):
        """Tính các chỉ báo thô (chưa normalize)."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Log Return & Realized Vol (PER BAR - NOT ANNUALIZED)
        # Unit: Standard Deviation of Log Returns per period (e.g. per M15 bar)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['realized_vol'] = df['log_ret'].rolling(window=20).std()
        
        # MACD (Raw)
        _, _, macd_hist = calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
        df['macd_raw'] = macd_hist / (close + 1e-9) 
        
        # RSI (Raw [0, 100])
        df['rsi_raw'] = calculate_rsi(close, period=14)
        
        # CCI (Raw)
        df['cci_raw'] = calculate_cci(high, low, close, period=20)
        
        # ATR Relative
        atr_val = calculate_atr(high, low, close, period=14)
        df['atr_rel'] = atr_val / (close + 1e-9)
        
        # Bollinger
        bb_upper, bb_middle, bb_lower = calculate_bollinger(close, period=20, k=2.0)
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)
        df['bb_pct'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-9)
        
        # Volume
        vol_ma = df['volume'].rolling(20).mean()
        df['vol_rel'] = df['volume'] / (vol_ma + 1e-9)
        
        return df

    def _add_regime_features_raw(self, df):
        """Các chỉ báo thống kê thô."""
        # Hurst (Dùng Log Price để scale invariant)
        log_close = np.log(df['close'])
        hurst_val = log_close.rolling(self.regime_window).apply(calc_hurst_numba, raw=True)
        
        # 1. Loại bỏ nhãn rời rạc, sử dụng giá trị Hurst liên tục
        # Thay vì discrete -1/0/1, dùng continuous Hurst
        df['hurst'] = hurst_val.fillna(0.5)
        
        # 2. Các biến đại diện cho sự kiện (Temporal Spikes)
        # Entropy Jump: Đo lường sự gia tăng đột biến của hỗn loạn
        df['entropy_raw'] = df['log_ret'].rolling(50).apply(calc_entropy_numba, raw=True).fillna(0)
        df['entropy_delta'] = df['entropy_raw'].diff().fillna(0)
        
        current_vol = df['realized_vol'].replace(0, 1e-6)
        df['price_shock'] = (df['log_ret'].abs() / current_vol).clip(0, 5.0).fillna(0)
        
        # Trend Efficiency (Alpha Proxy) - User Request
        # Avg Return per Unit of Risk
        df['trend_efficiency'] = df['log_ret'].rolling(20).sum() / (df['realized_vol'].rolling(20).mean() + 1e-9)

        # Directional Persistence (Win-rate Proxy) - User Request (Full Expectancy Pass)
        # Avg Sign of returns (-1 to 1). Closer to 1 -> Strong Up Trend consistency
        df['directional_persistence'] = np.sign(df['log_ret']).rolling(20).mean().fillna(0)
        
        # Efficiency
        change = (df['close'] - df['close'].shift(20)).abs()
        volatility = df['close'].diff().abs().rolling(20).sum()
        df['efficiency_raw'] = change / (volatility + 1e-9)
        
        # Interactions (Continuous)
        # Hurst > 0.5 (Trend), < 0.5 (Mean Rev)
        # Interaction = (Hurst - 0.5) * Volume
        # -> Dương lớn: Strong Trend + High Vol
        # -> Âm lớn: Mean Reversion + High Vol
        df['interaction_trend_vol'] = (df['hurst'] - 0.5) * df['vol_rel']
        
        return df
        
# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
    data = {
        'date': dates,
        'open': np.random.rand(1000) * 100 + 100,
        'high': np.random.rand(1000) * 100 + 105,
        'low': np.random.rand(1000) * 100 + 95,
        'close': np.random.rand(1000) * 100 + 100,
        'volume': np.random.randint(100, 10000, 1000)
    }
    df_raw = pd.DataFrame(data)
    feature_eng = QuantFeatureFactory(regime_window=100)
    df_processed = feature_eng.process_data(df_raw)
    print("Done:", df_processed.shape)
    print(df_processed.tail())