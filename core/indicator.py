import numpy as np
from numba import jit
import warnings

# Tắt cảnh báo chia cho 0 để tối ưu tốc độ tính toán
# warnings.filterwarnings("ignore") # REMOVED: Handle NaNs internally

# -----------------------------------------------------------------------------
# 1. ĐƯỜNG TRUNG BÌNH LŨY THỪA (EMA)
# Độ phức tạp: O(N) - Chạy một lượt qua dữ liệu
# Logic: Khởi tạo bằng SMA, sau đó thực hiện bước tính EMA.
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_ema(data: np.ndarray, span: int) -> np.ndarray:
    """
    Tính EMA siêu tốc bằng Numba.
    Khởi tạo bằng SMA của 'span' phần tử đầu tiên để đạt độ chính xác cao.
    Các giá trị trước 'span' sẽ được để là NaN.
    """
    n = len(data)
    ema = np.full(n, np.nan)
    
    if n == 0 or n < span:
        return ema
        
    alpha = 2.0 / (span + 1)
    
    # Khởi tạo giá trị đầu tiên bằng phương pháp SMA
    ema[span-1] = np.mean(data[:span])
    
    for i in range(span, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
    return ema

# -----------------------------------------------------------------------------
# 2. CHỈ SỐ SỨC MẠNH TƯƠNG ĐỐI (RSI)
# Độ phức tạp: O(N) - Sử dụng phương pháp làm mượt của Wilder (Wilder's Smoothing)
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_rsi(close_prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Tính RSI theo chuẩn Wilder's Smoothing.
    """
    n = len(close_prices)
    rsi = np.full(n, np.nan)
    
    if n == 0 or n <= period:
        return rsi

    # Tính toán chênh lệch giá (Deltas)
    deltas = np.diff(close_prices)
    
    # Khởi tạo Lợi nhuận/Thua lỗ (SMA cho chu kỳ đầu tiên)
    gains = np.maximum(deltas, 0.0)
    losses = np.abs(np.minimum(deltas, 0.0))
    
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    
    # Tính toán giá trị RSI đầu tiên
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
    # Tiếp tục làm mượt theo phương pháp Wilder
    for i in range(period + 1, n):
        delta = deltas[i-1]
        
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

# -----------------------------------------------------------------------------
# 3. MACD (ĐƯỜNG TRUNG BÌNH ĐỘNG HỘI TỤ PHÂN KỲ)
# Độ phức tạp: O(N)
# Logic: Đường tín hiệu khởi tạo bằng SMA của MACD để tránh nhiễu lúc đầu.
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_macd(close_prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """
    Tính toán MACD, Signal Line và Histogram.
    Độ chính xác cao: Đường Signal chỉ được tính khi đường MACD đã có giá trị hợp lệ.
    """
    # 1. Tính MACD Line = EMA(nhanh) - EMA(chậm)
    ema_fast = calculate_ema(close_prices, fast_period)
    ema_slow = calculate_ema(close_prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    
    # 2. Tính Signal Line = EMA(macd_line)
    n = len(close_prices)
    signal_line = np.full(n, np.nan)
    hist = np.full(n, np.nan)
    
    # Điểm bắt đầu có dữ liệu MACD hợp lệ là từ index = slow_period - 1
    start_idx = slow_period - 1
    if start_idx >= n:
        return macd_line, signal_line, hist
        
    valid_macd = macd_line[start_idx:]
    
    # Tính EMA cho phần MACD hợp lệ
    valid_signal = calculate_ema(valid_macd, signal_period)
    
    # Gán lại giá trị vào mảng kết quả
    signal_line[start_idx:] = valid_signal
    
    # Tính Histogram (Biểu đồ cột)
    hist = macd_line - signal_line
    
    return macd_line, signal_line, hist

# -----------------------------------------------------------------------------
# 4. DẢI BOLLINGER (BOLLINGER BANDS)
# Độ phức tạp: O(N) - Sử dụng thuật toán Welford / Rolling Sum
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_bollinger(close_prices, period=20, k=2.0):
    """
    Tính Bollinger Bands bằng phương pháp Rolling Mean & Std.
    """
    n = len(close_prices)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n == 0 or n < period:
        return upper, middle, lower
        
    # Tính toán tổng và bình phương ban đầu cho cửa sổ (window)
    sum_val = 0.0
    sum_sq = 0.0
    
    for i in range(period):
        val = close_prices[i]
        sum_val += val
        sum_sq += val * val
        
    mean = sum_val / period
    var = (sum_sq / period) - (mean * mean)
    std = np.sqrt(var if var > 0 else 0.0)
    
    middle[period-1] = mean
    upper[period-1] = mean + k * std
    lower[period-1] = mean - k * std
    
    # Thuật toán cửa sổ trượt (Sliding window)
    for i in range(period, n):
        val_new = close_prices[i]
        val_old = close_prices[i-period]
        
        sum_val = sum_val + val_new - val_old
        sum_sq = sum_sq + (val_new * val_new) - (val_old * val_old)
        
        mean = sum_val / period
        var = (sum_sq / period) - (mean * mean)
        std = np.sqrt(var if var > 0 else 0.0)
        
        middle[i] = mean
        upper[i] = mean + k * std
        lower[i] = mean - k * std
        
    return upper, middle, lower

# -----------------------------------------------------------------------------
# 5. ATR (KHOẢNG BIẾN ĐỘNG TRUNG BÌNH THỰC TẾ)
# Độ phức tạp: O(N)
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_atr(high, low, close, period=14):
    """
    Tính ATR chuẩn theo phương pháp của Wilder.
    """
    n = len(close)
    atr = np.full(n, np.nan)
    if n == 0: return atr
    
    tr = np.zeros(n)
    
    # TR (True Range) đầu tiên = Cao - Thấp
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
        
    # ATR đầu tiên = Trung bình cộng các TR
    first_atr = 0.0
    for i in range(period):
        first_atr += tr[i]
    first_atr /= period
    
    atr[period-1] = first_atr
    
    # Làm mượt theo kiểu Wilder (Wilder Smoothing)
    current_atr = first_atr
    for i in range(period, n):
        current_atr = ((current_atr * (period - 1)) + tr[i]) / period
        atr[i] = current_atr
        
    return atr

# -----------------------------------------------------------------------------
# 6. CCI (CHỈ SỐ KÊNH HÀNG HÓA - COMMODITY CHANNEL INDEX)
# Độ phức tạp: O(N) - Tối ưu hóa bằng cửa sổ trượt
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_cci(high, low, close, period=20):
    """
    Tính CCI siêu tốc bằng Numba.
    CCI = (Typical Price - SMA TP) / (0.015 * Mean Deviation)
    """
    n = len(close)
    cci = np.full(n, np.nan)
    tp = (high + low + close) / 3.0
    
    if n == 0 or n < period:
        return cci
        
    for i in range(period - 1, n):
        # [LỖI 46 OPTIMIZATION] Manual SMA to avoid slicing overhead
        sum_tp = 0.0
        for j in range(i - period + 1, i + 1):
            sum_tp += tp[j]
        sma_tp = sum_tp / period
        
        # Mean Deviation manual calculation
        sum_dev = 0.0
        for j in range(i - period + 1, i + 1):
            sum_dev += abs(tp[j] - sma_tp)
        mean_dev = sum_dev / period
        
        if mean_dev < 1e-9:
            cci[i] = 0.0
        else:
            cci[i] = (tp[i] - sma_tp) / (0.015 * mean_dev)
            
    return cci

# -----------------------------------------------------------------------------
# 7. OBV (KHỐI LƯỢNG CÂN BẰNG - ON BALANCE VOLUME)
# Độ phức tạp: O(N)
# -----------------------------------------------------------------------------
@jit(nopython=True)
def calculate_obv(close, volume):
    """
    Tính OBV siêu tốc bằng Numba.
    """
    n = len(close)
    obv = np.zeros(n)
    
    if n == 0:
        return obv
        
    obv[0] = volume[0]
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
            
    return obv
