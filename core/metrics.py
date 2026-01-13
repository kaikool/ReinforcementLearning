import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

class PerformanceMetrics:
    """
    Thư viện tính toán chỉ số hiệu suất giao dịch chuẩn Pro Trader.
    Hỗ trợ cả Equity Curve (Chuỗi thời gian vốn) và Trade List (Danh sách giao dịch).
    """
    
    @staticmethod
    def calculate_from_equity(equity_curve: pd.Series, periods_per_year, risk_free_rate=0.0):
        """
        Tính toán các chỉ số (metrics) từ đường cong vốn (Equity Curve).
        
        Args:
            equity_curve: pd.Series (Index: Thời gian, Value: Vốn)
            periods_per_year: Số kỳ mỗi năm (Ví dụ M15: 252 * 96)
            risk_free_rate: Lãi suất phi rủi ro (Năm hóa, ví dụ 0.04 cho 4%)
        """
        if len(equity_curve) < 2:
            return {
                "Tổng Lợi nhuận (%)": 0.0,
                "CAGR (%)": 0.0,
                "Sụt giảm tối đa (MaxDD) (%)": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "Calmar Ratio": 0.0,
                "Omega Ratio": 0.0,
                "Biến động (Năm hóa) (%)": 0.0,
                "Độ xiên (Skewness)": 0.0,
                "Độ nhọn (Kurtosis)": 0.0,
                "Thời gian bị drawdown (%)": 0.0
            }
            
        equity = equity_curve.values
        initial_equity = equity[0]
        final_equity = equity[-1]
        
        # 1. Phân tích Tỷ suất lợi nhuận (Returns)
        total_return_pct = (final_equity - initial_equity) / initial_equity * 100.0
        
        # FIX: Use Log Returns for Continuous Trading (Stabilize Noise)
        returns = np.diff(np.log(np.maximum(equity, 1e-9)))
        
        # Xử lý lỗi chia cho 0 hoặc giá trị không xác định
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Thống kê cơ bản
        n_periods = len(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # CAGR (Tỷ lệ tăng trưởng hàng năm kép)
        years = n_periods / periods_per_year
        if years > 0:
            cagr = ((final_equity / initial_equity) ** (1/years)) - 1
        else:
            cagr = 0.0
            
        # 3. Phân tích Sụt giảm tài sản (Drawdown)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown) # Giá trị âm
        max_dd_pct = max_dd * 100.0
        
        # Thời gian trong trạng thái Drawdown (% thời gian dưới mức đỉnh)
        is_in_dd = drawdown < 0
        time_in_dd_pct = np.mean(is_in_dd) * 100.0
        
        # 4. Chỉ số hiệu chỉnh theo rủi ro
        # Sharpe Ratio
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        
        if std_ret < 1e-9:
            sharpe = 0.0
        else:
            sharpe = (np.mean(excess_returns) / std_ret) * np.sqrt(periods_per_year)
            
        # Sortino Ratio (Chỉ tính độ lệch chuẩn phần âm)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            downside_std = 1e-9
        else:
            # [LỖI 47 FIX] Chỉ dùng downside_returns để tính Standard Deviation phần âm
            downside_std = np.sqrt(np.mean(downside_returns**2))
            
        if downside_std < 1e-9:
            sortino = 0.0
        else:
            sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
            
        # Calmar Ratio
        if abs(max_dd) < 1e-9:
            calmar = 0.0
        else:
            calmar = cagr / abs(max_dd)
            
        # Omega Ratio (Ngưỡng = 0)
        # Omega = Tổng Lợi nhuận Dương / Trị tuyệt đối Tổng Lợi nhuận Âm
        pos_ret_sum = np.sum(returns[returns > 0])
        neg_ret_sum = np.sum(returns[returns < 0])
        
        if abs(neg_ret_sum) < 1e-9:
            omega = float('inf')
        else:
            omega = pos_ret_sum / abs(neg_ret_sum)
            
        # 5. Độ tin cậy thống kê
        skewness = skew(returns)
        kurt = kurtosis(returns)
        
        skewness = skew(returns)
        kurt = kurtosis(returns)
        
        # REMOVED: Step-wise Win Rate is misleading for XAU/Continuous
        # win_rate_step = np.mean(returns > 0) * 100.0
        
        metrics = {
            "Tổng Lợi nhuận (%)": total_return_pct,
            "CAGR (%)": cagr * 100.0,
            "Sụt giảm tối đa (MaxDD) (%)": max_dd_pct,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Omega Ratio": omega,
            "Biến động (Năm hóa) (%)": std_ret * np.sqrt(periods_per_year) * 100.0,
            # "Tỷ lệ thắng (Step) (%)": win_rate_step, # REMOVED
            "Độ xiên (Skewness)": skewness,
            "Độ nhọn (Kurtosis)": kurt,
            "Thời gian bị drawdown (%)": time_in_dd_pct
        }
        
        return metrics

    @staticmethod
    def calculate_from_trades(trades: list):
        """
        Tính toán metrics dựa trên danh sách các lệnh đã đóng (Closed Trades).
        Đây mới là nơi tính Winrate và Kelly chính xác.
        """
        if not trades:
            return {}
            
        trades = np.array(trades)
        n_trades = len(trades)
        
        # 1. Winrate
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades <= 0]
        
        win_rate = len(winning_trades) / n_trades * 100.0
        
        # 2. Payoff Ratio
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0
        
        if avg_loss < 1e-9:
            payoff_ratio = 0.0
        else:
            payoff_ratio = avg_win / avg_loss
            
        # 3. Kelly Criterion (f = p - q/b)
        # p = win_prob, q = 1-p, b = payoff
        p = win_rate / 100.0
        q = 1.0 - p
        if payoff_ratio > 0:
            kelly = p - (q / payoff_ratio)
        else:
            kelly = 0.0
            
        return {
            "Số lượng lệnh": n_trades,
            "Tỷ lệ thắng (Trade) (%)": win_rate,
            "Lợi nhuận TB/Lệnh": np.mean(trades),
            "Hệ số chi trả (Payoff)": payoff_ratio,
            "Tiêu chuẩn Kelly": kelly,
            "Lệnh thắng lớn nhất": np.max(trades) if len(trades) > 0 else 0,
            "Lệnh thua lớn nhất": np.min(trades) if len(trades) > 0 else 0
        }

    @staticmethod
    def print_metrics(metrics):
        print("\n" + "="*40)
        print("   BÁO CÁO HIỆU SUẤT (PRO TRADER)")
        print("="*40)
        print("NOTE: EXPECTANCY ONLY VALID FROM calculate_from_trades()\n")
        
        # Phần 1: Khả năng sinh lời
        print(f"💰 KHẢ NĂNG SINH LỜI")
        print(f"  Tổng lợi nhuận:    {metrics.get('Tổng Lợi nhuận (%)', 0):>10.2f} %")
        print(f"  CAGR (Hàng năm):   {metrics.get('CAGR (%)', 0):>10.2f} %")
        # print(f"  Tỷ lệ thắng (Step):{metrics.get('Tỷ lệ thắng (Step) (%)', 0):>10.2f} %") # REMOVED
        print(f"  Hệ số chi trả:     {metrics.get('Hệ số chi trả (Payoff Ratio)', 0):>10.2f}")
        
        # Phần 2: Hồ sơ rủi ro
        print(f"\n🛡️ HỒ SƠ RỦI RO")
        print(f"  Sụt giảm tối đa:   {metrics.get('Sụt giảm tối đa (MaxDD) (%)', 0):>10.2f} %")
        print(f"  Biến động (Năm):   {metrics.get('Biến động (Năm hóa) (%)', 0):>10.2f} %")
        print(f"  Thời gian sụt giảm:{metrics.get('Thời gian bị drawdown (%)', 0):>10.2f} %")
        
        # Phần 3: Hiệu quả đầu tư
        print(f"\n⚖️ HIỆU QUẢ ĐẦU TƯ")
        print(f"  Sharpe Ratio:      {metrics.get('Sharpe Ratio', 0):>10.4f}")
        print(f"  Sortino Ratio:     {metrics.get('Sortino Ratio', 0):>10.4f}")
        print(f"  Calmar Ratio:      {metrics.get('Calmar Ratio', 0):>10.4f}")
        print(f"  Omega Ratio:       {metrics.get('Omega Ratio', 0):>10.4f}")
        print("="*40 + "\n")
