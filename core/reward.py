import numpy as np

class CompoundReward:
    """
    Hệ thống phần thưởng Compound (PnL & Risk Management).
    Cơ chế: Adaptive Clipping, Dynamic Volatility Scaling.
    """
    
    def __init__(self, initial_equity=10000.0, penalty_scale=1.0, target_vol=0.15):
        self.initial_equity = initial_equity
        self.penalty_scale = penalty_scale
        self.target_vol = target_vol
        
        # Tracking (Optional)
        self.cumulative_reward = 0.0
        
    def calculate(self, current_net_worth, prev_net_worth, current_drawdown_pct, prev_drawdown_pct=0.0, current_vol=0.15, current_weight=0.0):
        """Tính toán phần thưởng dựa trên lợi nhuận và kiểm soát rủi ro."""
        
        # 1. Phần thưởng từ lợi nhuận (Log Returns)
        safe_curr = max(1e-6, current_net_worth)
        safe_prev = max(1e-6, prev_net_worth)
        log_return = np.log(safe_curr / safe_prev)
        
        target_vol = 0.15
        safe_vol = max(current_vol, 0.05) 
        adaptive_scale = min(500.0, (target_vol / safe_vol) * 250.0) 
        pnl_reward = log_return * adaptive_scale
        
        # 2. Phạt theo mức sụt giảm tài sản (Drawdown)
        dd_penalty = 0.0
        if current_drawdown_pct > 5.0:
            dd_increase = current_drawdown_pct - prev_drawdown_pct
            if dd_increase > 0:
                dd_penalty = -dd_increase * 2.0 * self.penalty_scale 
                if dd_increase > 1.0: # Gia tăng mức phạt khi DD bùng nổ
                    dd_penalty *= 2.0 
            
        # 3. Adaptive Clipping & Vol scaling
        vol_ratio = safe_vol / self.target_vol
        clip_range = np.clip(20.0 * np.sqrt(vol_ratio), 10.0, 50.0)
        
        vol_damping = 1.0
        vol_boost = 1.0
        
        if vol_ratio > 3.0: # Giảm thưởng khi biến động cực đoan
            vol_damping = max(0.5, 1.0 / np.sqrt(vol_ratio - 2.0))
            pnl_reward *= vol_damping
        elif vol_ratio < 0.5: # Tăng thưởng khi thị trường ổn định
            vol_boost = 1.0 + (0.5 - vol_ratio) * 0.5
            pnl_reward *= vol_boost
            
        # 4. TOTAL
        total_reward = pnl_reward + dd_penalty + holding_penalty
        raw_reward = total_reward # Before clipping
        
        # Final Guard: Replace NaN with 0.0 before clipping
        if np.isnan(total_reward):
            total_reward = 0.0
            
        clipped_reward = np.clip(total_reward, -clip_range, clip_range)
        
        return clipped_reward, {
            'total': clipped_reward,
            'pnl_reward': pnl_reward,
            'dd_penalty': dd_penalty,
            'log_return': log_return,
            'current_dd': current_drawdown_pct,
            'pnl_to_dd_ratio': pnl_reward / (abs(dd_penalty) + 1e-4),
            'vol_scaling': vol_ratio,
            'clip_range': clip_range,
            'vol_damping': vol_damping,
            'vol_boost': vol_boost,
            'raw_reward': raw_reward
        }

    def reset(self):
        self.cumulative_reward = 0.0
