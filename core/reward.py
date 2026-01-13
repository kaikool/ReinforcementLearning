import numpy as np

class CompoundReward:
    """
    Hệ thống phần thưởng Compound (V8 - Institutional Grade & Continuous).
    Tích hợp: Adaptive Scaling, Fully Smooth Multi-factor Penalties, và Safe Clipping.
    
    Cơ chế V8:
    - PnL Reward based on Log Returns.
    - Volatility Targeting (Equalization across regimes).
    - Fully Continuous Drawdown Penalty (Soft-thresholding via Sigmoid).
    - Smooth Volatility Dampening/Boosting (Sigmoid-based transitions).
    - Adaptive Clipping (Volatility-scaled bounds).
    """
    
    # --- Operational Constants ---
    SAFETY_EPSILON = 1e-6
    MIN_VOL = 0.05            # Sàn volatility
    VOL_SCALE_FACTOR = 250.0  # Hệ số nhân reward
    MAX_ADAPTIVE_SCALE = 500.0
    
    # Drawdown Penalty
    DD_THRESHOLD = 5.0        # Ngưỡng bắt đầu phạt DD (%)
    DD_BASE_PENALTY = 2.0     # Hệ số phạt cơ bản
    
    # Adaptive Clipping
    CLIP_BASE = 20.0          # Clip range cơ bản
    CLIP_MIN = 10.0
    CLIP_MAX = 50.0
    
    # Volatility Regimes
    HIGH_VOL_THRESHOLD = 3.0  # Ratio so với target_vol
    LOW_VOL_THRESHOLD = 0.5
    
    def __init__(self, initial_equity=10000.0, penalty_scale=1.0, target_vol=0.15):
        self.initial_equity = initial_equity
        self.penalty_scale = penalty_scale
        self.target_vol = max(self.SAFETY_EPSILON, target_vol)
        
        # Tracking
        self.cumulative_reward = 0.0
        
    def calculate(self, current_net_worth, prev_net_worth, current_drawdown_pct, 
                  prev_drawdown_pct=0.0, current_vol=None):
        """
        Tính toán phần thưởng đa nhân tố liên tục.
        """
        if current_vol is None:
            current_vol = self.target_vol
            
        # 1. Phần thưởng từ lợi nhuận (Log Returns)
        safe_curr = max(self.SAFETY_EPSILON, current_net_worth)
        safe_prev = max(self.SAFETY_EPSILON, prev_net_worth)
        log_return = np.log(safe_curr / safe_prev)
        
        # 2. Volatility Targeting
        safe_vol = max(current_vol, self.MIN_VOL) 
        adaptive_scale = min(self.MAX_ADAPTIVE_SCALE, (self.target_vol / safe_vol) * self.VOL_SCALE_FACTOR) 
        pnl_reward = log_return * adaptive_scale
        
        # 3. Fully Continuous Drawdown Penalty (V8 Upgrade)
        # Sử dụng Sigmoid để kích hoạt phạt mượt mà xung quanh ngưỡng DD_THRESHOLD
        dd_active = 1.0 / (1.0 + np.exp(-(current_drawdown_pct - self.DD_THRESHOLD) / 2.0))
        dd_increase = current_drawdown_pct - prev_drawdown_pct
        
        dd_penalty = 0.0
        if dd_increase > 0:
            # Phạt khi DD tăng, nhân với hệ số kích hoạt mượt
            dd_penalty = -np.tanh(dd_increase / 2.0) * dd_increase * self.DD_BASE_PENALTY * self.penalty_scale * dd_active
            
        # 4. Adaptive Clipping & Smooth Volatility Smoothing (V8 Upgrade)
        vol_ratio = safe_vol / self.target_vol
        clip_range = np.clip(self.CLIP_BASE * np.sqrt(vol_ratio), self.CLIP_MIN, self.CLIP_MAX)
        
        # Smooth Sigmoid-based factors for Dampening and Boosting
        # Tránh các bước nhảy (step) tại ngưỡng thresholds gây nhiễu gradient
        high_vol_factor = 1.0 / (1.0 + np.exp(-(vol_ratio - self.HIGH_VOL_THRESHOLD) * 2.0))
        low_vol_factor = 1.0 / (1.0 + np.exp((vol_ratio - self.LOW_VOL_THRESHOLD) * 2.0))
        
        # Damping mượt tiến về 0.5 khi vol cực cao
        vol_damping = 1.0 - high_vol_factor * 0.5  
        # Boost mượt khi vol cực thấp
        vol_boost = 1.0 + low_vol_factor * (self.LOW_VOL_THRESHOLD - vol_ratio) * 0.5
        
        pnl_reward *= (vol_damping * vol_boost)
            
        # 5. TOTAL & GUARD
        total_reward = pnl_reward + dd_penalty
        raw_reward = total_reward 
        
        if np.isnan(total_reward) or np.isinf(total_reward):
            total_reward = 0.0
            
        clipped_reward = np.clip(total_reward, -clip_range, clip_range)
        
        # Metadata Dashboard
        info = {
            'total': float(clipped_reward),
            'pnl_reward': float(pnl_reward),
            'dd_penalty': float(dd_penalty),
            'log_return': float(log_return),
            'current_dd': float(current_drawdown_pct),
            'vol_scaling': float(vol_ratio),
            'clip_range': float(clip_range),
            'vol_damping': float(vol_damping),
            'vol_boost': float(vol_boost),
            'raw_reward': float(raw_reward)
        }
        
        return clipped_reward, info

    def reset(self):
        self.cumulative_reward = 0.0
