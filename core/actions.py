import numpy as np
from gymnasium import spaces

class ActionHandler:
    """
    ActionHandler: Chuyển đổi đầu ra của mạng thần kinh thành trọng số vị thế (Target Weight).
    Tích hợp cơ chế Volatility Targeting và Hạch toán chuẩn XAUUSD (V4.1 Production-Ready).
    """
    
    def __init__(self, target_vol=0.15, max_leverage=1.0, 
                 spread_usd=0.30, commission_usd=0.0, lot_size=100.0):
        self.target_vol = max(1e-6, target_vol)
        self.max_leverage = max_leverage
        
        # XAUUSD Standard Accounting
        self.contract_size = lot_size 
        self.tick_size = 0.01
        # [LỖI 21 FIX] For reference only in accounting logs
        self.tick_value = self.contract_size * self.tick_size 
        
        self.spread_usd = spread_usd 
        self.commission_usd = commission_usd 
        
        # [LỖI 18 & 14 FIX] Hedge Fund Swap Convention
        # POSITIVE = Cost (bị trừ), NEGATIVE = Gain (được cộng)
        # Standard: Long pay swap (-$5/lot), Short receive swap (+$3/lot)
        self.swap_long_oz = 0.05   # Cost per oz per day
        self.swap_short_oz = -0.03  # Gain per oz per day

    @property
    def lot_size(self):
        return self.contract_size
        
    def get_target_weight(self, action_logit: float, vol_scalar: float, current_weight: float = 0.0) -> float:
        if not np.isfinite(action_logit): return 0.0
        action_tanh = np.tanh(action_logit)
        vol_scalar = np.clip(vol_scalar, 0.0, self.max_leverage)
        if not np.isfinite(vol_scalar): vol_scalar = 0.0
        final_weight = action_tanh * vol_scalar
        return np.clip(final_weight, -self.max_leverage, self.max_leverage)

    def should_halt_trading(self, current_volatility):
        """
        [LỖI 20 FIX] Circuit Breaker Recommendation.
        WARNING: This method only returns a recommendation. 
        Caller (TradingEnv) MUST check this before executing trades.
        """
        if not np.isfinite(current_volatility) or current_volatility <= 0: return True
        return (current_volatility / self.target_vol) > 5.0

    def get_effective_spread(self, current_volatility):
        if not np.isfinite(current_volatility) or current_volatility < 0:
            return self.spread_usd
        vol_ratio = current_volatility / self.target_vol
        spread_multiplier = 1.0 + max(0, (vol_ratio**1.2 - 1.0) * 1.5)
        spread_multiplier = min(spread_multiplier, 10.0)
        return self.spread_usd * spread_multiplier

    def get_fill_price(self, mid_price, weight_delta, current_volatility):
        """
        [LỖI 17 FIX] Raise error instead of silent failure.
        """
        if not np.isfinite(mid_price):
            raise ValueError(f"CRITICAL: mid_price is NaN/Inf: {mid_price}")
        if mid_price <= 0:
            if mid_price == 0: return 0.0
            raise ValueError(f"CRITICAL: mid_price is negative: {mid_price}")

        effective_spread = self.get_effective_spread(current_volatility)
        half_spread = effective_spread * 0.5
        
        if weight_delta > 0: # BUY -> Pay Ask
            return mid_price + half_spread
        elif weight_delta < 0: # SELL -> Receive Bid
            return mid_price - half_spread
        return mid_price

    def calculate_cost_advanced(self, ounces_traded, current_volatility, current_price):
        """
        [LỖI 19 FIX] Input Validation to prevent Infinity/NaN propagation.
        """
        if not np.isfinite(ounces_traded) or not np.isfinite(current_volatility) or not np.isfinite(current_price):
            return 0.0
        if current_price <= 0: return 0.0
        
        ounces_abs = abs(ounces_traded)
        base_comm = ounces_abs * self.commission_usd
        
        vol_ratio = current_volatility / self.target_vol
        if vol_ratio > 1.0:
            excess_vol = vol_ratio - 1.0
            extra_slippage_pct = excess_vol * 0.001 
            extra_slippage_pct = min(extra_slippage_pct, 0.01)
            extra_slippage_usd = ounces_abs * current_price * extra_slippage_pct
        else:
            extra_slippage_usd = 0.0
        return base_comm + extra_slippage_usd

    def calculate_swap_cost(self, ounces_held, holding_days):
        """
        [LỖI 18 FIX] Return POSITIVE = cost (subtract), NEGATIVE = gain (add).
        """
        if not np.isfinite(ounces_held) or abs(ounces_held) < 1e-9: return 0.0
        
        # swap = units * rate_per_unit * time
        if ounces_held > 0: # Long
            return ounces_held * self.swap_long_oz * holding_days
        else: # Short -> Use signed result (Gain is negative)
            return abs(ounces_held) * self.swap_short_oz * holding_days
        

