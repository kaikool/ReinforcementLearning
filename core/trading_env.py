import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from core.feature_factory import QuantFeatureFactory
from core.regime import MarketRegime
from core.reward import CompoundReward
from core.actions import ActionHandler
from core.state_builder import StateBuilder

class AdvancedTradingEnv(gym.Env):
    """
    Môi trường giao dịch XAUUSD tối ưu hóa cho RL.
    Tính năng: Volatility Targeting, Edge Gate, Causal Observations.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_factory: QuantFeatureFactory,
        regime_model: MarketRegime,
        window_size: int = 64,
        initial_equity: float = 10000.0,
        spread_usd: float = 0.30, 
        commission_usd: float = 0.0,
        lot_size: float = 100.0, 
        df_features: pd.DataFrame = None,
        regime_probs: np.ndarray = None,
        is_training: bool = True
    ):
        super().__init__()
        self.is_training = is_training
        
        # 1. Configuration & Dependencies
        self.df_raw = df.copy()
        if isinstance(self.df_raw.index, pd.DatetimeIndex):
            self.df_raw.reset_index(inplace=True)
            if 'date' not in self.df_raw.columns:
                 col_name = self.df_raw.index.name if self.df_raw.index.name else 'date'
                 self.df_raw.rename(columns={self.df_raw.columns[0]: 'date'}, inplace=True)
        else:
            self.df_raw.reset_index(drop=True, inplace=True)

        self.factory = feature_factory
        self.regime = regime_model
        self.initial_equity = initial_equity
        self.window_size = window_size
        
        # Cấu hình ngưỡng tái lập vị thế
        self.REBALANCE_THRESHOLD = 0.03 
        
        # Thành phần cốt lõi
        self.state_builder = StateBuilder(window_size=window_size)
        self.reward_engine = CompoundReward(initial_equity=initial_equity)
        self.max_net_worth = initial_equity
        
        self.action_handler = ActionHandler(
            target_vol=0.15, 
            max_leverage=5.0,
            spread_usd=spread_usd,
            commission_usd=commission_usd,
            lot_size=lot_size
        )
        self.lot_size = lot_size
        
        # 3. Performance Tracking
        self.profitable_trades = 0
        self.total_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.sum_exposure = 0.0
        self.step_counter = 0
        self.trade_pnl_acc = 0.0
        
        # 3. Data Processing
        if df_features is not None:
             self.df_features = df_features.copy()
        else:
            print("Môi trường: Đang tiền xử lý Đặc trưng & Trạng thái thị trường...")
            self.df_features = self.factory.process_data(self.df_raw)
        
        self.close_arr = self.df_features['close'].values.astype(np.float32)
        self.vol_arr = self.df_features['volume'].values.astype(np.float32)
        
        if 'date' in self.df_features.columns:
            self.date_arr = pd.to_datetime(self.df_features['date'], dayfirst=True, format='mixed').values
        else:
            self.date_arr = self.df_features.index.values
        
        # Indicator Keys
        indicator_keys = [
            'log_ret', 'realized_vol',
            'rsi_raw', 'macd_raw', 'cci_raw', 'atr_rel', 'bb_pct', 'vol_rel',
            'is_active_hour', 
            'entropy_raw', 'efficiency_raw',
            'hurst', 'entropy_delta', 'price_shock',
            'interaction_trend_vol', 'trend_efficiency', 'directional_persistence'
        ]
        self.indicator_arrays = {}
        for k in indicator_keys:
            if k in self.df_features.columns:
                self.indicator_arrays[k] = self.df_features[k].values.astype(np.float32)
        
        # Timeframe detection
        self.timeframe_minutes = self._detect_timeframe()
        bars_per_day = 1440 / self.timeframe_minutes
        self.annualization_factor = np.sqrt(252 * bars_per_day)
        
        if 'realized_vol' in self.df_features.columns:
            self.vol_ann_arr = self.df_features['realized_vol'].values.astype(np.float32) * self.annualization_factor
        elif 'natr' in self.df_features.columns:
            self.vol_ann_arr = self.df_features['natr'].values.astype(np.float32) * self.annualization_factor
        else:
            self.vol_ann_arr = np.full(len(self.df_features), 0.20, dtype=np.float32)

        # [LỖI 32 FIX] Pre-calc Volatility Scalar (Cap in Flatline)
        eps = 1e-6
        safe_vol = np.maximum(self.vol_ann_arr, 0.01) # Minimum 1% annual vol to prevent noise trades
        self.vol_scalar_arr = self.action_handler.target_vol / safe_vol
        self.vol_scalar_arr = np.clip(self.vol_scalar_arr, 0.0, self.action_handler.max_leverage)
        self.indicator_arrays['vol_scalar'] = self.vol_scalar_arr.astype(np.float32)

        # Pre-calc Market Regime Ratio
        self.vol_ratio_arr = safe_vol / self.action_handler.target_vol
        self.indicator_arrays['vol_ratio'] = self.vol_ratio_arr.astype(np.float32)

        # Ensure columns exist
        if 'abs_log_ret' not in self.df_features.columns:
             self.df_features['abs_log_ret'] = self.df_features['log_ret'].abs()
             
        regime_cols = ['log_ret', 'realized_vol', 'entropy_raw', 'abs_log_ret']
        regime_data = self.df_features[regime_cols].fillna(0.0).values
        
        if regime_probs is not None:
            self.regime_probs = regime_probs
        else:
            print(">> Pre-calculating Causal HMM States...")
            self.regime_probs = self.regime.predict_proba_causal(regime_data)
            
        self.n_steps = len(self.df_features)
        
        # [EDGE GATE CONSTANTS]
        self.EFF_REF = 0.15
        self.EDGE_OPEN = 0.65
        self.EDGE_CLOSE = 0.45
        # Calculate 75th percentile of entropy for normalization
        if 'entropy_raw' in self.indicator_arrays:
            self.ENTROPY_MAX = np.percentile(self.indicator_arrays['entropy_raw'], 75)
            if self.ENTROPY_MAX < 1e-6: self.ENTROPY_MAX = 0.5
        else:
            self.ENTROPY_MAX = 0.5
            
        self.edge_active = False
        
        # 4. Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self._reset_state()
        dummy_obs = self._get_current_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=dummy_obs.shape, 
            dtype=np.float32
        )


    def _detect_timeframe(self):
        """Tự động nhận diện khung thời gian từ khoảng cách giữa các dấu thời gian (timestamps)."""
        dates = None
        
        # 1. Check 'date' column
        if 'date' in self.df_raw.columns:
            dates = self.df_raw['date']
        
        # 2. Check Index
        elif isinstance(self.df_raw.index, pd.DatetimeIndex):
            dates = pd.Series(self.df_raw.index)
            
        # 3. Check loose matching (case-insensitive)
        if dates is None:
            possible = ['date', 'time', 'timestamp', 'gmt time', 'time (eet)']
            # Find first column that matches any in 'possible' list
            col = next((c for c in self.df_raw.columns if c.lower() in possible), None)
            if col:
                dates = self.df_raw[col]
        
        if dates is None:
             # Silent fallback or minimal log if critical
             return 15

        try:
            # Convert to datetime if NOT already
            if not pd.api.types.is_datetime64_any_dtype(dates):
                dates = pd.to_datetime(dates, dayfirst=True, format='mixed', errors='coerce')
                
            time_diffs = dates.diff().dropna()
            if len(time_diffs) == 0:
                return 15
                
            avg_diff_minutes = time_diffs.median().total_seconds() / 60
            if avg_diff_minutes <= 0: return 15
            
            return int(avg_diff_minutes)
        except:
            return 15
    
    def _reset_state(self):
        """Khởi tạo lại trạng thái nội bộ của môi trường."""
        self.current_step = self.window_size
        self.balance = self.initial_equity
        self.shares = 0.0
        self.current_weight = 0.0
        self.prev_net_worth = self.initial_equity
        self.steps_in_trade = 0
        self.position_cost = 0.0 # Cost Basis
        
        # Reset feedback variables
        self.last_trade_cost = 0.0
        self.last_vol_scalar = 1.0 # Default Neutral
        self.current_dd_pct = 0.0
        self.prev_dd_pct = 0.0
        self.max_net_worth = self.initial_equity
        
        self.state_builder.reset()
        self.reward_engine.reset()
        self.prev_time = None
        
        # Reset counters
        self.profitable_trades = 0
        self.total_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.sum_exposure = 0.0
        self.step_counter = 0
        self.trade_pnl_acc = 0.0
        self.edge_active = False
        
        # Warm-up StateBuilder
        self._warmup_state_builder()
        
    def reset(self, seed=None, options=None):
        """Đặt lại môi trường cho Episode mới."""
        super().reset(seed=seed)
        self._reset_state()
        
        # Chọn điểm bắt đầu ngẫu nhiên khi Training
        if self.is_training and self.n_steps > self.window_size + 2000:
            self.current_step = np.random.randint(self.window_size, self.n_steps - 1000)
            
        return self._get_current_observation(), {}

    def step(self, action):
        """Thực hiện một bước thời gian trong môi trường."""
        self.last_trade_cost = 0.0 # Reset chi phí của bước này
        
        # 1. Collect Data
        current_price = self.close_arr[self.current_step]
        current_vol_ann = self.vol_ann_arr[self.current_step]
        
        current_value = self.shares * current_price * self.action_handler.lot_size
        net_worth = self.balance + current_value
        
        # 2. Process Action
        action_logit = float(action[0])
        
        # FIX: NaN Action Guard (Bảo vệ môi trường khỏi đầu ra lỗi của mạng thần kinh)
        if np.isnan(action_logit) or np.isinf(action_logit):
            action_logit = 0.0
        
        # [STEP 1: CALCULATE RAW WEIGHT]
        raw_weight = self.action_handler.get_target_weight(
            action_logit, 
            self.vol_scalar_arr[self.current_step],
            current_weight=self.current_weight
        )

        # [STEP 2: EDGE GATE - ALPHA PERMISSION LAYER]
        edge_score = self.compute_edge_score(action_logit)
        
        # Hysteresis Logic
        if edge_score >= self.EDGE_OPEN:
            self.edge_active = True
        elif edge_score <= self.EDGE_CLOSE:
            self.edge_active = False
            
        # Permission Enforcement
        if self.edge_active:
            # Full permission, scale size by edge quality
            target_weight = raw_weight * edge_score
        else:
            if edge_score <= self.EDGE_CLOSE:
                # Absolute exclusion
                target_weight = 0.0
            else:
                # Transition zone: No new opens, only maintain or reduce
                # Reversing sign (Short to Long or Long to Short) is considered "opening new"
                if np.sign(raw_weight) == np.sign(self.current_weight):
                    # Only allow reduction or holding
                    # If current is 0, target stays 0
                    target_weight = np.sign(raw_weight) * min(abs(self.current_weight), abs(raw_weight * edge_score))
                else:
                    target_weight = 0.0
        
        # [LỖI 33 FIX] Check Halt once
        is_halted = self.action_handler.should_halt_trading(current_vol_ann)
        
        # [STEP 3: ACTION INERTIA]
        if is_halted or abs(target_weight) < 1e-4:
            alpha = 1.0 # Thoát lệnh/Halt: Thực thi ngay lập tức
        else:
            alpha = 0.3 if self.is_training else 0.2
            
        target_weight = self.current_weight + alpha * (target_weight - self.current_weight)
        weight_delta = target_weight - self.current_weight
        
        # Theo dõi trạng thái đóng vị thế/đảo lệnh
        is_closing = False
        if abs(self.current_weight) > 1e-4:
             if abs(target_weight) < 1e-4: 
                 is_closing = True
             elif np.sign(target_weight) != np.sign(self.current_weight) and abs(target_weight) > 1e-4: 
                 is_closing = True
        
        should_rebalance = (abs(weight_delta) >= self.REBALANCE_THRESHOLD) and not is_halted
        
        total_real_cost = 0.0
        
        if should_rebalance:
            # Rebalance Logic
            max_notional = net_worth * self.action_handler.max_leverage
            target_value = net_worth * target_weight
            target_value = np.clip(target_value, -max_notional, max_notional)
            
            delta_value = target_value - current_value
            
            # [LỖI 22 & 31 FIX] FILL PRICE MODEL (No Look-ahead)
            # Dùng giá Close HIỆN TẠI (t) thay vì nến tương lai (t+1)
            current_price = self.close_arr[self.current_step]
            
            # [LỖI 16 FIX] Xác định giá khớp chuẩn Broker dựa trên weight_delta (BUY pay Ask, SELL receive Bid)
            order_price = self.action_handler.get_fill_price(current_price, weight_delta, current_vol_ann)
            
            # [LỖI 25 FIX] Calculate Shares (Unify lot_size usage)
            safe_order_price = max(1e-6, order_price)
            shares_delta = delta_value / (safe_order_price * self.action_handler.lot_size)
            ounces_traded = abs(shares_delta * self.action_handler.lot_size)
            
            # [LỖI 12 FIX] Tính phí commission và slippage bổ sung (không trùng spread)
            explicit_fees = self.action_handler.calculate_cost_advanced(
                ounces_traded, 
                current_vol_ann,
                current_price=current_price
            )
            
            prev_shares = self.shares
            self.shares += shares_delta
            
            # CẬP NHẬT BALANCE: Trừ tiền mua shares (tính theo mid) + Phí ngoài
            # Chênh lệch giá (Spread/Slippage) được hạch toán ngầm qua order_price
            self.balance -= (shares_delta * current_price * self.action_handler.lot_size + explicit_fees)
            
            # Phí thực phát sinh (với mục đích báo cáo) = Chênh lệch giá khớp + Phí ngoài
            total_real_cost = (abs(shares_delta * (order_price - current_price) * self.action_handler.lot_size) + explicit_fees)
            self.last_trade_cost = total_real_cost
            
            if self.current_step < len(self.vol_scalar_arr):
                 self.last_vol_scalar = self.vol_scalar_arr[self.current_step]
            else:
                 self.last_vol_scalar = 1.0

            # [LỖI 36 FIX] Update Cost Basis (Handle Flip Position)
            if prev_shares == 0:
                self.position_cost = abs(self.shares) * order_price * self.action_handler.lot_size
            else:
                is_increasing = (shares_delta * prev_shares) > 0 
                if is_increasing:
                    self.position_cost += (abs(shares_delta) * order_price * self.action_handler.lot_size)
                else:
                    if abs(shares_delta) < abs(prev_shares):
                        # Partial Close
                        ratio_reduced = abs(shares_delta) / abs(prev_shares)
                        self.position_cost *= (1.0 - ratio_reduced)
                    else:
                        # Full close or Flip (e.g. Long -> Short)
                        self.position_cost = abs(self.shares) * order_price * self.action_handler.lot_size

            # Update Trade Status
            if abs(target_weight) < 0.01:
                self.steps_in_trade = 0
            else:
                self.steps_in_trade = 1 if abs(self.current_weight) < 0.01 else self.steps_in_trade + 1
        else:
            if abs(self.current_weight) >= 0.01:
                self.steps_in_trade += 1
            
            if self.current_step < len(self.vol_scalar_arr):
                 self.last_vol_scalar = self.vol_scalar_arr[self.current_step]
        
        # Next Step
        self.current_step += 1
        terminated = False
        truncated = False

        # [LỖI 23 & 31 & 35 FIX] Handle Weekend Gaps & Swap
        try:
            ts = self.date_arr[self.current_step]
            current_time = pd.Timestamp(ts)
            
            if self.prev_time is not None:
                # [LỖI 23 FIX] Detect Weekend Gap (> 24h)
                time_diff_hours = (current_time - self.prev_time).total_seconds() / 3600
                if time_diff_hours > 24:
                    # [LỖI 38 FIX] Store prices for accurate Gap analysis (t-1 vs t)
                    prev_bar_price = self.close_arr[self.current_step - 1]
                    curr_bar_price = self.close_arr[self.current_step]
                    gap_pct = abs(curr_bar_price - prev_bar_price) / (prev_bar_price + 1e-9)
                    
                    # Apply Gap Slippage directly to balance
                    pos_val = abs(self.shares * curr_bar_price * self.action_handler.lot_size)
                    gap_penalty = pos_val * gap_pct * 0.5 # Pay half the gap movement as slippage
                    self.balance -= gap_penalty
                    
                    # Force exit/termination if gap > 2%
                    if gap_pct > 0.02:
                        terminated = True
                
                # Apply Swap
                elif current_time.date() > self.prev_time.date():
                    days = (current_time.date() - self.prev_time.date()).days
                    ounces_held = self.shares * self.action_handler.lot_size
                    swap_cost = self.action_handler.calculate_swap_cost(ounces_held, days)
                    self.balance -= swap_cost 
                
            self.prev_time = current_time
        except:
             pass
        
        if self.current_step >= self.n_steps - 1:
            truncated = True
        
        # 4. Tính Net Worth & Kiểm soát rủi ro
        new_price = self.close_arr[self.current_step]
        new_net_worth = self.balance + (self.shares * new_price * self.action_handler.lot_size)
        
        # Equity Integrity Guard
        if not np.isfinite(new_net_worth):
            new_net_worth = self.prev_net_worth
        new_net_worth = np.clip(new_net_worth, -1e9, 1e9)
        
        # Margin Call (Tháo khoán tại 130% Maintenance Margin)
        position_notional = abs(self.shares * new_price * self.action_handler.lot_size)
        if position_notional > 0:
            required_margin = position_notional / self.action_handler.max_leverage
            margin_ratio = self.balance / (required_margin + 1e-9)
            
            if margin_ratio < 1.30: 
                self.balance -= (position_notional * 0.02) # Liquidation penalty
                self.shares = 0.0
                self.current_weight = 0.0
                self.position_cost = 0.0
                terminated = True
        
        # Bankruptcy Check (Lỗ 70% vốn ban đầu)
        if new_net_worth < self.initial_equity * 0.3:
            terminated = True
            
        step_return = (new_net_worth - self.prev_net_worth) / (max(1.0, self.prev_net_worth) + 1e-9)
        # [LỖI 28 FIX] Guard step_return
        if not np.isfinite(step_return): step_return = 0.0
        step_return = np.clip(step_return, -1.0, 1.0)
        
        # 4. Calculate Reward
        if new_net_worth > self.max_net_worth:
            self.max_net_worth = new_net_worth
            
        self.current_dd_pct = (self.max_net_worth - new_net_worth) / (self.max_net_worth + 1e-9) * 100.0
        
        old_net_worth = self.prev_net_worth
        
        # Metrics Update
        step_pnl = (new_net_worth - old_net_worth)
        self.trade_pnl_acc += step_pnl

        if step_pnl > 0:
             self.gross_profit += step_pnl
        else:
             self.gross_loss += abs(step_pnl)
             
        if is_closing:
             self.total_trades += 1
             if self.trade_pnl_acc > 0:
                 self.profitable_trades += 1
             self.trade_pnl_acc = 0.0

        self.sum_exposure += abs(self.current_weight)
        self.step_counter += 1

        # Update Weight Reference BEFORE Reward (for state consistency)
        new_weight = (self.shares * new_price * self.action_handler.lot_size) / (new_net_worth + 1e-9)
        self.current_weight = new_weight
        self.prev_net_worth = new_net_worth
        
        # Smoothing/Capping Vol input for Reward (Preventing noise/explosions)
        current_vol_v = max(self.vol_ann_arr[self.current_step], 0.05)
        
        reward_val, reward_info = self.reward_engine.calculate(
             current_net_worth=new_net_worth,
             prev_net_worth=old_net_worth,
             current_drawdown_pct=self.current_dd_pct,
             prev_drawdown_pct=self.prev_dd_pct,
             current_vol=current_vol_v,
             current_weight=self.current_weight
        )
        
        # Áp dụng hình phạt cực nặng khi cháy tài khoản (Lỗ > 50%)
        if terminated and new_net_worth < self.initial_equity * 0.5:
            reward_val = -100.0
            reward_info['total'] = -100.0
            reward_info['is_bankrupt'] = True
            
        reward_info['regime_idx'] = int(np.argmax(self.regime_probs[self.current_step])) if hasattr(self, 'regime_probs') else 0
        
        self.prev_dd_pct = self.current_dd_pct # Hết lượt: Lưu lại DD để so sánh ở bước sau
        
        current_noise = self.df_features['entropy_raw'].values[self.current_step] if 'entropy_raw' in self.df_features.columns else 0.0
        
        if truncated and abs(self.current_weight) > 1e-4:
             self.total_trades += 1
             if self.trade_pnl_acc > 0:
                 self.profitable_trades += 1
             self.trade_pnl_acc = 0.0

        # Info Dict (Dashboard Protocol)
        info = {
            'step': self.current_step,
            'episode_step': self.current_step,
            'net_worth': new_net_worth,
            'price': new_price,
            'shares': self.shares,
            'last_trade_cost': getattr(self, 'last_trade_cost', 0.0), # Trả về phí thực tế của bước này
            'step_reward': reward_val,
            'equity': new_net_worth,
            'pnl': new_net_worth - self.initial_equity,
            'pnl_pct': (new_net_worth - self.initial_equity) / (self.initial_equity + 1e-9),
            'max_drawdown': self.current_dd_pct,
            'win_rate': self.profitable_trades / (self.total_trades + 1e-9),
            'profit_factor': self.gross_profit / (self.gross_loss + 1e-9),
            'avg_exposure': self.sum_exposure / (self.step_counter if self.step_counter > 0 else 1),
            'num_trades': self.total_trades,
            'reward_decomposition': reward_info,
             'regime_stats': {
                'noise': current_noise,
                'vol': current_vol_ann,
                'macro': self.regime_probs[self.current_step].tolist() if hasattr(self, 'regime_probs') else [],
                'hurst': float(self.indicator_arrays['hurst'][self.current_step]) if 'hurst' in self.indicator_arrays else 0.5,
                'efficiency': float(self.indicator_arrays['efficiency_raw'][self.current_step]) if 'efficiency_raw' in self.indicator_arrays else 0.0,
                'interactions': float(self.indicator_arrays['interaction_trend_vol'][self.current_step]) if 'interaction_trend_vol' in self.indicator_arrays else 0.0,
                'persistence': float(self.indicator_arrays['directional_persistence'][self.current_step]) if 'directional_persistence' in self.indicator_arrays else 0.0,
                'shock': float(self.indicator_arrays['price_shock'][self.current_step]) if 'price_shock' in self.indicator_arrays else 0.0
            },
            'weight': float(self.current_weight),
            'weight_delta': float(weight_delta),
            'action_logit': float(action_logit),
            'step_return': float(step_return),
            'edge_score': float(edge_score),
            'edge_active': bool(self.edge_active)
        }
        
        return self._get_current_observation(), float(reward_val), terminated, truncated, info

    def _get_current_observation(self):
        """Xây dựng quan sát dựa trên nến đã đóng (Causality)."""
        obs_idx = max(0, self.current_step - 1)
        
        price = self.close_arr[obs_idx]
        volume = self.vol_arr[obs_idx]
        
        indicators = {}
        # Lấy các chỉ báo tại obs_idx (nến đã đóng)
        for k, arr in self.indicator_arrays.items():
            indicators[k] = float(arr[obs_idx])
            
        # Feedback State Variables (Context tại bước hiện tại)
        indicators['last_cost_pct'] = self.last_trade_cost / (self.prev_net_worth + 1e-9)
        indicators['last_vol_scalar'] = self.last_vol_scalar

        # Macro Regime (HMM Probability Vector)
        regime_vec = self.regime_probs[obs_idx]
        
        # Portfolio State (Trạng thái HIỆN TẠI dựa trên giá chốt nến vừa quan sát)
        raw_pnl = (self.shares * price * self.action_handler.lot_size) - self.position_cost
        
        port_state = {
            'unrealized_pnl': raw_pnl,
            'equity': self.prev_net_worth, 
            'position': self.current_weight,
            'time_in_trade': min(self.steps_in_trade / 1000.0, 1.0)
        }
        
        # Inject Risk Metrics (Pre-calculated for obs_idx)
        indicators['vol_ratio'] = float(self.vol_ratio_arr[obs_idx])
        indicators['vol_scalar'] = float(self.vol_scalar_arr[obs_idx]) if obs_idx < len(self.vol_scalar_arr) else 1.0
        
        # Chuẩn hóa quan sát (Freeze nếu đang evaluate)
        obs = self.state_builder.process_one_step(
            price, volume, indicators, regime_vec, port_state,
            perform_normalization=True,
            frozen=not self.is_training
        )
        
        return obs

    def compute_edge_score(self, action_logit: float) -> float:
        """
        Tính toán Edge Score dựa trên 5 thành phần rủi ro & hiệu suất.
        Công thức chuẩn Hedge Fund: Alpha Permission Layer.
        """
        idx = self.current_step
        
        # 1. Structural persistence (Hurst)
        hurst = self.indicator_arrays['hurst'][idx] if 'hurst' in self.indicator_arrays else 0.5
        e_hurst = np.clip((hurst - 0.50) / 0.15, 0.0, 1.0)
        
        # 2. Economic efficiency
        eff = self.indicator_arrays['trend_efficiency'][idx] if 'trend_efficiency' in self.indicator_arrays else 0.0
        e_eff = np.clip(abs(eff) / self.EFF_REF, 0.0, 1.0)
        
        # 3. Volatility survivability
        vol_ratio = self.vol_ratio_arr[idx]
        e_vol = np.exp(-abs(vol_ratio - 1.0))
        
        # 4. Noise filter (Entropy)
        entropy = self.indicator_arrays['entropy_raw'][idx] if 'entropy_raw' in self.indicator_arrays else 0.0
        e_entropy = np.clip(1.0 - (entropy / self.ENTROPY_MAX), 0.0, 1.0)
        
        # 5. Directional validity
        expected_dir = np.sign(action_logit)
        persistence = self.indicator_arrays['directional_persistence'][idx] if 'directional_persistence' in self.indicator_arrays else 0.0
        # Valid if agent intention matches recent persistence
        e_dir = 1.0 if (persistence * expected_dir) > 0 else 0.0
        
        # Weighted Score
        score = (
            0.30 * e_hurst +
            0.25 * e_eff +
            0.20 * e_vol +
            0.15 * e_entropy +
            0.10 * e_dir
        )
        return float(score)

    def _warmup_state_builder(self):
        """Khởi động StateBuilder bằng dữ liệu lịch sử trước khi bắt đầu Episode."""
        start_idx = max(0, self.current_step - self.window_size - 1)
        end_idx = max(0, self.current_step - 1)
        
        for i in range(start_idx, end_idx):
            price = self.close_arr[i]
            volume = self.vol_arr[i]
            regime_v = self.regime_probs[i]
            
            indicators = {}
            for k, arr in self.indicator_arrays.items():
                indicators[k] = float(arr[i])
            
            # Khởi tạo trạng thái danh mục mặc định
            port_state = {
                'unrealized_pnl': 0.0,
                'equity': self.initial_equity,
                'position': 0.0,
                'time_in_trade': 0.0,
            }
            
            # Truyền các chỉ số rủi ro cho bước Warmup
            indicators['vol_ratio'] = float(self.vol_ratio_arr[i])
            indicators['vol_scalar'] = float(self.vol_scalar_arr[i]) if i < len(self.vol_scalar_arr) else 1.0
            indicators['last_vol_scalar'] = 1.0
            indicators['last_cost_pct'] = 0.0
            self.state_builder.process_one_step(
                price, volume, indicators, regime_v, port_state,
                perform_normalization=True,
                frozen=False # Cập nhật thống kê trong warmup
            )

