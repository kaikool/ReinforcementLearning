import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

from core.feature_factory import QuantFeatureFactory
from core.regime import MarketRegime
from core.reward import CompoundReward
from core.actions import ActionHandler
from core.state_builder import StateBuilder

@dataclass
class EnvConfig:
    """Cấu hình vận hành (Operational Configuration) cho AdvancedTradingEnv."""
    # Quản lý vị thế
    rebalance_threshold: float = 0.03
    action_inertia_train: float = 0.3
    action_inertia_eval: float = 0.2
    
    # Quản trị rủi ro
    margin_call_threshold: float = 1.30
    liquidation_penalty: float = 0.02
    bankruptcy_threshold: float = 0.30  # Lỗ 70% vốn
    
    # Mô hình chi phí
    weekend_gap_threshold: float = 0.02 # 2% gap
    gap_slippage_factor: float = 0.5    # Trả 50% mức nhảy giá làm slippage
    
    # Alpha Permission (Edge Gate)
    edge_open_threshold: float = 0.65
    edge_close_threshold: float = 0.45
    eff_reference: float = 0.15
    edge_weights: Tuple[float, ...] = (0.30, 0.25, 0.20, 0.15, 0.10)
    
    # Vận hành Episode
    min_episode_length: int = 1000
    random_start_buffer: int = 2000
    observation_lag: int = 1  # 1-bar lag for causality (mặc định)

class AdvancedTradingEnv(gym.Env):
    """
    Môi trường giao dịch XAUUSD tối ưu hóa cho RL (V9 - Clean Architecture).
    Tính năng: Modular Logic, Dynamic Config, Alpha Permission Layer.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_factory: QuantFeatureFactory,
        regime_model: MarketRegime,
        config: Optional[EnvConfig] = None,
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
        self.config = config or EnvConfig()
        self.is_training = is_training
        
        # 1. Cơ sở dữ liệu
        self.df_raw = df.copy()
        self._prepare_raw_data()

        self.factory = feature_factory
        self.regime = regime_model
        self.initial_equity = initial_equity
        self.window_size = window_size
        
        # 2. Thành phần Logic
        self.state_builder = StateBuilder(window_size=window_size)
        self.reward_engine = CompoundReward(initial_equity=initial_equity)
        self.action_handler = ActionHandler(
            target_vol=0.15, 
            max_leverage=5.0,
            spread_usd=spread_usd,
            commission_usd=commission_usd,
            lot_size=lot_size
        )
        self.lot_size = lot_size
        
        # 3. Xử lý Đặc trưng (Features)
        self._initialize_features(df_features, regime_probs)
        
        # 4. Phởi tạo trạng thái lần đầu
        self._reset_state()

        # 5. Observation & Action Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        dummy_obs = self._get_current_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=dummy_obs.shape, 
            dtype=np.float32
        )

    def _prepare_raw_data(self):
        """Đảm bảo định dạng index và cột date chuẩn xác."""
        if isinstance(self.df_raw.index, pd.DatetimeIndex):
            self.df_raw.reset_index(inplace=True)
            if 'date' not in self.df_raw.columns:
                self.df_raw.rename(columns={self.df_raw.columns[0]: 'date'}, inplace=True)
        else:
            self.df_raw.reset_index(drop=True, inplace=True)

    def _initialize_features(self, df_features, regime_probs):
        """Khởi tạo toàn bộ mảng đặc trưng và trạng thái thị trường."""
        if df_features is not None:
            self.df_features = df_features.copy()
        else:
            self.df_features = self.factory.process_data(self.df_raw)
        
        self.close_arr = self.df_features['close'].values.astype(np.float32)
        self.vol_arr = self.df_features['volume'].values.astype(np.float32)
        
        # GMT/Date handling
        if 'date' in self.df_features.columns:
            self.date_arr = pd.to_datetime(self.df_features['date'], dayfirst=True, format='mixed').values
        else:
            self.date_arr = self.df_features.index.values
            
        # Indicator Extraction
        indicator_keys = [
            'log_ret', 'realized_vol', 'rsi_raw', 'macd_raw', 'cci_raw', 'atr_rel', 
            'bb_pct', 'vol_rel', 'is_active_hour', 'entropy_raw', 'efficiency_raw',
            'hurst', 'entropy_delta', 'price_shock', 'interaction_trend_vol', 
            'trend_efficiency', 'directional_persistence'
        ]
        self.indicator_arrays = {k: self.df_features[k].values.astype(np.float32) 
                               for k in indicator_keys if k in self.df_features.columns}
        
        # Timeframe & Annualization
        self.timeframe_minutes = self._detect_timeframe()
        bars_per_day = 1440 / (self.timeframe_minutes or 15)
        self.annualization_factor = np.sqrt(252 * bars_per_day)
        
        # Risk Metric Pre-calculation
        self._precalculate_risk_metrics()
        
        # Market Regime
        self._initialize_regime(regime_probs)
        
        self.n_steps = len(self.df_features)
        
        # Constants extraction from data
        if 'entropy_raw' in self.indicator_arrays:
            self.ENTROPY_MAX = np.percentile(self.indicator_arrays['entropy_raw'], 75)
            if self.ENTROPY_MAX < 1e-6: self.ENTROPY_MAX = 0.5
        else:
            self.ENTROPY_MAX = 0.5

    def _precalculate_risk_metrics(self):
        """Tính toán trước các tham số Vol Scalar và Vol Ratio."""
        if 'realized_vol' in self.df_features.columns:
            self.vol_ann_arr = self.df_features['realized_vol'].values.astype(np.float32) * self.annualization_factor
        else:
            self.vol_ann_arr = np.full(len(self.df_features), 0.20, dtype=np.float32)

        safe_vol = np.maximum(self.vol_ann_arr, 0.01)
        self.vol_scalar_arr = np.clip(self.action_handler.target_vol / safe_vol, 0.0, self.action_handler.max_leverage)
        self.indicator_arrays['vol_scalar'] = self.vol_scalar_arr.astype(np.float32)
        self.vol_ratio_arr = (safe_vol / self.action_handler.target_vol).astype(np.float32)
        self.indicator_arrays['vol_ratio'] = self.vol_ratio_arr

    def _initialize_regime(self, regime_probs):
        """Khởi tạo xác suất Regime trạng thái thị trường."""
        if regime_probs is not None:
            self.regime_probs = regime_probs
        else:
            regime_cols = ['log_ret', 'realized_vol', 'entropy_raw', 'abs_log_ret']
            if 'abs_log_ret' not in self.df_features.columns:
                self.df_features['abs_log_ret'] = self.df_features['log_ret'].abs()
            regime_data = self.df_features[regime_cols].fillna(0.0).values
            self.regime_probs = self.regime.predict_proba_causal(regime_data)

    def _detect_timeframe(self):
        """Tự động nhận diện khung thời gian."""
        try:
            dates = pd.to_datetime(pd.Series(self.date_arr), dayfirst=True, format='mixed', errors='coerce')
            time_diffs = dates.diff().dropna()
            if len(time_diffs) == 0: return 15
            avg_diff_minutes = time_diffs.median().total_seconds() / 60
            return int(avg_diff_minutes) if avg_diff_minutes > 0 else 15
        except:
            return 15
    
    def _reset_state(self):
        """Khởi tạo lại trạng thái nội bộ."""
        self.current_step = self.window_size
        self.balance = self.initial_equity
        self.shares = 0.0
        self.current_weight = 0.0
        self.prev_net_worth = self.initial_equity
        self.steps_in_trade = 0
        self.position_cost = 0.0
        self.last_trade_cost = 0.0
        self.last_vol_scalar = 1.0
        self.current_dd_pct = 0.0
        self.prev_dd_pct = 0.0
        self.max_net_worth = self.initial_equity
        self.prev_time = None
        self.profitable_trades = 0
        self.total_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.sum_exposure = 0.0
        self.step_counter = 0
        self.trade_pnl_acc = 0.0
        self.edge_active = False
        self.state_builder.reset()
        self.reward_engine.reset()
        self._warmup_state_builder()
        
    def reset(self, seed=None, options=None):
        """Đặt lại môi trường cho Episode mới."""
        super().reset(seed=seed)
        self._reset_state()
        if self.is_training and self.n_steps > self.window_size + self.config.random_start_buffer:
            self.current_step = np.random.randint(self.window_size, self.n_steps - self.config.min_episode_length)
        return self._get_current_observation(), {}

    def step(self, action):
        """Thực hiện một bước thời gian (V9 Modular Implementation)."""
        self.last_trade_cost = 0.0
        current_price = self.close_arr[self.current_step]
        current_vol_ann = self.vol_ann_arr[self.current_step]
        
        # 1. Action Processing & Alpha Permission
        action_logit = self._validate_action(action)
        target_weight = self._calculate_target_weight(action_logit, current_vol_ann)
        
        # 2. Execution logic
        weight_delta = target_weight - self.current_weight
        is_closing = self._detect_closing_event(target_weight)
        
        self._execute_trade(target_weight, weight_delta, current_price, current_vol_ann)
        
        # 3. Time passage & Special events
        self.current_step += 1
        terminated, truncated = self._handle_time_events()
        
        # 4. Risk guards & PnL
        new_price = self.close_arr[self.current_step]
        new_net_worth = self._calculate_net_worth(new_price)
        terminated = terminated or self._check_risk_limits(new_net_worth, new_price)
        
        # 5. Metrics & Rewards
        self._update_metrics(new_net_worth, target_weight, weight_delta, is_closing)
        reward_val, reward_info = self._calculate_reward(new_net_worth, terminated)
        
        # 6. Final Outputs
        self.prev_net_worth = new_net_worth
        self.prev_dd_pct = self.current_dd_pct
        
        return self._get_current_observation(), float(reward_val), terminated, truncated, self._build_info_dict(new_net_worth, new_price, reward_val, reward_info, action_logit, weight_delta, current_vol_ann)

    # --- Helper Methods for step() ---

    def _validate_action(self, action: np.ndarray) -> float:
        """Kiểm tra và ngăn chặn giá trị NaN/Inf từ mạng thần kinh."""
        logit = float(action[0])
        return 0.0 if np.isnan(logit) or np.isinf(logit) else logit

    def _calculate_target_weight(self, action_logit: float, current_vol_ann: float) -> float:
        """Tính toán tỷ trọng mục tiêu sau khi đi qua Alpha Permission Layer (Edge Gate)."""
        raw_weight = self.action_handler.get_target_weight(
            action_logit, self.vol_scalar_arr[self.current_step],
            current_weight=self.current_weight
        )
        edge_score = self.compute_edge_score(action_logit)
        
        # Hysteresis Logic
        if edge_score >= self.config.edge_open_threshold:
            self.edge_active = True
        elif edge_score <= self.config.edge_close_threshold:
            self.edge_active = False
            
        if self.edge_active:
            target_weight = raw_weight * edge_score
        else:
            if edge_score <= self.config.edge_close_threshold:
                target_weight = 0.0
            else:
                # Transition zone: Duy trì hoặc giảm, không mở mới
                if np.sign(raw_weight) == np.sign(self.current_weight):
                    target_weight = np.sign(raw_weight) * min(abs(self.current_weight), abs(raw_weight * edge_score))
                else:
                    target_weight = 0.0
        
        # Action Inertia (Smoothing)
        is_halted = self.action_handler.should_halt_trading(current_vol_ann)
        alpha = 1.0 if (is_halted or abs(target_weight) < 1e-4) else (self.config.action_inertia_train if self.is_training else self.config.action_inertia_eval)
        
        return self.current_weight + alpha * (target_weight - self.current_weight)

    def _detect_closing_event(self, target_weight: float) -> bool:
        """Phát hiện nếu agent đang đóng vị thế cũ."""
        if abs(self.current_weight) < 1e-4: return False
        if abs(target_weight) < 1e-4: return True
        return np.sign(target_weight) != np.sign(self.current_weight)

    def _execute_trade(self, target_weight: float, weight_delta: float, current_price: float, current_vol: float):
        """Thực thi lệnh giao dịch và cập nhật giá vốn (Cost Basis)."""
        should_rebalance = (abs(weight_delta) >= self.config.rebalance_threshold) and not self.action_handler.should_halt_trading(current_vol)
        
        if not should_rebalance:
            if abs(self.current_weight) >= 0.01: self.steps_in_trade += 1
            self.last_vol_scalar = self.vol_scalar_arr[self.current_step]
            return

        net_worth = self.balance + (self.shares * current_price * self.action_handler.lot_size)
        target_value = np.clip(net_worth * target_weight, -net_worth * self.action_handler.max_leverage, net_worth * self.action_handler.max_leverage)
        current_value = self.shares * current_price * self.action_handler.lot_size
        delta_value = target_value - current_value
        
        order_price = self.action_handler.get_fill_price(current_price, weight_delta, current_vol)
        shares_delta = delta_value / (max(1e-6, order_price) * self.action_handler.lot_size)
        ounces_traded = abs(shares_delta * self.action_handler.lot_size)
        
        fees = self.action_handler.calculate_cost_advanced(ounces_traded, current_vol, current_price=current_price)
        
        # Update Balance & Shares
        prev_shares = self.shares
        self.shares += shares_delta
        self.balance -= (shares_delta * current_price * self.action_handler.lot_size + fees)
        self.last_trade_cost = abs(shares_delta * (order_price - current_price) * self.action_handler.lot_size) + fees
        self.last_vol_scalar = self.vol_scalar_arr[self.current_step]
        
        self._update_cost_basis(prev_shares, shares_delta, order_price)
        self.steps_in_trade = 0 if abs(target_weight) < 0.01 else (1 if abs(self.current_weight) < 0.01 else self.steps_in_trade + 1)
        self.current_weight = (self.shares * current_price * self.action_handler.lot_size) / (net_worth + 1e-9)

    def _update_cost_basis(self, prev_shares: float, shares_delta: float, order_price: float):
        """Tính toán lại giá vốn dựa trên thay đổi khối lượng."""
        if prev_shares == 0:
            self.position_cost = abs(self.shares) * order_price * self.action_handler.lot_size
        else:
            if (shares_delta * prev_shares) > 0: # Tăng quy mô
                self.position_cost += (abs(shares_delta) * order_price * self.action_handler.lot_size)
            else: # Giảm/Đảo quy mô
                if abs(shares_delta) < abs(prev_shares):
                    self.position_cost *= (1.0 - abs(shares_delta) / abs(prev_shares))
                else:
                    self.position_cost = abs(self.shares) * order_price * self.action_handler.lot_size

    def _handle_time_events(self) -> Tuple[bool, bool]:
        """Xử lý Gap cuối tuần, Swap và kiểm tra kết thúc dữ liệu."""
        terminated, truncated = False, False
        if self.current_step >= self.n_steps - 1:
            return False, True
            
        try:
            curr_time = pd.Timestamp(self.date_arr[self.current_step])
            if self.prev_time is not None:
                # Gap detect
                if (curr_time - self.prev_time).total_seconds() / 3600 > 24:
                    gap_pct = abs(self.close_arr[self.current_step] - self.close_arr[self.current_step-1]) / (self.close_arr[self.current_step-1] + 1e-9)
                    self.balance -= abs(self.shares * self.close_arr[self.current_step] * self.action_handler.lot_size) * gap_pct * self.config.gap_slippage_factor
                    if gap_pct > self.config.weekend_gap_threshold: terminated = True
                # Swap detect
                elif curr_time.date() > self.prev_time.date():
                    self.balance -= self.action_handler.calculate_swap_cost(self.shares * self.action_handler.lot_size, (curr_time.date() - self.prev_time.date()).days)
            self.prev_time = curr_time
        except: pass
        return terminated, truncated

    def _calculate_net_worth(self, price: float) -> float:
        """Tính toán tổng tài sản ròng và giới hạn tài chính."""
        nw = self.balance + (self.shares * price * self.action_handler.lot_size)
        return float(np.clip(nw, -1e9, 1e9)) if np.isfinite(nw) else float(self.prev_net_worth)

    def _check_risk_limits(self, net_worth: float, price: float) -> bool:
        """Kiểm tra Margin Call và Phá sản (Bankruptcy)."""
        pos_notional = abs(self.shares * price * self.action_handler.lot_size)
        if pos_notional > 0:
            margin_ratio = self.balance / (pos_notional / self.action_handler.max_leverage + 1e-9)
            if margin_ratio < self.config.margin_call_threshold:
                self.balance -= (pos_notional * self.config.liquidation_penalty)
                self.shares, self.current_weight, self.position_cost = 0.0, 0.0, 0.0
                return True
        return net_worth < self.initial_equity * self.config.bankruptcy_threshold

    def _update_metrics(self, net_worth: float, target_weight: float, weight_delta: float, is_closing: bool):
        """Cập nhật các chỉ số hiệu suất nội bộ."""
        step_pnl = net_worth - self.prev_net_worth
        self.trade_pnl_acc += step_pnl
        if step_pnl > 0: self.gross_profit += step_pnl
        else: self.gross_loss += abs(step_pnl)
        
        if is_closing:
            self.total_trades += 1
            if self.trade_pnl_acc > 0: self.profitable_trades += 1
            self.trade_pnl_acc = 0.0
            
        self.sum_exposure += abs(self.current_weight)
        self.step_counter += 1
        self.max_net_worth = max(self.max_net_worth, net_worth)
        self.current_dd_pct = (self.max_net_worth - net_worth) / (self.max_net_worth + 1e-9) * 100.0

    def _calculate_reward(self, net_worth: float, terminated: bool) -> Tuple[float, Dict]:
        """Tính toán phần thưởng đa nhân tố thông qua Reward Engine."""
        current_vol_v = max(self.vol_ann_arr[self.current_step], 0.05)
        reward_val, reward_info = self.reward_engine.calculate(
             current_net_worth=net_worth,
             prev_net_worth=self.prev_net_worth,
             current_drawdown_pct=self.current_dd_pct,
             prev_drawdown_pct=self.prev_dd_pct,
             current_vol=current_vol_v
        )
        if terminated and net_worth < self.initial_equity * 0.5:
            reward_val = -100.0
            reward_info.update({'total': -100.0, 'is_bankrupt': True})
        reward_info['regime_idx'] = int(np.argmax(self.regime_probs[self.current_step])) if hasattr(self, 'regime_probs') else 0
        return float(reward_val), reward_info

    def _build_info_dict(self, nw, price, reward_val, reward_info, logit, delta, vol_ann) -> Dict:
        """Xây dựng Info Dict đồng nhất để Dashboard trích xuất dữ liệu."""
        return {
            'step': self.current_step,
            'net_worth': nw,
            'equity': nw,
            'price': price,
            'shares': self.shares,
            'step_reward': reward_val,
            'pnl_pct': (nw - self.initial_equity) / (self.initial_equity + 1e-9),
            'max_drawdown': self.current_dd_pct,
            'win_rate': self.profitable_trades / (self.total_trades + 1e-9),
            'profit_factor': self.gross_profit / (self.gross_loss + 1e-9),
            'num_trades': self.total_trades,
            'reward_decomposition': reward_info,
            'regime_stats': self._get_regime_stats(vol_ann),
            'weight': float(self.current_weight),
            'action_logit': float(logit),
            'edge_score': self.compute_edge_score(logit),
            'edge_active': bool(self.edge_active)
        }

    def _get_regime_stats(self, vol_ann: float) -> Dict:
        """Trích xuất trạng thái thị trường chi tiết."""
        idx = self.current_step
        return {
            'vol': vol_ann,
            'macro': self.regime_probs[idx].tolist() if hasattr(self, 'regime_probs') else [],
            'hurst': float(self.indicator_arrays['hurst'][idx]),
            'efficiency': float(self.indicator_arrays['efficiency_raw'][idx]),
            'persistence': float(self.indicator_arrays['directional_persistence'][idx]),
            'shock': float(self.indicator_arrays['price_shock'][idx])
        }

    def _get_current_observation(self):
        """Xây dựng quan sát dựa trên độ trễ cấu hình (Causal Lag)."""
        idx = max(0, self.current_step - self.config.observation_lag)
        price = self.close_arr[idx]
        indicators = {k: float(arr[idx]) for k, arr in self.indicator_arrays.items()}
        indicators.update({
            'last_cost_pct': self.last_trade_cost / (self.prev_net_worth + 1e-9),
            'last_vol_scalar': self.last_vol_scalar,
            'vol_ratio': float(self.vol_ratio_arr[idx]),
            'vol_scalar': float(self.vol_scalar_arr[idx])
        })
        port_state = {
            'unrealized_pnl': (self.shares * price * self.action_handler.lot_size) - self.position_cost,
            'equity': self.prev_net_worth, 
            'position': self.current_weight,
            'time_in_trade': min(self.steps_in_trade / 1000.0, 1.0)
        }
        return self.state_builder.process_one_step(price, self.vol_arr[idx], indicators, self.regime_probs[idx], port_state, perform_normalization=True, frozen=not self.is_training)

    def compute_edge_score(self, action_logit: float) -> float:
        """Alpha Permission Layer - Tính toán chất lượng tín hiệu."""
        idx = self.current_step
        w = self.config.edge_weights
        e_hurst = np.clip((self.indicator_arrays['hurst'][idx] - 0.50) / 0.15, 0.0, 1.0)
        e_eff = np.clip(abs(self.indicator_arrays['trend_efficiency'][idx]) / self.config.eff_reference, 0.0, 1.0)
        e_vol = np.exp(-abs(self.vol_ratio_arr[idx] - 1.0))
        e_entropy = np.clip(1.0 - (self.indicator_arrays['entropy_raw'][idx] / self.ENTROPY_MAX), 0.0, 1.0)
        e_dir = 1.0 if (self.indicator_arrays['directional_persistence'][idx] * np.sign(action_logit)) > 0 else 0.0
        return float(w[0]*e_hurst + w[1]*e_eff + w[2]*e_vol + w[3]*e_entropy + w[4]*e_dir)

    def _warmup_state_builder(self):
        """Khởi động StateBuilder bằng dữ liệu lịch sử."""
        for i in range(max(0, self.current_step - self.window_size - 1), max(0, self.current_step - 1)):
            price, volume, regime_v = self.close_arr[i], self.vol_arr[i], self.regime_probs[i]
            indicators = {k: float(arr[i]) for k, arr in self.indicator_arrays.items()}
            indicators.update({'vol_ratio': float(self.vol_ratio_arr[i]), 'vol_scalar': float(self.vol_scalar_arr[i]), 'last_vol_scalar': 1.0, 'last_cost_pct': 0.0})
            self.state_builder.process_one_step(price, volume, indicators, regime_v, {'unrealized_pnl': 0.0, 'equity': self.initial_equity, 'position': 0.0, 'time_in_trade': 0.0}, perform_normalization=True, frozen=False)
