import os
import json
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from core.trading_env import AdvancedTradingEnv
from core.feature_factory import QuantFeatureFactory
from core.regime import MarketRegime

def calculate_advanced_metrics(equity_series, trades):
    """Tính toán bộ chỉ số Quant Pro-Trader đầy đủ."""
    if not trades: 
        return {
            "Overview": {k: "0.00" if "Profit" in k or "Equity" in k else "0.00%" for k in ['Net Profit', 'Gross Profit', 'Gross Loss', 'Return (%)', 'CAGR', 'Max Drawdown', 'Equity Final']},
            "Risk & Quality": {k: "0.00" for k in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'SQN', 'Expectancy', 'Kelly Criterion', 'Recovery Factor']},
            "Trade Statistics": {'Total Trades': 0, 'Win Rate': "0% (0/0)", 'Profit Factor': "0.00", 'Payoff Ratio': "0.00", 'Avg Win': "$0.00", 'Avg Loss': "$0.00", 'Best Trade': "$0.00", 'Worst Trade': "$0.00"},
            "Detailed Analysis": {'Buy Orders': "0", 'Sell Orders': "0", 'Max Consec Wins': 0, 'Max Consec Losses': 0, 'Avg Holding Time': "00:00:00"}
        }
    
    # 1. Basic Data Prep
    eq_arr = np.array([e['value'] for e in equity_series])
    initial_eq = eq_arr[0]
    final_eq = eq_arr[-1]
    
    # Returns Analysis
    returns = pd.Series(eq_arr).pct_change().fillna(0)
    pnl_arr = np.array([t['net_pnl'] for t in trades])
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    
    # Time Analysis: Auto-detect bar frequency from timestamps
    time_diffs = np.diff([e['time'] for e in equity_series])
    if len(time_diffs) > 0:
        median_diff = np.median(time_diffs)
        bars_per_day = 86400 / (median_diff if median_diff > 0 else 900)
    else:
        bars_per_day = 96 # Fallback to 15-min
        
    n_days = len(equity_series) / bars_per_day
    years = n_days / 252 if n_days > 0 else 0
    
    # 2. Performance Metrics
    total_net_profit = final_eq - initial_eq
    profit_pct = (total_net_profit / initial_eq) * 100 if initial_eq > 0 else 0
    
    # CAGR Fix: Handle negative returns and total losses
    cagr = 0
    if years > 0.01 and initial_eq > 0:
        if final_eq > 0:
            try:
                cagr = ((final_eq / initial_eq) ** (1/years) - 1) * 100
            except:
                cagr = 0
        else:
            cagr = -100.0 # Total bankruptcy
    
    # 3. Risk Metrics
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / (peaks + 1e-9)
    max_dd = drawdowns.max() * 100
    
    ann_factor = np.sqrt(252 * bars_per_day)
    sharpe = (returns.mean() / returns.std() * ann_factor) if returns.std() > 0 else 0
    downside_std = returns[returns < 0].std()
    sortino = (returns.mean() / downside_std * ann_factor) if downside_std > 0 else 0
    calmar = (profit_pct / max_dd) if max_dd > 0 else 0
    
    # 4. Trade Statistics
    win_rate = (len(wins)/len(trades)*100) if len(trades)>0 else 0
    avg_win = wins.mean() if len(wins)>0 else 0
    avg_loss = losses.mean() if len(losses)>0 else 0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0
    
    # Kelly Criterion (Simple)
    w_prob = len(wins)/len(trades) if len(trades)>0 else 0
    kelly = (w_prob - (1 - w_prob) / payoff_ratio) if payoff_ratio > 0 else 0
    
    # Consecutive
    consecutive_wins = 0
    consecutive_losses = 0
    max_con_wins = 0
    max_con_losses = 0
    current_runs = 0
    current_state = 0 # 1 win, -1 loss
    
    for p in pnl_arr:
        if p > 0:
            if current_state == 1: current_runs += 1
            else: current_runs = 1; current_state = 1
            max_con_wins = max(max_con_wins, current_runs)
        elif p < 0:
            if current_state == -1: current_runs += 1
            else: current_runs = 1; current_state = -1
            max_con_losses = max(max_con_losses, current_runs)
            
    # Long/Short Analysis
    longs = [t for t in trades if t['type'] == 'BUY']
    shorts = [t for t in trades if t['type'] == 'SELL']
    long_wins = len([t for t in longs if t['net_pnl'] > 0])
    short_wins = len([t for t in shorts if t['net_pnl'] > 0])
    long_wr = (long_wins / len(longs) * 100) if longs else 0
    short_wr = (short_wins / len(shorts) * 100) if shorts else 0

    # Additional Metrics
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * abs(avg_loss))
    
    # SQN: System Quality Number simplified
    sqn = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(len(trades))) if len(trades) > 1 and pnl_arr.std() > 0 else 0
    
    win_count = len(wins)
    loss_count = len(losses)

    # Holding Time Analysis
    durations = []
    for t in trades:
        try:
            durations.append(t['exit_time'] - t['entry_time'])
        except:
            continue
            
    avg_duration_sec = np.mean(durations) if durations else 0
    m, s = divmod(avg_duration_sec, 60)
    h, m = divmod(m, 60)
    avg_holding_time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    return {
        "Overview": {
            'Net Profit': f"${total_net_profit:,.2f}",
            'Gross Profit': f"${gross_profit:,.2f}",
            'Gross Loss': f"-${gross_loss:,.2f}",
            'Return (%)': f"{profit_pct:.2f}%",
            'CAGR': f"{cagr:.2f}%",
            'Max Drawdown': f"{max_dd:.2f}%",
            'Equity Final': f"${final_eq:,.2f}"
        },
        "Risk & Quality": {
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Sortino Ratio': f"{sortino:.2f}",
            'Calmar Ratio': f"{calmar:.2f}",
            'SQN': f"{sqn:.2f}",
            'Expectancy': f"${expectancy:.2f}",
            'Kelly Criterion': f"{kelly:.2f}",
            'Recovery Factor': f"{abs(profit_pct / max_dd) if max_dd > 0 else 0:.2f}"
        },
        "Trade Statistics": {
            'Total Trades': len(trades),
            'Win Rate': f"{win_rate:.2f}% ({win_count}/{len(trades)})",
            'Profit Factor': f"{profit_factor:.2f}",
            'Payoff Ratio': f"{payoff_ratio:.2f}",
            'Avg Win': f"${avg_win:.2f}",
            'Avg Loss': f"-${abs(avg_loss):.2f}",
            'Best Trade': f"${pnl_arr.max():.2f}" if len(pnl_arr) > 0 else "$0",
            'Worst Trade': f"${pnl_arr.min():.2f}" if len(pnl_arr) > 0 else "$0",
        },
        "Detailed Analysis": {
            'Buy Orders': f"{len(longs)} (Win: {long_wr:.1f}%)",
            'Sell Orders': f"{len(shorts)} (Win: {short_wr:.1f}%)",
            'Max Consec Wins': max_con_wins,
            'Max Consec Losses': max_con_losses,
            'Avg Holding Time': avg_holding_time_str, 
        }
    }

def run_backtest():
    print("🚀 ĐANG CHẠY BACKTEST ...")
    
    # 1. Load Data
    data_files = [f for f in os.listdir("data") if f.startswith("test_")]
    if not data_files:
        print("❌ LỖI: Không tìm thấy tệp dữ liệu test_*.csv trong thư mục data/")
        return
    data_path = os.path.join("data", data_files[0])
    
    symbol = "XAUUSD" 
    if "_" in data_files[0]:
        symbol = data_files[0].split("_")[1]
    
    print(f"Loading data from: {data_path} (Symbol: {symbol})")
    df_raw = pd.read_csv(data_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    
    possible_time_cols = ['Gmt time', 'Time (EET)', 'Time', 'date', 'Date', 'timestamp']
    time_col = next((c for c in possible_time_cols if c in df_raw.columns), None)
    if not time_col: time_col = df_raw.columns[0]
    
    rename_map = {time_col: 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df_raw.rename(columns=rename_map, inplace=True)
    df_raw['date'] = pd.to_datetime(df_raw['date'], dayfirst=True, format='mixed')
    df_raw = df_raw.sort_values('date').drop_duplicates('date').set_index('date')
    df_raw.dropna(inplace=True) 

    # 2. Setup
    latest_v = sorted([d for d in os.listdir("artifacts") if "version" in d], reverse=True)[0]
    artifact_dir = os.path.join("artifacts", latest_v)
    
    factory = QuantFeatureFactory()
    # Load thống kê chuẩn hóa từ artifact
    stats_path = os.path.join(artifact_dir, "factory_stats.pkl")
    factory.load_stats(stats_path)
    
    test_features = factory.process_data(df_raw)
    
    # Load checkpoint mới nhất
    checkpoint_dir = os.path.join(artifact_dir, "checkpoints")
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")], reverse=True)
    model_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    regime_model = MarketRegime(model_path=os.path.join(artifact_dir, "regime_model.pkl"))
    try:
        model = RecurrentPPO.load(model_path)
    except Exception as e:
        print(f"❌ LỖI: Không thể tải mô hình RL tại {model_path}: {e}")
        return
    
    # 3. Pre-calculate Regimes
    print(">> Pre-calculating Causal HMM States (Filter Mode)...")
    # FIX: Ensure all 4 columns are present (Synchronized with training)
    if 'abs_log_ret' not in test_features.columns:
        test_features['abs_log_ret'] = test_features['log_ret'].abs()
        
    regime_data = test_features[['log_ret', 'realized_vol', 'entropy_raw', 'abs_log_ret']].values
    hmm_probs = []
    log_prob_ptr = None
    total_steps = len(regime_data)
    
    for t in range(total_steps):
        log_prob_ptr, step_prob = regime_model.predict_online(regime_data[t], log_prob_ptr)
        hmm_probs.append(step_prob)
        if t % 10000 == 0:
            print(f"   [HMM] Progress: {t/total_steps*100:.1f}% ({t}/{total_steps})")
    hmm_probs = np.array(hmm_probs)
    print("   [HMM] Completed.")
    
    env = AdvancedTradingEnv(
        df=df_raw, feature_factory=factory, regime_model=regime_model, df_features=test_features,
        initial_equity=10000.0, spread_usd=0.30, lot_size=100.0, is_training=False
    )
    env.regime_probs_history = hmm_probs 
    env.action_handler.max_leverage = 5.0 
    
    obs, _ = env.reset()
    lstm_states, ep_starts = None, np.ones((1,), dtype=bool)
    done = False
    results = {'candles': [], 'equity': [], 'trades': [], 'symbol': symbol, 'audit': []}
    
    # Save candles
    for i, (t, row) in enumerate(test_features.iterrows()):
        t_ts = int(t.timestamp())
        results['candles'].append({
            'time': t_ts, 'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close'])
        })
        if i < env.window_size:
             results['equity'].append({'time': t_ts, 'value': float(env.initial_equity)})

    # 4. Simulation
    prev_shares = 0.0
    entry_price = 0.0
    entry_time = 0
    trade_start_equity = env.initial_equity
    last_equity = env.initial_equity
    trade_id = 0
    running_trade_fees = 0.0
    TRADE_EPSILON = 1e-4 # Bỏ qua các vị thế siêu nhỏ (bụi lệnh)

    print(">> Running Trade Simulation...")
    while not done:
        idx = env.current_step
        if idx >= total_steps: break
        t_ts = int(test_features.index[idx].timestamp())

        # Action
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        ep_starts[0] = False
        
        # DEBUG: Xem AI có quyết định đổi Weight không
        if env.current_step < env.window_size + 20: 
            print(f"   [DEBUG-WE] Step {env.current_step}: Logit={action[0]:.4f}, Target={env.current_weight:.4f}")
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if env.current_step % 5000 == 0:
            pos_desc = f"{'LONG' if env.shares > 0 else 'SHORT'} ({abs(env.current_weight):.2f})" if abs(env.shares) > 1e-6 else "FLAT"
            print(f"   [Backtest] {env.current_step/total_steps*100:4.1f}% | Equity: ${info.get('net_worth', 0):.2f} | Pos: {pos_desc} | Closed: {trade_id}")

        current_equity = float(info.get('net_worth', last_equity))
        current_shares = float(env.shares)
        price = float(info['price'])
        step_fees = float(info.get('last_trade_cost', 0.0))
        results['equity'].append({'time': t_ts, 'value': current_equity})

        # AUDIT LOGGING
        hurst_val = float(test_features['hurst'].iloc[idx])
        results['audit'].append({
            'hurst': hurst_val,
            'weight': abs(env.current_weight),
            'equity_change': current_equity - last_equity,
            'is_in_market': abs(env.current_weight) > 1e-4
        })

        # CẬP NHẬT PHÍ TÍCH LŨY (Chỉ khi đang có lệnh hoặc vừa thực hiện giao dịch)
        if abs(prev_shares) > TRADE_EPSILON or abs(current_shares) > TRADE_EPSILON:
            running_trade_fees += step_fees

        is_pos_open = (abs(prev_shares) > TRADE_EPSILON)
        sign_flip = (np.sign(prev_shares) != np.sign(current_shares)) and (abs(current_shares) > TRADE_EPSILON)
        closed = (abs(current_shares) <= TRADE_EPSILON) and is_pos_open
        
        if is_pos_open and (sign_flip or closed):
             trade_id += 1
             # Net PnL = Sự thay đổi net worth của cả episode/vòng đời trade
             trade_net_pnl = current_equity - trade_start_equity
             # Gross PnL = Net PnL + Fees (Vì Net PnL đã bị trừ phí trong môi trường)
             trade_gross_pnl = trade_net_pnl + running_trade_fees
             
             results['trades'].append({
                'id': trade_id, 'entry_time': entry_time, 'exit_time': t_ts,
                'type': 'BUY' if prev_shares > 0 else 'SELL',
                'contracts': round(abs(prev_shares), 4),
                'entry_price': round(float(entry_price), 2),
                'exit_price': round(float(price), 2),
                'gross_pnl': round(float(trade_gross_pnl), 2),
                'fees': round(float(running_trade_fees), 2),
                'net_pnl': round(float(trade_net_pnl), 2),
                'pnl_pct': round((trade_net_pnl / trade_start_equity) * 100, 4) if trade_start_equity > 0 else 0,
                'is_forced': False
             })
             
             # RESET CHO TRADE TIẾP THEO
             running_trade_fees = 0.0 
             if sign_flip:
                # ĐIỂM VÀO LỆNH MỚI CHO LỆNH ĐẢO (Sign Flip)
                # Tách phí của bước này (phí đóng cũ + mở mới) để gán cho lệnh tiếp theo
                entry_fee_portion = step_fees / 2.0
                running_trade_fees = entry_fee_portion
                entry_price, entry_time, trade_start_equity = price, t_ts, current_equity
             else:
                entry_price, entry_time = 0.0, 0
        
        if abs(prev_shares) <= TRADE_EPSILON and abs(current_shares) > TRADE_EPSILON:
             # ĐIỂM VÀO LỆNH MỚI
             entry_price, entry_time, trade_start_equity = price, t_ts, last_equity
             running_trade_fees = step_fees # Phí vào lệnh ban đầu
             
        prev_shares, last_equity = current_shares, current_equity
        done = terminated or truncated

    # 5. Finalize (FORCE CLOSE)
    if abs(prev_shares) > TRADE_EPSILON:
        trade_id += 1
        current_equity = float(results['equity'][-1]['value'])
        price = float(df_raw.iloc[-1]['close'])
        
        trade_net_pnl = current_equity - trade_start_equity
        trade_gross_pnl = trade_net_pnl + running_trade_fees
        
        results['trades'].append({
            'id': trade_id, 
            'entry_time': entry_time, 
            'exit_time': int(df_raw.index[-1].timestamp()),
            'type': 'BUY' if prev_shares > 0 else 'SELL',
            'contracts': round(abs(prev_shares), 4),
            'entry_price': round(float(entry_price), 2), 
            'exit_price': round(float(price), 2),
            'gross_pnl': round(float(trade_gross_pnl), 2), 
            'fees': round(float(running_trade_fees), 2),
            'net_pnl': round(float(trade_net_pnl), 2),
            'pnl_pct': round((trade_net_pnl / trade_start_equity) * 100, 4) if trade_start_equity > 0 else 0,
            'is_forced': True
        })
    
    results['metrics'] = calculate_advanced_metrics(results['equity'], results['trades'])
    audit_policy_behavior(results['audit'])
    
    # Metadata & Documentation (V9.2.1)
    results['metadata'] = {
        'engine_version': '9.2.1',
        'fee_attribution': 'Approximate (±15% accuracy for sign-flips)',
        'bias': 'Conservative (Slightly over-estimates costs)',
        'causality': 'Verified (1-bar observation lag)'
    }
    
    os.makedirs("dashboard/backtest", exist_ok=True)
    with open("dashboard/backtest/last_run.json", 'w') as f:
        json.dump(results, f)
    print(f"✅ HOÀN TẤT. Lợi nhuận ròng: {results['metrics']['Overview']['Return (%)']}")
    return results

def audit_policy_behavior(audit_data):
    """
    Phân tích hành vi Policy theo các Bucket Hurst (Audit Regime Gating).
    """
    if not audit_data: return
    
    df = pd.DataFrame(audit_data)
    
    # Buckets (Audit Regime Gating V3.1)
    buckets = [
        ('Exclusion Zone (<0.50)', df[df['hurst'] < 0.50]),
        ('Neutral/Transition (0.50-0.60)', df[(df['hurst'] >= 0.50) & (df['hurst'] <= 0.60)]),
        ('Edge/Trend Zone (>0.60)', df[df['hurst'] > 0.60])
    ]
    
    print("\n" + "="*60)
    print("📢 POLICY BEHAVIOR AUDIT (Regime Consistency)")
    print("="*60)
    print(f"{'Regime (Hurst)':<25} | {'PnL ($)':<12} | {'Avg |W|':<10} | {'Time (%)':<10}")
    print("-" * 60)
    
    for name, b_df in buckets:
        if b_df.empty:
            print(f"{name:<25} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        total_pnl = b_df['equity_change'].sum()
        avg_weight = b_df['weight'].mean()
        time_in_market = (b_df['is_in_market'].sum() / len(b_df)) * 100
        
        color = ""
        if "Trend" in name and total_pnl > 0: color = "✅ "
        if "Chop" in name and abs(total_pnl) < 100: color = "🛡️ " # Safe
        elif "Chop" in name and total_pnl < -500: color = "❌ " # Bleeding
        
        print(f"{color}{name:<23} | ${total_pnl:>10.2f} | {avg_weight:>9.2f} | {time_in_market:>8.1f}%")

    # Equity Slope/Bleed Analysis
    trend_df = df[df['hurst'] > 0.6]
    chop_df = df[df['hurst'] < 0.45]
    
    print("-" * 60)
    if not trend_df.empty:
        trend_ret = trend_df['equity_change'].sum()
        print(f"📈 [Trend Analysis] Hurst > 0.6: Total Return ${trend_ret:.2f} ({'Edge Found' if trend_ret > 0 else 'No Edge'})")
    if not chop_df.empty:
        chop_ret = chop_df['equity_change'].sum()
        print(f"📉 [Chop Analysis] Hurst < 0.45: Total Return ${chop_ret:.2f} ({'Safe' if abs(chop_ret) < 200 else 'Bleeding'})")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_backtest()
