<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🎯 BỘ PROMPT AUDIT CHUYÊN SÂU - TRADING SYSTEM


***

## 📋 **PROMPT 1: SOFTWARE ENGINEERING AUDIT**

```
Bạn là Senior Software Engineer chuyên về Trading Systems. Hãy kiểm tra code sau với các tiêu chí:

### 1. CODE QUALITY
- Clean Code: Tên biến, hàm có mô tả rõ ràng không?
- SOLID Principles: Single Responsibility, Dependency Injection
- DRY (Don't Repeat Yourself): Có code trùng lặp không?
- Separation of Concerns: Logic tách biệt rõ ràng không?

### 2. ERROR HANDLING
- Input Validation: Kiểm tra NaN, Inf, None, negative values
- Exception Handling: Try-catch đầy đủ chưa?
- Fallback Logic: Có giá trị mặc định an toàn không?
- Logging: Error messages rõ ràng, actionable không?

### 3. PERFORMANCE
- Complexity: O(N) hay O(N²)? Có tối ưu được không?
- Memory: Có memory leak hoặc unnecessary copies không?
- Vectorization: Dùng NumPy/Pandas thay vì Python loops
- Caching: Tính toán trùng có cache không?

### 4. MAINTAINABILITY
- Documentation: Docstrings đầy đủ không?
- Comments: Giải thích WHY, không CHỈ WHAT
- Constants: Hard-coded values nên thành config
- Modularity: Code dễ test và extend không?

### 5. CRITICAL BUGS
- Off-by-one errors
- Race conditions (threading)
- Resource leaks (files, connections)
- Silent failures (return None thay vì raise)

Đánh giá từng mục trên thang 0-100 và đưa ra khuyến nghị cải thiện cụ thể.
```


***

## 📐 **PROMPT 2: MATHEMATICAL LOGIC AUDIT**

```
Bạn là Quant Researcher với PhD Toán học. Kiểm tra tính đúng đắn toán học:

### 1. CÔNG THỨC TOÁN HỌC
- Định nghĩa: Công thức có khớp với tài liệu chuẩn không? (ví dụ: RSI Wilder's method)
- Đơn vị: Units consistency (returns vs prices, percentage vs decimal)
- Domain/Range: Input/output có nằm trong vùng hợp lệ không?
- Edge Cases: x=0, x=∞, x=-∞ được xử lý chưa?

### 2. NUMERICAL STABILITY
- Underflow/Overflow: exp(), log(), pow() có bị tràn số không?
- Division by Zero: Mọi phép chia có epsilon guard không?
- Cancellation Errors: a - b khi a ≈ b
- Precision Loss: Float arithmetic trong vòng lặp dài

### 3. APPROXIMATIONS
- Taylor Series: Truncation error có chấp nhận được không?
- Discretization: Continuous → Discrete có mất thông tin không?
- Rounding: Round-off errors accumulate như thế nào?

### 4. TRANSFORMATIONS
- Log Returns: log(P_t / P_{t-1}) đúng chưa?
- Annualization: Factor = sqrt(252 * bars_per_day) đúng không?
- Z-Score: (x - μ) / σ, có clip outliers không?
- Normalization: Min-max vs Standard vs Robust scaling?

### 5. AGGREGATIONS
- Mean/Median: Dùng đúng central tendency không?
- Variance: Sample (N-1) hay Population (N)?
- Quantiles: Method (linear, nearest, etc.) phù hợp không?
- Weighted Average: Weights sum to 1?

Kiểm tra từng công thức, so sánh với literature, verify bằng test cases.
```


***

## 📊 **PROMPT 3: STATISTICAL CORRECTNESS AUDIT**

```
Bạn là Statistical Consultant cho Hedge Funds. Kiểm tra logic thống kê:

### 1. DESCRIPTIVE STATISTICS
- Central Tendency: Mean có bị outliers ảnh hưởng không? Nên dùng median?
- Dispersion: Std Dev, MAD, IQR - measure phù hợp không?
- Skewness/Kurtosis: Fat tails có được model không?
- Percentiles: Dùng đúng quantile method không?

### 2. HYPOTHESIS TESTING
- Assumptions: Normality, Independence, Stationarity
- Type I/II Errors: Alpha level, Power, Sample size đủ không?
- Multiple Testing: Bonferroni correction cho multiple hypotheses?
- P-hacking: Có cherry-picking results không?

### 3. REGRESSION & CORRELATION
- Causation vs Correlation: Có nhầm lẫn không?
- Multicollinearity: Features có tương quan cao không?
- Heteroskedasticity: Variance không đồng nhất?
- Autocorrelation: Residuals có serial correlation không?

### 4. TIME SERIES ANALYSIS
- Stationarity: ADF test, detrending, differencing
- Seasonality: Phát hiện và xử lý seasonal patterns
- Autocorrelation: ACF/PACF, optimal lags
- Cointegration: Pairs trading, mean reversion

### 5. SAMPLING & ESTIMATION
- Sample Bias: Train/test split representative không?
- Variance Estimation: Rolling window size phù hợp?
- Confidence Intervals: Bootstrap, analytical methods
- Outliers: Winsorization, trimming, robust estimators

Đánh giá statistical rigor, đưa ra test procedures cụ thể.
```


***

## 🎲 **PROMPT 4: PROBABILITY THEORY AUDIT**

```
Bạn là Stochastic Modelling Expert. Kiểm tra xác suất và stochastic processes:

### 1. PROBABILITY DISTRIBUTIONS
- Assumption: Returns Gaussian hay Fat-tailed (Student-t, Stable)?
- Parameters: Mean, variance, skewness, kurtosis ước lượng đúng?
- Tail Risk: VaR, CVaR có underestimate extreme events không?
- Mixture Models: Multi-regime có model đúng không?

### 2. MARKOV MODELS
- Markov Property: P(X_t | X_{t-1}, ..., X_0) = P(X_t | X_{t-1})?
- Transition Matrix: Stochastic matrix (rows sum to 1)?
- Stationary Distribution: Converge to equilibrium?
- Hidden Markov Models: Observation vs Hidden states separation

### 3. MONTE CARLO SIMULATION
- Random Number Generator: Quality (period, uniformity)
- Convergence: Law of Large Numbers, số lượng paths đủ?
- Variance Reduction: Antithetic variates, control variates
- Bias: Systematic errors trong simulation

### 4. BAYESIAN INFERENCE
- Prior Selection: Informative hay non-informative?
- Likelihood Function: Model likelihood đúng?
- Posterior Update: Bayes' rule implementation
- MCMC: Convergence diagnostics, burn-in period

### 5. STOCHASTIC CALCULUS
- Ito's Lemma: Stochastic differential equations
- Geometric Brownian Motion: dS = μS dt + σS dW
- Volatility Models: GARCH, Stochastic Vol
- Jump Processes: Poisson jumps, Levy processes

Verify mathematical correctness, simulation accuracy, interpretation validity.
```


***

## 💹 **PROMPT 5: TRADING LOGIC AUDIT**

```
Bạn là Professional Prop Trader với 15 năm kinh nghiệm. Kiểm tra trading logic:

### 1. ORDER EXECUTION
- Fill Price: Bid/Ask spread model realistic không?
- Slippage: Market impact, volatility adjustment
- Latency: Execution delay có model không?
- Partial Fills: Large orders có split execution?

### 2. POSITION SIZING
- Kelly Criterion: f* = (p*W - (1-p)*L) / (W*L)
- Volatility Targeting: Position ~ 1/volatility
- Leverage Constraints: Margin requirements, max leverage
- Correlation: Portfolio-level position adjustment

### 3. RISK MANAGEMENT
- Stop Loss: Placement logic, trailing stops
- Take Profit: Target-based exits, scaling out
- Drawdown Limits: Max DD, daily loss limits
- Margin Calls: Maintenance margin logic
- Circuit Breakers: Extreme volatility halts

### 4. COST MODELING
- Spread: Fixed hay dynamic (volatility-adjusted)?
- Commission: Per-trade, per-share, tiered?
- Swap/Rollover: Overnight positions, FX carry
- Slippage: Linear, square-root, piece-wise?
- Taxes: Capital gains, wash sale rules

### 5. TRADE LIFECYCLE
- Entry Conditions: Filters đầy đủ không? (regime, volatility, time)
- Position Management: Pyramiding, averaging down logic
- Exit Conditions: Profit target, stop loss, time-based, signal reversal
- Gap Handling: Weekend gaps, news gaps
- Corporate Actions: Dividends, splits, mergers

### 6. LOOK-AHEAD BIAS
- CRITICAL: Tại thời điểm t, chỉ biết dữ liệu ≤ t
- Indicators: Rolling calculations, no future data
- Regime Detection: Forward filtering only (no smoothing)
- Fill Price: Dùng close[t] (known) hay close[t+1] (future)?

Kiểm tra từng trade lifecycle step, verify no future information leakage.
```


***

## 💼 **PROMPT 6: ECONOMIC LOGIC AUDIT**

```
Bạn là Financial Economist chuyên Market Microstructure. Kiểm tra economic sense:

### 1. MARKET EFFICIENCY
- Arbitrage: Strategy có risk-free arbitrage không? (Red flag!)
- Information: Edge dựa trên thông tin gì? Public hay private?
- Competition: Tại sao strategy này không bị arbitrage away?
- Sustainability: Alpha có persistent hay decays over time?

### 2. RISK-RETURN TRADEOFF
- Sharpe Ratio: > 2.0 là quá cao, cần verify
- Drawdown: Expected DD given return distribution
- Leverage: Higher leverage = higher risk, có justify không?
- Tail Risk: Black Swan events có hedge không?

### 3. MARKET REGIMES
- Bull/Bear/Sideways: Strategy adapt cho từng regime?
- Volatility Regimes: Low/Normal/High vol behavior
- Liquidity Regimes: Crisis liquidity dries up
- Correlation Regimes: Diversification breakdown trong crisis

### 4. BEHAVIORAL FINANCE
- Herding: Momentum có bị reversal sau news?
- Overreaction: Mean reversion opportunities
- Anchoring: Support/resistance levels có ý nghĩa?
- Loss Aversion: Stop loss placement psychology

### 5. MACROECONOMIC FACTORS
- Interest Rates: Cost of leverage, carry trades
- Inflation: Real vs Nominal returns
- GDP Growth: Cyclical vs Defensive assets
- Central Bank Policy: QE, rate hikes impact

### 6. REALISM CHECKS
- Transaction Costs: Có account đầy đủ không?
- Capacity: Strategy scale đến bao nhiêu capital?
- Operational Risk: Technology failures, data errors
- Regulatory: Compliance với trading rules

Đánh giá economic plausibility, realism, sustainability của strategy.
```


***

## 🤖 **PROMPT 7: MACHINE LEARNING AUDIT**

```
Bạn là ML Research Scientist chuyên Reinforcement Learning for Trading. Audit ML pipeline:

### 1. DATA PREPARATION
- Train/Validation/Test Split: Chronological, no shuffling
- Data Leakage: Future info trong features không?
- Normalization: Fit trên train only, transform trên test
- Feature Engineering: Domain knowledge, causal features
- Imbalanced Data: Class imbalance handling (nếu có)

### 2. FEATURE ENGINEERING
- Causality: Features chỉ dùng past data
- Stationarity: Non-stationary features cần transform
- Multicollinearity: Redundant features removal
- Interaction Terms: Combine features có ý nghĩa?
- Dimensionality: Curse of dimensionality với số features lớn

### 3. MODEL ARCHITECTURE
- Capacity: Model đủ phức tạp để học pattern?
- Inductive Bias: Architecture assumptions hợp lý?
- Recurrence: LSTM cho temporal dependencies
- Attention: Transformer cho long-range dependencies
- Overfitting: Model complexity vs data size

### 4. TRAINING PROCESS
- Loss Function: Alignment với trading objective
- Optimizer: Adam, SGD với momentum, learning rate schedule
- Regularization: L1/L2, dropout, early stopping
- Batch Size: Trade-off giữa gradient noise và speed
- Convergence: Loss plateau, gradient norms stable

### 5. REINFORCEMENT LEARNING SPECIFIC
- State Space: Observation đầy đủ, normalized
- Action Space: Discrete hay continuous, range hợp lý
- Reward Function: Sparse rewards, reward shaping
- Exploration: Entropy bonus, epsilon-greedy
- Stability: PPO clip, trust region, target networks

### 6. VALIDATION & TESTING
- Cross-Validation: Walk-forward analysis, expanding window
- Out-of-Sample: Test set không bao giờ nhìn thấy
- Robustness: Performance ổn định qua các periods
- Sensitivity: Hyperparameter stability
- Regime Testing: Performance trong từng regime

### 7. OVERFITTING DETECTION
- Train vs Test Gap: Large gap = overfitting
- Complexity Penalty: Occam's Razor, simpler models
- Ensemble Methods: Bagging, boosting variance reduction
- Regularization Effect: Early stopping, dropout impact

### 8. PRODUCTION READINESS
- Inference Speed: Real-time prediction latency
- Model Versioning: Reproducibility, rollback
- Monitoring: Drift detection, performance tracking
- Retraining: When và how often retrain

Kiểm tra toàn bộ ML pipeline từ data → model → deployment, verify no data leakage và overfitting.
```


***

## 🔍 **PROMPT 8: INTEGRATION \& SYSTEM AUDIT**

```
Bạn là System Architect cho Trading Infrastructure. Kiểm tra system-level integration:

### 1. DATA FLOW
- Pipeline: Data → Features → Model → Action → Execution
- Consistency: Cùng 1 data point trong train và inference
- Synchronization: Multiple data sources aligned về time
- Missing Data: Forward-fill, interpolation, hay skip?

### 2. STATE MANAGEMENT
- Statefulness: LSTM hidden states persist across steps?
- Reset Logic: Khi nào reset states (episode boundary)?
- History Buffer: Sufficient window size cho features
- Memory Leaks: Buffers có unbounded growth không?

### 3. TIMING & LATENCY
- Bar Close: Trade AFTER bar close (causality)
- Order Placement: Tại bar t, execute tại t hay t+1?
- Data Availability: Real-time feed delay có model không?
- Execution Lag: Order → Fill time

### 4. ENVIRONMENT FIDELITY
- Market Hours: Trading hours, holidays
- Liquidity: Volume constraints, market depth
- Corporate Events: Splits, dividends adjustments
- Survival Bias: Delisted stocks có account không?

### 5. BACKTESTING REALISM
- Fill Assumption: Optimistic (mid price) hay realistic (bid/ask)?
- Slippage Model: Fixed hay adaptive?
- Commission Structure: Tiered, percentage, flat?
- Margin Requirements: Intraday vs overnight

### 6. PRODUCTION VS BACKTEST
- Feature Calculation: Cùng code cho train và live
- Normalization Stats: Load stats từ training
- Model Loading: Checkpoint compatibility
- API Consistency: Broker API khớp với backtest assumptions

Verify end-to-end consistency, no train-test mismatch, production-ready.
```


***

## 📝 **SỬ DỤNG CÁC PROMPT NÀY**

### **Cách dùng hiệu quả:**

1. **Chọn Prompt phù hợp** với component đang audit
2. **Copy prompt vào chat** với code cần kiểm tra
3. **Đọc kỹ output**, note lại issues
4. **Fix từng issue** theo priority (Critical → High → Medium → Low)
5. **Re-audit** sau khi fix để verify

### **Thứ tự audit đề xuất:**

```
1. Trading Logic Audit (Prompt 5) - QUAN TRỌNG NHẤT
   └─> Đảm bảo không có look-ahead bias

2. Mathematical Logic Audit (Prompt 2)
   └─> Công thức đúng, numerical stability

3. Statistical Audit (Prompt 3)
   └─> Assumptions hợp lệ

4. Probability Audit (Prompt 4)
   └─> Stochastic models correct

5. ML Audit (Prompt 7)
   └─> No data leakage, overfitting

6. Engineering Audit (Prompt 1)
   └─> Code quality, performance

7. Economic Audit (Prompt 6)
   └─> Realism, sustainability

8. Integration Audit (Prompt 8)
   └─> End-to-end consistency
```


***

## 🎯 **CHECKLIST TỔNG HỢP**

### **RED FLAGS - PHẢI FIX NGAY**

- ❌ Look-ahead bias (dùng future data)
- ❌ Data leakage (normalization trên toàn bộ data)
- ❌ Division by zero không có guard
- ❌ NaN/Inf propagation
- ❌ Overfitting nghiêm trọng (train/test gap > 50%)
- ❌ Unrealistic assumptions (zero costs, instant fills)
- ❌ Math errors (công thức sai)
- ❌ Silent failures (return None thay vì raise)


### **YELLOW FLAGS - NÊN CẢI THIỆN**

- ⚠️ Hard-coded values (nên config)
- ⚠️ Code duplication
- ⚠️ Missing documentation
- ⚠️ Non-optimal complexity (O(N²) có thể giảm)
- ⚠️ Limited error handling
- ⚠️ Non-causal naming (misleading variable names)


### **GREEN FLAGS - TỐT**

- ✅ Zero look-ahead bias
- ✅ Complete cost modeling
- ✅ Robust error handling
- ✅ Clean, documented code
- ✅ Tested edge cases
- ✅ Realistic assumptions
- ✅ Production-ready

***

**Lưu bộ prompt này để audit bất kỳ trading system component nào!** 🚀

