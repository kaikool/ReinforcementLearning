import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

class MarketRegime:
    """
    Phát hiện Chế độ thị trường (Macro Regime) sử dụng Gaussian HMM.
    Bổ sung các chốt chặn an toàn:
    - Return Scaling: Tránh HMM collapse khi thay đổi scale dữ liệu.
    - Degeneracy Check: Phát hiện trạng thái chết hoặc quá lỳ.
    """
    
    def __init__(self, n_components=3, covariance_type="diag", n_iter=100, model_path="regime_model.pkl"):
        self.n_components = n_components
        self.model_path = model_path
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            n_iter=n_iter,
            random_state=42,
            init_params="stmc",
            min_covar=1e-4 # FIX: Prevent Covariance Collapse
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    # Đã loại bỏ fit_predict_rolling để tránh lỗi Label Switching và Global Scaler Leakage.
    # Sử dụng phương pháp chuẩn: Fit Offline (trên tập Train) -> Predict Online (trên tập Test/Live)


    def _reorder_states(self):
        if self.covariance_type == "diag":
            # 1. Calculate Variances for Sorting
            if self.model.covars_.ndim == 2:
                 variances = self.model.covars_.sum(axis=1)
            else:
                 # Unexpected 3D covars in diag mode? Extract diagonal sum (Trace)
                 variances = np.array([np.trace(self.model.covars_[i]) for i in range(self.n_components)])
            
            vol_order = np.argsort(variances).flatten()
            
            # Reorder States
            self.model.startprob_ = self.model.startprob_[vol_order]
            self.model.transmat_ = self.model.transmat_[vol_order, :][:, vol_order]
            self.model.means_ = self.model.means_[vol_order]
            
            # Handle Covariance Reordering & Shape Correction
            new_cov = self.model.covars_[vol_order].copy()
            if new_cov.ndim == 3:
                 print(f"⚠️ Reshaping 3D Covars {new_cov.shape} to 2D Diag...")
                 # Extract diagonal: (n, k, k) -> (n, k)
                 new_cov = np.diagonal(new_cov, axis1=1, axis2=2).copy()
            
            self.model.covars_ = new_cov
            
        else: # Full Covariance
            # Calculate Variances (Trace)
            if self.model.covars_.ndim == 3:
                variances = np.array([np.trace(self.model.covars_[i]) for i in range(self.n_components)])
            else:
                # Unexpected 2D in full mode? 
                variances = self.model.covars_.sum(axis=1)

            vol_order = np.argsort(variances).flatten()

            self.model.startprob_ = self.model.startprob_[vol_order]
            self.model.transmat_ = self.model.transmat_[vol_order, :][:, vol_order]
            self.model.means_ = self.model.means_[vol_order]
            
            self.model.covars_ = self.model.covars_[vol_order].copy()

    def fit(self, X_train):
        """
        Huấn luyện HMM OFFLINE với Multi-variate Features.
        Input: X_train shape (n_samples, n_features)
        """
        X = np.asarray(X_train)
        if X.ndim == 1:
            X = X.reshape(-1, 1) # Fallback for single feature
            
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) == 0:
            print("⚠️ CẢNH BÁO: Data rỗng. Không thể huấn luyện Regime.")
            return

        # 1. Scaling: Chuẩn hóa toàn bộ feature
        X_scaled = self.scaler.fit_transform(X)
        
        # ADD NOISE to prevent Covariance Collapse (CRITICAL FIX)
        # Add micro random noise to prevent constant 0 variance
        noise = np.random.normal(0, 1e-4, X_scaled.shape) 
        X_scaled += noise
        
        print(f"HMM Offline Fit: {X_scaled.shape} shape. Data Range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}], Mean: {X_scaled.mean():.4f}")
        
        # 2. Robust Fit Loop with Degeneracy Check
        best_score = -np.inf
        best_model = None
        
        for i in range(3): # Retry up to 3 times
            try:
                # Create FRESH model instance to avoid state pollution
                candidate_model = GaussianHMM(
                    n_components=self.n_components, 
                    covariance_type=self.covariance_type, 
                    n_iter=self.n_iter,
                    random_state=42 + i,
                    init_params="stmc",
                    min_covar=1e-4
                )
                
                candidate_model.fit(X_scaled)
                
                # Check 1: Convergence
                if not candidate_model.monitor_.converged:
                    print(f"⚠️ HMM Try {i+1}: Did not converge.")
                    continue
                    
                # Check 2: Covariance Collapse
                if self.covariance_type == "diag":
                    min_covar = np.min(candidate_model.covars_)
                else:
                    min_covar = np.min([np.min(np.diag(c)) for c in candidate_model.covars_])
                    
                if min_covar < 1e-9: # Relaxed check, trust min_covar param of HMM
                     print(f"⚠️ HMM Try {i+1}: Covariance Collapse detected (min_covar={min_covar:.10f}).")
                     continue
                
                # If passed checks, keep this model
                # FIX: Normalize score by n_features to make it comparable train vs retrain
                score = candidate_model.score(X_scaled) / X_scaled.shape[1]
                if score > best_score:
                    best_score = score
                    best_model = candidate_model
                    break
            except Exception as e:
                 print(f"⚠️ HMM Try {i+1}: Failed with error {e}")
        
        if best_model is None:
             print("❌ CRITICAL: HMM failed to fit stable model after 3 attempts. Using last model state (Risk of Instability).")
             # Use the last candidate if all failed (fallback)
             self.model = candidate_model
        else:
             self.model = best_model
             
        self.is_fitted = True
            
        # Re-order states uniformly (Semantic Sorting: State 0 = Low Vol, State N = High Vol)
        self._reorder_states()
        
        print(f"HMM Fit thành công. Sorted State Variances.")
        self.save_model(self.model_path)

    def predict_proba_smoothed_LOOKAHEAD_ONLY(self, X_input):
        """
        DANGER: Use Forward-Backward algorithm (Looks into the future).
        ONLY for post-hoc analysis/Backtesting visualization. 
        NEVER USE IN LIVE TRADING OR RL AGENT OBSERVATION.
        """
        if not self.is_fitted:
            self.load_model(self.model_path)
            
        if not self.is_fitted:
            print("⚠️ WARNING: HMM not fitted, returning uniform regime probs")
            return np.ones((len(X_input), self.n_components)) / self.n_components

        X = np.asarray(X_input)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X = np.nan_to_num(X)
        
        # Transform bằng scaler
        X_scaled = self.scaler.transform(X)
        
        # Default hmmlearn implementation is Forward-Backward (Smoothing)
        return self.model.predict_proba(X_scaled)
    
    def predict_proba_causal(self, X_input):
        """
        Inference xác suất trạng thái Causal (FILTERED - NO LOOK-AHEAD).
        Chạy vòng lặp Online Filter để đảm bảo tại thời điểm t chỉ biết t.
        """
        if not self.is_fitted:
            self.load_model(self.model_path)
            
        if not self.is_fitted:
            print("⚠️ WARNING: HMM not fitted, returning uniform (causal) regime probs")
            return np.ones((len(X_input), self.n_components)) / self.n_components

        X = np.asarray(X_input)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.nan_to_num(X)
        
        log_prob = None
        filtered_probs = []
        
        for t in range(len(X)):
            log_prob, step_prob = self.predict_online(X[t], log_prob)
            filtered_probs.append(step_prob)
            
        return np.array(filtered_probs)
    
    def get_step_regime(self, recent_returns):
        # FIX: Force strict causal filter
        probs = self.predict_proba_causal(recent_returns)
        return probs[-1]

    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'is_fitted': self.is_fitted
        }, path)
        
    def load_model(self, path):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.n_components = data['n_components']
            self.is_fitted = data['is_fitted']
            
            # Cấu hình lại min_covar để tránh lỗi "Covariance collapsed" khi load model
            if hasattr(self.model, 'min_covar'):
                self.model.min_covar = 1e-4

    def predict_online(self, X_step, prev_log_prob=None):
        """
        Dự đoán trạng thái Online (Chỉ dùng dữ liệu quá khứ & hiện tại) - Causal Filter.
        Thay thế cho predict_proba (Forward-Backward) vốn bị nhìn trước tương lai.
        
        Input: 
            X_step: (n_features,) data điểm hiện tại
            prev_log_prob: (n_components,) Log probability của bước trước
        Output:
            curr_log_prob: (n_components,) Log prob mới
            state_probs: (n_components,) Probability vector (normalized)
        """
        if not self.is_fitted:
            print("⚠️ WARNING: HMM not fitted, returning uniform online regime prob")
            return None, np.ones(self.n_components) / self.n_components
            
        X = np.asarray(X_step).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Calculate Log Emission Probs: P(X_t | Z_t)
        # Using internal GaussianPDF of hmmlearn model
        framelogprob = self.model._compute_log_likelihood(X_scaled) # shape (1, n_comp)
        
        if prev_log_prob is None:
            # Initial Step: log(StartProb) + log(Emission)
            # Clip startprob to avoid log(0)
            safe_start = np.clip(self.model.startprob_, 1e-10, 1.0)
            curr_log_unnorm = np.log(safe_start) + framelogprob.flatten()
        else:
            # Recursion: Log-Sum-Exp for Transition
            # log P(Z_t | X_1:t-1) = log sum_j ( exp(log P(Z_{t-1}=j | X_1:t-1)) * P(Z_t | Z_{t-1}=j) )
            #                      = log sum_j exp( log_prev_prob[j] + log_transmat[j, :] )
            
            # 1. Compute Log Transition Matrix safely
            safe_transmat = np.clip(self.model.transmat_, 1e-10, 1.0)
            log_transmat = np.log(safe_transmat)
            
            # 2. Compute Log Alpha (Forward) Step
            # work_buffer[i, j] = log_prev[i] + log_trans[i, j]
            # We want for each state j (next): logsumexp over i (prev)
            
            # Broadcast prev_log_prob (N,) to (N, N) where rows are i
            log_prev_tiled = np.tile(prev_log_prob.reshape(-1, 1), (1, self.n_components))
            
            # Matrix of all path log-probs: M[i, j]
            log_paths = log_prev_tiled + log_transmat
            
            # LogSumExp over axis 0 (sum over previous states i)
            log_trans_prob = np.zeros(self.n_components)
            for j in range(self.n_components):
                # Manual LogSumExp for stability: max + log(sum(exp(x-max)))
                vec = log_paths[:, j]
                max_val = np.max(vec)
                log_trans_prob[j] = max_val + np.log(np.sum(np.exp(vec - max_val)))
                
            # 3. Update with Emission: log P(Z_t | X_1:t) = log P(X_t|Z_t) + log P(Z_t|X_1:t-1)
            curr_log_unnorm = framelogprob.flatten() + log_trans_prob

        # 4. Normalize in Log Space
        # log P(Z_t) = log_unnorm - log_sum_exp(log_unnorm)
        max_log = np.max(curr_log_unnorm)
        log_norm = max_log + np.log(np.sum(np.exp(curr_log_unnorm - max_log)))
        curr_log_prob = curr_log_unnorm - log_norm
        
        # Convert to normal probability for return
        curr_prob = np.exp(curr_log_prob)
        
        return curr_log_prob, curr_prob

