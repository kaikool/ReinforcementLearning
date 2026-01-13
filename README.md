# Antigravity Quant: Hedge-Fund Grade RL Trading System (XAUUSD)

Hệ thống giao dịch tự động sử dụng Học tăng cường (Reinforcement Learning - RL) chuyên biệt cho thị trường Vàng (XAUUSD). Dự án tập trung vào việc áp dụng các kỹ thuật tài chính định lượng chuyên sâu (Quant) và quản trị rủi ro cấp độ quỹ phòng hộ.

---

## 🛠️ Hướng dẫn Cài đặt (Installation)

### 1. Yêu cầu hệ thống
* Python 3.10 trở lên.
* Cài đặt các thư viện cần thiết:
```bash
pip install -r Requirements.txt
```

### 2. Chuẩn bị dữ liệu
* Đặt các tệp dữ liệu huấn luyện vào thư mục `data/` với định dạng tên `train_*.csv`.
* Cấu trúc tệp CSV yêu cầu các cột: `Time (EET)` hoặc `Gmt time`, `Open`, `High`, `Low`, `Close`, `Volume`.

### 3. Huấn luyện (Training)
Chạy lệnh sau để bắt đầu quá trình huấn luyện:
```bash
python train_core.py
```
*Mô hình và các chỉ số sẽ được lưu trong thư mục `artifacts/versionX`.*

### 4. Kiểm thử (Backtesting)
Chạy lệnh sau để đánh giá mô hình trên dữ liệu kiểm thử:
```bash
python test_core.py
```

---

## 🏗️ Kiến trúc Hệ thống (System Architecture)

### 1. `AdvancedTradingEnv` (Environment)
*   **Alpha Permission Layer (Edge Gate):** Cơ chế lọc tín hiệu dựa trên Hysteresis. Ngăn chặn giao dịch trong vùng nhiễu.
*   **Volatility Targeting:** Tự động điều chỉnh quy mô vị thế dựa trên biến động thị trường.
*   **Causality Guard:** Loại bỏ hoàn toàn Look-ahead bias, Agent chỉ nhìn thấy nến đã đóng.

### 2. `QuantFeatureFactory` (Feature Engineering)
*   **Online Z-Score:** Chuẩn hóa dữ liệu động qua EMA, đảm bảo tính liên tục khi Online.
*   **Advanced Indicators:** Hurst Exponent (Độ bền xu hướng), Shannon Entropy (Độ nhiễu), Trend Efficiency.

### 3. `MarketRegime` (HMM Analysis)
*   **Gaussian HMM:** Phân loại 3 trạng thái thị trường: Trend, Mean Reversion, và Noise.
*   **Causal Online Filter:** Dự báo trạng thái real-time mà không nhìn trước tương lai.

### 4. `CompoundReward` (Reward Shaping)
*   **Log-Return Based:** Tối ưu hóa lợi nhuận kỳ vọng theo Log-scale.
*   **Adaptive Vol Scaling:** Co giãn phần thưởng theo mức độ rủi ro của thị trường.

---

## 🛡️ Quản trị Rủi ro (Risk Management)

*   **Margin Call Termination:** Tự đóng Episode nếu chạm ngưỡng rủi ro vốn.
*   **Circuit Breaker:** Ngừng giao dịch khi biến động thị trường vượt ngưỡng cực đoan.
*   **Action Inertia:** Cơ chế làm mượt hành động giúp giảm thiểu Over-trading và phí giao dịch.

---

## 🤝 Đóng góp (Contribution)

Cộng đồng có thể đóng góp ý kiến về các phần:
*   **Feature Library:** Thêm các chỉ báo định lượng mới.
*   **Reward Function:** Tối ưu hóa hàm phạt để giảm Drawdown.
*   **Stress Testing:** Thêm các kịch bản thị trường khắc nghiệt.

---
**Lưu ý:** Dự án này dành cho mục đích nghiên cứu. Giao dịch tài chính luôn đi kèm rủi ro mất mát vốn. Khuyến cáo kiểm thử kỹ lưỡng trên tài khoản Demo.
