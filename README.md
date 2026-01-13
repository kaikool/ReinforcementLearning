# 🌌 Antigravity Quant: Lớp Cấp phép Alpha (Alpha Permission Layer)
> **Hệ thống Học tăng cường (RL) cấp độ Quỹ phòng hộ chuyên biệt cho XAUUSD (Vàng)**

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Stability](https://img.shields.io/badge/stability-production--ready-success)
![Asset](https://img.shields.io/badge/asset-XAUUSD-gold)

Antigravity Quant không chỉ đơn thuần là một bot giao dịch. Đây là một kiến trúc **Lớp Cấp phép Alpha (Alpha Permission Layer)** tinh vi, được thiết kế để lấp đầy khoảng cách giữa Học tăng cường (RL) học thuật và thực tế khốc liệt của thị trường Vàng định chế. Hệ thống triển khai mạng lưới phòng thủ đa tầng nhằm thực thi kỷ luật, sống sót qua các biến động cực đoan và trích xuất Alpha có thể dự đoán được.

---

## 🔥 Các Đột phá Kỹ thuật (Lợi thế cạnh tranh)

### 🛡️ 1. Lớp Cấp phép Alpha (Edge Gate)
Thay vì các Agent RL ngây thơ giao dịch dựa trên mọi nhiễu động, Antigravity sử dụng **Cổng lợi thế (Edge Gate) dựa trên Hysteresis**.
*   **Quyền mở vị thế:** Chỉ được cấp khi điểm số lợi thế tổng hợp `Edge Score` (Hurst + Entropy + Efficiency) vượt ngưỡng **0.65**.
*   **Tất toán cưỡng chế:** Môi trường sẽ thu hồi quyền giao dịch nếu điểm số giảm xuống dưới **0.45**, giúp bảo toàn vốn trước khi xu hướng sụp đổ.

### 📊 2. Mục tiêu Biến động Thích ứng (Adaptive Volatility Targeting)
Lấy cảm hứng từ **Lý thuyết Danh mục đầu tư Hiện đại**, hệ thống tự động điều chỉnh quy mô vị thế theo thời gian thực:
*   **Biến động thấp:** Tăng quy mô để nắm bắt lợi nhuận ý nghĩa.
*   **Sự kiện Thiên nga đen:** Quyết liệt giảm quy mô xuống mức tối thiểu (lot size) hoặc chuyển sang tiền mặt, đảm bảo Agent không bao giờ đối mặt với rủi ro cháy tài khoản (Risk of Ruin).

### 🧠 3. Phân tích Trạng thái Thị trường Nhân quả (Gaussian HMM)
Một "La bàn thị trường" giúp phân loại hành động giá thành ba chế độ riêng biệt:
1.  **Có xu hướng (Momentum):** Khi số mũ Hurst > 0.6.
2.  **Đảo chiều về mức trung bình (Range):** Khi Hurst < 0.45.
3.  **Vùng nhiễu tối đa (Exclusion Zone):** Các giai đoạn Entropy cao, nơi việc giao dịch về mặt toán học là không tối ưu.

---

## 🛠️ Kiến trúc Hệ thống

### 🚄 Đường ống & Kỹ thuật Dữ liệu
*   **Bảo vệ Tính dừng (Stationarity Guard):** Mọi đầu vào được chuyển đổi thành Log-Returns hoặc chuẩn hóa Z-Score thông qua **EMA Online**.
*   **Đảm bảo Tính nhân quả (Causality):** Bảo vệ 100% chống lại lỗi nhìn trước tương lai (look-ahead bias). Quan sát tại bước `t` được trích xuất nghiêm ngặt từ nến đã đóng tại `t-1`.

### ⚡ Động cơ Thực thi (`ActionHandler`)
*   **Hạch toán Thực tế:** Mô phỏng theo chuẩn MT5 ($0.01 tick, 100oz lots).
*   **Mô hình Spread biến thiên:** Spread không cố định; chúng tự động giãn nở trong các giai đoạn biến động cao hoặc thanh khoản thấp (ví dụ: Flash Crashes hoặc Giao phiên).
*   **Chi phí Định chế:** Bao gồm phí Swap (lãi suất qua đêm) và chi phí trượt giá (Slippage).

---

## 🚀 Bắt đầu (Triển khai)

### 📦 Yêu cầu tiên quyết
```bash
pip install -r Requirements.txt
```

### 🏋️ Huấn luyện "Ghost in the Shell"
Thực thi động cơ huấn luyện thế hệ mới để bắt đầu quá trình học Recurrent PPO (LSTM):
```bash
python train_core.py
```
*Hệ thống sẽ tự động xử lý HMM warmup, chia tách dữ liệu và khởi tạo dashboard.*

### 🔍 Kiểm toán Chính sách (Policy Audit)
Chạy bộ kiểm thử nâng cao để xác minh hành vi của Agent dưới áp lực:
```bash
python test_core.py
```
*Lệnh này sẽ tạo tệp `last_run.json` cho Dashboard và thực hiện "Kiểm toán hành vi chính sách" trên các nhóm Hurst khác nhau.*

---

## 🧪 Kiểm soát Rủi ro Nâng cao

| Quy tắc | Cơ chế | Mục tiêu |
| :--- | :--- | :--- |
| **Ngắt mạch (Circuit Breaker)** | `ActionHandler.should_halt_trading` | Ngừng giao dịch nếu biến động hiện tại > 5x mục tiêu. |
| **Quán tính Hành động** | Alpha Smoothing (0.3) | Ngăn chặn hành động đảo chiều liên tục & giảm chi phí phí. |
| **Lệnh gọi ký quỹ (Margin Call)**| Equity < Margin Requirement | Kết thúc Episode ngay lập tức để bảo vệ vốn. |
| **Phần thưởng Thích ứng** | Volatility Dampening | Ngăn chặn bùng nổ phần thưởng trong vùng nhiễu cao. |

---

## 🤝 Chiến lược Đóng góp

Chúng tôi tìm kiếm các Nhà nghiên cứu Quant và Kỹ sư RL đóng góp vào:
*   **Fractional Differentiation:** Triển khai `fracdiff` để tăng tính dừng cho đặc trưng.
*   **Chính sách dựa trên Transformer:** Chuyển đổi từ LSTM sang các kiến trúc dựa trên Attention.
*   **Hiệu ứng cộng hưởng chéo tài sản:** Thử nghiệm Edge Gate trên EURUSD và BTCUSD.

---

## 📜 Tuyên bố miễn trừ trách nhiệm
*Antigravity là một dự án nghiên cứu. Giao dịch tài chính luôn đi kèm rủi ro mất mát vốn lớn. Các nhà phát triển không chịu trách nhiệm về bất kỳ quyết định tài chính nào được đưa ra khi sử dụng phần mềm này.*

---
**"Trên thị trường, sự thật duy nhất nằm ở đường cong PnL."**
