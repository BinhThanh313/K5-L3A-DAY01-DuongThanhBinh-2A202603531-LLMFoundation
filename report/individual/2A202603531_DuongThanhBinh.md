# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.781      0.877        0.919   
2. BoW + MLP (deep learning)         1.000   0.886      0.939        0.947   
3a. Encoder (MiniLM) + kNN           0.876   0.943      0.908        0.947   
3b. Encoder (MiniLM) + LogReg        0.980   0.943      0.961        0.974   
3c. Encoder (MiniLM) + MLP head      0.980   0.943      0.961        0.974   
4a. LLM nhỏ zero-shot                0.950   0.950      0.950        0.950   
4b. LLM lớn zero-shot                0.950   0.950      0.950        0.950   
4c. LLM nhỏ few-shot                 0.950   0.950      0.950        0.950   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            9.033      10162.879       0.044   
2. BoW + MLP (deep learning)        10.195      29496.788       0.306   
3a. Encoder (MiniLM) + kNN          76.779         26.508      50.135   
3b. Encoder (MiniLM) + LogReg       77.247         36.432      26.877   
3c. Encoder (MiniLM) + MLP head     77.759         38.733      28.187   
4a. LLM nhỏ zero-shot                0.000          1.006     530.429   
4b. LLM lớn zero-shot                0.000          1.283     294.661   
4c. LLM nhỏ few-shot                 0.000          0.549    2348.279   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.104           0.0  
2. BoW + MLP (deep learning)          0.786           0.0  
3a. Encoder (MiniLM) + kNN           73.656           0.0  
3b. Encoder (MiniLM) + LogReg        39.472           0.0  
3c. Encoder (MiniLM) + MLP head      36.632           0.0  
4a. LLM nhỏ zero-shot              2697.880           0.0  
4b. LLM lớn zero-shot              2334.296           0.0  
4c. LLM nhỏ few-shot               3561.040           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
