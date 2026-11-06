# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

  -----------------------------------------------------------------------------------------------------
  Mô hình        Precision   Recall       F1 F1 (SUB    Train   Tin/s (cả        p50        p95   USD /
                                      (test)     40)      (s)         lô)   (ms/tin)   (ms/tin)  1k tin
  ------------ ----------- -------- -------- ------- -------- ----------- ---------- ---------- -------
  1\. BoW +          1.000    0.781    0.877   0.919    9.033   10162.879      0.044      0.104     0.0
  LogReg (tự                                                                                    
  code)                                                                                         

  2\. BoW +          1.000    0.886    0.939   0.947   10.195   29496.788      0.306      0.786     0.0
  MLP (deep                                                                                     
  learning)                                                                                     

  3a. Encoder        0.876    0.943    0.908   0.947   76.779      26.508     50.135     73.656     0.0
  (MiniLM) +                                                                                    
  kNN                                                                                           

  3b. Encoder        0.980    0.943    0.961   0.974   77.247      36.432     26.877     39.472     0.0
  (MiniLM) +                                                                                    
  LogReg                                                                                        

  3c. Encoder        0.980    0.943    0.961   0.974   77.759      38.733     28.187     36.632     0.0
  (MiniLM) +                                                                                    
  MLP head                                                                                      

  4a. LLM nhỏ        0.950    0.950    0.950   0.950    0.000       1.006    530.429   2697.880     0.0
  zero-shot                                                                                     

  4b. LLM lớn        0.950    0.950    0.950   0.950    0.000       1.283    294.661   2334.296     0.0
  zero-shot                                                                                     

  4c. LLM nhỏ        0.950    0.950    0.950   0.950    0.000       0.549   2348.279   3561.040     0.0
  few-shot                                                                                      
  -----------------------------------------------------------------------------------------------------  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

**predict("ml_lr", "Với lr=0.001: loss giảm cực kỳ chậm và có thể không kịp hội tụ sau 300 epochs vì bước đi quá nhỏ. Với lr=50: loss sẽ dao động mạnh hoặc tăng vọt (bị phân kỳ) do bước đi quá lớn vượt qua điểm cực tiểu.")**

**Kết quả so với dự đoán**

- **lr = 0.001: Khớp.** Loss chỉ giảm từ 0.693 → 0.625 sau 300 epochs, chưa hội tụ vì bước đi quá nhỏ.
- **lr = 50: Chưa chính xác.** Loss có tăng vọt lên ~2.5 ở epoch đầu (overshoot), nhưng **không phân kỳ** mà nhanh chóng giảm về ~0.004.

**Giải thích hiện tượng ở lr = 50**

1. Gradient của cross-entropy + sigmoid có dạng `(ŷ − y)·x` với `|ŷ − y| ≤ 1`, nên bị chặn. Bước lớn không làm loss nổ như khi dùng MSE.
2. Sau bước nhảy đầu, trọng số rất lớn khiến sigmoid bão hòa, mô hình phân loại đúng với độ tin cậy cao, nên loss về gần 0.

loss thấp không có nghĩa lr = 50 tốt. Trọng số quá lớn dễ gây overfit nên đây không phải lựa chọn an toàn.

**predict("llm_fewshot", "F1 tăng nhẹ hoặc gần như không đổi (zero-shot đã 0.95); chi phí/tin tăng khoảng 4-6 lần vì prompt dài hơn nhiều")**

| Dự đoán | Thực tế (Groq, SUB 40 tin) | Khớp? |
|---|---|---|
| Model lớn F1 cao hơn nhỏ ~0.05 | 4a = 4b = **0.950** | Không |
| Lớn đắt hơn ~2.5x input / 1.5x output | USD = 0 (Groq chưa có trong `PRICE_PER_1M`) nên chưa đo được | Chưa kiểm chứng |
| LLM ngang hoặc nhỉnh hơn 3b | 3b = 0.974 > LLM 0.950 | Không |
| Few-shot: F1 tăng nhẹ hoặc không đổi | 4c = **0.950** (không đổi) | Khớp |
| Few-shot: chi phí tăng 4-6x | USD = 0, chưa đo được. Độ trễ p50 tăng 270 → 2278 ms | Chưa kiểm chứng |

**Bất ngờ nhất:** model lớn không hơn model nhỏ, và cả hai đều sai tin "Sorry I missed your call... 07090201529" (spam viết như tin cá nhân). Few-shot 4 ví dụ cũng không giúp. Lỗi còn lại của hai model khác nhau, và với SUB chỉ 40 tin, chênh 1-2 tin đã đổi F1 vài điểm nên coi như **hoà**.

**predict("dl_curve", "Val loss sẽ giảm lúc đầu nhưng sau đó sẽ tăng ngược trở lại (overfitting) khi số epoch tăng lên. Về chất lượng, MLP sử dụng Bag-of-Words sẽ có F1 tương đương hoặc chỉ nhỉnh hơn Logistic Regression một chút vì dữ liệu BoW dạng tần suất từ khá đơn giản, không tận dụng hết sức mạnh học phi tuyến của lớp ẩn.")"**

Dự đoán ban đầu phù hợp với thực nghiệm. MLP xuất hiện hiện tượng overfitting sau epoch 5, thể hiện qua train loss tiếp tục giảm trong khi validation loss tăng. Tuy nhiên, mô hình vẫn đạt F1 = 0.939, cho thấy khả năng phân loại tốt hơn Logistic Regression (F1 = 0.877).
## 4. Phân tích 50 case khó

| Model | F1 SUB | F1 50 case khó | Chênh |
|---|---|---|---|
| 1. BoW + LogReg | 0.919 | 0.286 | **-0.63** (nhiều nhất) |
| 2. BoW + MLP | 0.947 | 0.462 | -0.49 |
| 3a. Emb + kNN | 0.947 | 0.714 | -0.23 |
| 3b/3c. Emb + LogReg/MLP | 0.974 | 0.667 | -0.31 |
| 4a/4d. LLM | 0.950 | 1.000 | +0.05 |
# Phân tích các trường hợp khó đối với mô hình BoW + Logistic Regression

## 1. Mô hình tụt giảm hiệu năng nhiều nhất và nhóm case khó nhất

-   **Mô hình tụt giảm nhiều nhất:** **1. BoW + LogReg (tự code)**.
    -   **F1-score trên tập test thường (SUB):** 0.919.
    -   **F1-score trên 50 case khó:** 0.286.
    -   **Mức giảm:** khoảng 0.633 điểm F1.
-   **Nhóm case khó nhất đối với mô hình BoW:**
    -   **Tưởng spam nhưng không phải (`ham_looks_spam`)** và **nói về
        spam (`talks_about_spam`)**: độ chính xác được báo cáo là 0%.
    -   **Spam giọng tự nhiên (`natural_spam`)** và **né từ khóa
        (`obfuscation`)**: độ chính xác rất thấp, khoảng 0%--17%.

Các kết quả này cho thấy mô hình BoW + Logistic Regression gặp khó khi
cần phân biệt ý nghĩa và ngữ cảnh, hoặc khi nội dung spam cố tình biến
đổi từ ngữ để tránh các đặc trưng đã học.

## 2. Hai ví dụ cụ thể và cơ chế khiến mô hình BoW thất bại

### Ví dụ 1: Dương tính giả (False Positive) --- Ham bị dự đoán nhầm thành Spam

-   **Nội dung tin nhắn (#47):**

    > lol I just got a text saying I won £1000 and to call some 0906
    > number, people still fall for that?

-   **Nhãn thực tế:** `ham`.

-   **Nhãn dự đoán:** `spam`.

**Cơ chế lỗi:**

-   BoW biểu diễn văn bản chủ yếu bằng sự xuất hiện hoặc tần suất của
    từng từ, không trực tiếp nắm bắt đầy đủ ngữ cảnh và ý định của cả
    câu.
-   Tin nhắn chứa nhiều dấu hiệu thường gặp trong spam như `won`, `£`,
    `1000`, `call` và `number`. Những từ này có thể làm điểm dự đoán
    nghiêng mạnh về nhãn `spam`.
-   Tuy nhiên, phần còn lại của câu --- đặc biệt là `lol` và
    `people still fall for that?` --- cho thấy người gửi đang kể lại
    hoặc bình luận về một tin nhắn lừa đảo, chứ không quảng bá phần
    thưởng.
-   Mô hình dựa nhiều vào các từ khóa riêng lẻ nên không nhận ra sự khác
    biệt về ngữ cảnh này.

### Ví dụ 2: Âm tính giả (False Negative) --- Spam bị dự đoán nhầm thành Ham

-   **Nội dung tin nhắn (#29):**

    > C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at
    > w1n-gift.c0m

-   **Nhãn thực tế:** `spam`.

-   **Nhãn dự đoán:** `ham`.

**Cơ chế lỗi:**

-   Tin nhắn sử dụng kỹ thuật thay thế chữ cái bằng chữ số (Leetspeak),
    ví dụ `0` thay cho `o`, `4` thay cho `a` và `1` thay cho `i`.
-   Các từ như `C0ngr4tulat10ns`, `Y0u`, `w0n`, `g1ft` và `Cl4im` có thể
    trở thành token khác với dạng từ thông thường mà mô hình đã học.
-   Nếu những token này không có trong từ điển đặc trưng (`vocab`) được
    xây dựng từ tập train, chúng không đóng góp tín hiệu hữu ích vào
    vector đầu vào.
-   Vì vậy, mô hình có thể không nhận diện được các dấu hiệu spam quen
    thuộc và dự đoán nhầm tin nhắn thành `ham`.

## 5. Thử thách cá nhân

## 1. Cách Làm (Ý Tưởng & Phương Pháp)

* **Thiết lập bài toán:** Khởi tạo tính ngẫu nhiên với `SEED = 32` và giữ cố định số lượng 30 epochs cho mọi lượt chạy để đảm bảo so sánh công bằng.
* **Phương pháp thực nghiệm:** Lần lượt xây dựng và huấn luyện hai mô hình Multi-Layer Perceptron (MLP) chỉ khác nhau ở quy mô lớp ẩn:
  * **Mạng nhỏ:** 4 nơ-ron ẩn (tối ưu hóa quy mô tham số).
  * **Mạng lớn:** 512 nơ-ron ẩn (mở rộng dung lượng mô hình).
* **Quy trình đánh giá:** Đo đạc các thông số phần cứng/tính toán (số lượng tham số, thời gian train) và độ chính xác trên tập test (Accuracy, Precision, Recall, F1-score) để so sánh hiệu quả giữa 2 cấu hình.

---

## 2. Kết Quả Thực Nghiệm

| Tiêu chí đánh giá | Mạng Nhỏ (4 nơ-ron) | Mạng Lớn (512 nơ-ron) | Mức độ chênh lệch |
| :--- | :---: | :---: | :---: |
| **Số lượng tham số** | **8,009** | **1,025,025** | Mạng lớn gấp **~128 lần** |
| **Thời gian huấn luyện** | **3.23 giây** | **26.77 giây** | Mạng lớn chậm hơn **~8.3 lần** |
| **Accuracy (Độ chính xác)** | **98.71%** | **98.06%** | Mạng nhỏ cao hơn **0.65%** |
| **Precision (Độ chuẩn xác)** | **100.00%** | **97.87%** | Mạng nhỏ cao hơn **2.13%** |
| **Recall (Độ nhạy)** | **90.48%** | **87.62%** | Mạng nhỏ cao hơn **2.86%** |
| **F1-score (Test)** | **0.9500** | **0.9246** | Mạng nhỏ vượt trội **2.54%** |

---

## 3. Phân Tích & Giải Thích

### Tại sao Mạng Nhỏ (4 nơ-ron) lại vượt trội hơn?
1. **Hiện tượng Học thuộc lòng (Overfitting):** Mạng 512 nơ-ron sở hữu hơn 1 triệu tham số — quá dư thừa so với quy mô dữ liệu. Mô hình lớn dễ học thuộc cả nhiễu (*noise*) trong tập train, dẫn đến khả năng dự đoán trên dữ liệu thực tế (test) bị suy giảm.
2. **Sức chứa vừa đủ (Capacity Alignment):** Chỉ với 8,009 tham số, mạng 4 nơ-ron đã đủ khả năng học trọn vẹn ranh giới phân loại của bài toán mà không bị dư thừa tài nguyên.

### Khi nào nên ưu tiên chọn Mạng Nhỏ?
* **Dữ liệu nhỏ & đơn giản:** Đóng vai trò như một cơ chế nén tự nhiên, ngăn ngừa hiện tượng quá khớp (overfitting).
* **Hệ thống thời gian thực (Low Latency):** Phép tính ít (FLOPs thấp) giúp thời gian phản hồi cực nhanh.
* **Thiết bị biên (IoT / Mobile):** Tiết kiệm dung lượng RAM/VRAM và dung lượng pin.
* **Tối ưu chi phí:** Giảm thời gian và chi phí tính toán khi cần tái huấn luyện thường xuyên.

## 6. Phương án cho sản phẩm thật

p95 < 50 ms/tin. Với công suất trung bình ~116 tin/giây, cả 5 phương án học máy truyền thống và embedding đều đạt yêu cầu độ trễ (p95 từ 0.07 ms đến 31.45 ms). Hai phương án LLM (4a: p95 ~880.6 ms, 4c: p95 ~1319 ms) hoàn toàn thất bại về mặt độ trễ và không thể áp dụng trực tiếp.
   * **Đạt cả độ trễ lẫn chất lượng:** Phương án **3b. Encoder (MiniLM) + LogReg** hoặc **3c. Encoder (MiniLM) + MLP head** là tối ưu nhất. Cả hai đạt F1 test cao nhất (**0.961**), F1 SUB cực tốt (**0.974**), và tốc độ xử lý p95 cực kỳ an toàn (~8 ms đến 12.2 ms), nằm sâu dưới ngưỡng 50 ms.
   * **Ước tính chi phí/tháng nếu dùng hoàn toàn bằng LLM (gpt-6-luna):**
     * Chi phí 1 ngày (từ câu 5 phần tính tay) = $10.50 USD.
     * Chi phí 1 tháng (30 ngày) = $10.50 * 30 = **$315.00 USD** (khoảng 8.000.000 VNĐ) chưa tính chi phí mạng, hạ tầng gọi API và rủi ro rate limit.

## 7. AI log và điều đã học

## 1. Sử dụng AI (Hỏi - Nhận - Sửa)

* **Tác vụ 1: Tra cứu Cú pháp & Khởi tạo Code (Syntax)**
  * **Hỏi gì:** Đặt câu hỏi về cú pháp và cách triển khai các hàm khởi tạo.
  * **Nhận gì:** Đối với AI Agent, nhận được câu lệnh thực thi trực tiếp đúng cú pháp. Đối với Chatbot, ban đầu nhận được công thức/khung chung, sau khi bổ sung thêm dữ kiện đề bài thì nhận được câu trả lời tùy biến phù hợp.
  * **Bạn sửa gì:** Chọn lọc và tích hợp các đoạn cú pháp chuẩn vào mô hình thực tế của bài toán.

* **Tác vụ 2: Xử lý Lỗi API & Tối ưu hóa Model (Debug)**
  * **Hỏi gì:** Hỏi cách sửa lỗi khi kết nối API của mô hình ngôn ngữ và cách xử lý rate limit/latency.
  * **Nhận gì:** AI cung cấp code khởi tạo `OpenAI()`, sử dụng phương thức `create()` với các tham số tương ứng, viết logic đo latency, vòng lặp gửi request tuần tự, thêm `time.sleep()` để giãn cách và công thức ước tính token.
  * **Bạn sửa gì:** Dù AI sửa code liên tục nhưng hệ thống vẫn báo lỗi lặp đi lặp lại. Nhận thấy AI chỉ quanh quẩn sửa cú pháp mà không giải quyết được gốc rễ, tôi đã chủ động thay đổi **API Key / Provider từ Gemini sang Groq**, giúp chương trình hoạt động thành công.

---

## 2. Bài Học Rút Ra (Những điều hiểu ra hôm nay)

1. **AI mạnh về cú pháp nhưng hạn chế về tầm nhìn hệ thống:** AI xử lý rất tốt các tác vụ tạo code và logic cục bộ, nhưng dễ bị "mù" trước các vấn đề thuộc về hạ tầng, quota, hay tính sẵn sàng của nhà cung cấp dịch vụ (API Provider).
2. **Vai trò định hướng của con người là quyết định:** Khi gặp bế tắc kỹ thuật lặp đi lặp lại, AI có xu hướng xoay vòng trong các giải pháp cú pháp cũ. Lập trình viên cần giữ vai trò tư duy phản biện, bước ra khỏi gợi ý của AI để đưa ra quyết định chiến lược (như thay đổi hoàn toàn nền tảng API).
