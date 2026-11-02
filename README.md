# Nhận diện chữ viết tay A–Z

Ứng dụng Python nhận diện từng chữ cái Latin viết hoa bằng mạng nơ-ron tích chập (CNN). Người dùng vẽ chữ bằng chuột trên giao diện Tkinter, sau đó xem ký tự dự đoán và độ tin cậy của mô hình.

## Chức năng

- Vẽ chữ trên vùng canvas 300 × 300 pixel.
- Nhận diện 26 chữ cái A–Z, hiển thị kết quả và độ tin cậy.
- Xóa vùng vẽ để thử chữ mới.
- Huấn luyện, đánh giá và lưu mô hình bằng TensorFlow/Keras.

## Cấu trúc project

```text
BTLAI/
├── A_Z Handwritten Data.csv   # Dữ liệu chữ viết tay
├── preprocess.py              # Đọc, chuẩn hóa và chia dữ liệu
├── train.py                   # Xây dựng, huấn luyện và đánh giá CNN
├── gui.py                     # Giao diện vẽ và nhận diện chữ
├── models/
│   └── best_model.keras       # Mô hình dùng để dự đoán
└── README.md
```

## Cài đặt

Cần Python có Tkinter và môi trường desktop để hiển thị giao diện, chụp vùng vẽ. Mã nguồn có xử lý DPI dành cho Windows; khả năng chụp màn hình trên hệ điều hành khác phụ thuộc môi trường sử dụng.

Mở PowerShell tại thư mục project, tạo môi trường ảo và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install tensorflow numpy pandas scikit-learn opencv-python Pillow
```

Tkinter thường đi kèm bộ cài Python trên Windows. Kiểm tra bằng:

```powershell
.\.venv\Scripts\python.exe -m tkinter
```

Project chưa khóa phiên bản thư viện. Cần chọn phiên bản Python tương thích với bản TensorFlow được cài.

## Chạy ứng dụng

Đảm bảo có file `models/best_model.keras`, sau đó chạy:

```powershell
.\.venv\Scripts\python.exe gui.py
```

1. Giữ chuột trái để vẽ một chữ cái viết hoa trong vùng trắng.
2. Nhấn **Predict** để xem kết quả.
3. Nhấn **Clear** để xóa và vẽ lại.

Nên vẽ chữ rõ, đủ lớn và gần giữa vùng vẽ. Giữ canvas hiển thị đầy đủ khi dự đoán vì ứng dụng lấy ảnh bằng cách chụp màn hình.

**Lưu ý:** Giá trị `Accuracy` trên giao diện là xác suất cao nhất mô hình gán cho một lớp (độ tin cậy), không phải độ chính xác trên tập kiểm thử. Ứng dụng chỉ nhận diện từng chữ cái A–Z, chưa hỗ trợ cả từ, câu, chữ số hoặc chữ tiếng Việt có dấu.

## Huấn luyện mô hình

Đặt `A_Z Handwritten Data.csv` ở thư mục gốc, cùng cấp với `train.py`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe train.py
```

Quy trình hiện tại:

1. Đọc CSV, lấy cột tên `0` làm nhãn và các cột còn lại làm pixel.
2. Chuyển ảnh về 28 × 28 × 1, kiểu `float32`, chuẩn hóa về [0, 1].
3. Tách 10% dữ liệu làm tập kiểm thử với `random_state=42`.
4. Dùng 10% phần dữ liệu còn lại làm validation khi huấn luyện.
5. Huấn luyện tối đa 60 epoch, batch size 128; dừng sớm nếu `val_accuracy` không cải thiện trong 12 epoch.
6. Lưu checkpoint tốt nhất vào `models/best_model.keras` và in `Final test accuracy` khi đánh giá xong.

Nhãn cần tương ứng từ 0 đến 25, lần lượt là A đến Z. `preprocess.py` dùng `pd.read_csv(csv_path)` với cách đọc header mặc định. File CSV hiện có dòng đầu là dữ liệu số; cách đọc hiện tại sẽ coi dòng này là header. Nếu sử dụng CSV không có tiêu đề, cần điều chỉnh cách đọc và chọn cột để giữ đầy đủ mẫu dữ liệu.

CSV hiện có dung lượng khoảng 699 MB; quá trình đọc và chuyển ảnh có thể sử dụng nhiều RAM. Huấn luyện lại có thể ghi đè `models/best_model.keras`; hãy sao lưu nếu muốn giữ mô hình cũ.

## Kiến trúc mô hình

| Thành phần | Cấu hình |
| --- | --- |
| Đầu vào | Ảnh xám 28 × 28 × 1 |
| Tăng cường dữ liệu | Xoay, zoom và dịch chuyển ngẫu nhiên |
| Ba khối tích chập | Conv2D 3 × 3 → BatchNormalization → ReLU → MaxPooling2D; lần lượt 64, 128, 256 bộ lọc |
| Regularization | L2 = 0.0001 cho Conv2D |
| Tổng hợp đặc trưng | GlobalAveragePooling2D |
| Dropout | 0.2 |
| Đầu ra | Dense 26 lớp, softmax |
| Tối ưu | Adam, sparse categorical crossentropy |

## Xử lý lỗi thường gặp

- **Không tìm thấy mô hình:** kiểm tra `models/best_model.keras` hoặc chạy `train.py` để tạo mô hình.
- **Không tìm thấy dữ liệu:** kiểm tra tên và vị trí `A_Z Handwritten Data.csv`.
- **Thiếu thư viện:** chạy lại lệnh cài đặt bằng đúng Python trong môi trường ảo.
- **Không mở được Tkinter:** kiểm tra bộ cài Python có Tcl/Tk và máy có môi trường đồ họa.
- **Dự đoán sai hoặc không ổn định:** thử vẽ rõ hơn, căn giữa và kiểm tra vùng ảnh chụp canvas. Ảnh hiện được đảo màu và thu trực tiếp về 28 × 28, chưa cắt sát nét chữ hoặc căn giữa tự động.
