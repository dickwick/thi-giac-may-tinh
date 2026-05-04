Animal Recognition AI (Vision Transformer)
Ứng dụng web sử dụng mô hình Vision Transformer (ViT) để nhận diện 100 loài động vật từ hình ảnh.
Hệ thống được xây dựng bằng PyTorch + HuggingFace Transformers và triển khai giao diện bằng Streamlit.


Cách sử dụng hướng dẫn sử dụng

### 1. Cài đặt thư viện
Cài đặt các thư viện cần thiết trước khi chạy:
pip install torch torchvision streamlit pillow transformers

Các thành phần chính:
torch, torchvision: xây dựng và chạy model AI
transformers: sử dụng mô hình ViT
streamlit: tạo web interface
pillow: xử lý ảnh đầu vào

### 2. Chuẩn bị Model
tải model về đặt vào file

📁 Cấu trúc project:
│── appVit.py
│── classes.json
│── style.css
│── model/
│     └── vit_animals_best.pth

Lưu ý:
Nếu đặt sai vị trí → chương trình sẽ lỗi khi load model

### 3. Chạy ứng dụng
Trong thư mục project, chạy lệnh:
streamlit run appVit.py

## 🧠 Cách hoạt động

Pipeline của hệ thống:

1. Người dùng upload ảnh động vật
2. Ảnh được đưa vào mô hình Vision Transformer (ViT)
3. Model thực hiện phân loại (classification)
4. Kết quả trả về:
   * 🐾 Tên động vật
   * 📊 Độ tin cậy (%)



### Huấn luyện lại model (nếu cần thiết) ###

### Yêu cầu:
tải về dataset về

### Chạy training:
python train_vit.py

Sau khi hoàn thành:
Model tốt nhất sẽ được lưu thành file `.pth`
Sử dụng lại file này cho ứng dụng web