

# Dự án Nhận diện gương mặt và Dự đoán tuổi sử dụng YOLOv8 và ResNet34

## Giới thiệu

Dự án này tập trung phát triển hệ thống nhận diện gương mặt và dự đoán tuổi dựa trên hai mô hình deep learning hiện đại và hiệu quả là YOLOv8 và ResNet34. Mục tiêu là xây dựng mô hình có thể phát hiện vị trí gương mặt trên ảnh/video và dự đoán chính xác tuổi của từng cá nhân trong ảnh.

---

## Bộ dữ liệu sử dụng

* **WIDER FACE**
  Bộ dữ liệu WIDER FACE là tập hợp lớn gồm hơn 32,000 ảnh với hơn 393,000 gương mặt được đánh dấu bounding box trong nhiều điều kiện phức tạp: các góc mặt khác nhau, ánh sáng, độ che phủ, biểu cảm... Đây là bộ dữ liệu tiêu chuẩn trong lĩnh vực nhận diện gương mặt, được sử dụng để huấn luyện mô hình YOLOv8 nhằm tăng khả năng phát hiện nhanh và chính xác các khuôn mặt trong môi trường thực tế.

* **UTKFace**
  Bộ dữ liệu UTKFace chứa hơn 20,000 ảnh khuôn mặt kèm nhãn tuổi từ 0 đến 116 tuổi, đa dạng về chủng tộc, giới tính và điều kiện ánh sáng. Đây là bộ dữ liệu tiêu chuẩn dùng để huấn luyện các mô hình dự đoán tuổi dựa trên hình ảnh gương mặt. Trong dự án này, UTKFace được sử dụng để huấn luyện mô hình ResNet34 với mục tiêu dự đoán tuổi của từng khuôn mặt đã được phát hiện.

---

## Phương pháp thực hiện

### Nhận diện gương mặt với YOLOv8

* Sử dụng phiên bản YOLOv8, một kiến trúc mới của dòng YOLO với ưu điểm về tốc độ và độ chính xác trong phát hiện đối tượng.
* Mô hình được huấn luyện trên bộ dữ liệu WIDER FACE, giúp nhận diện gương mặt hiệu quả trong nhiều điều kiện thực tế phức tạp.
* Đầu ra của mô hình là các bounding box xác định vị trí chính xác của từng gương mặt trong ảnh hoặc video.

### Dự đoán tuổi với ResNet34

* Mỗi khuôn mặt được phát hiện sẽ được cắt ra và đưa vào mô hình ResNet34.
* ResNet34 là một kiến trúc mạng CNN sâu, được thiết kế để xử lý tốt các đặc trưng hình ảnh phức tạp.
* Mô hình ResNet34 được huấn luyện trên bộ dữ liệu UTKFace với nhãn tuổi tương ứng, nhằm dự đoán tuổi của đối tượng trong ảnh.
* Quá trình huấn luyện bao gồm các kỹ thuật như điều chỉnh learning rate, data augmentation để tăng độ chính xác và khả năng khái quát hóa của mô hình.

---

## Quá trình thực hiện

* Tiền xử lý dữ liệu, bao gồm chuẩn hóa, chia ảnh khuôn mặt theo nhãn tuổi, và chuẩn bị bộ dữ liệu phù hợp cho huấn luyện.
* Huấn luyện YOLOv8 trên bộ WIDER FACE để mô hình có thể phát hiện chính xác gương mặt trong nhiều tình huống.
* Huấn luyện ResNet34 trên bộ UTKFace để dự đoán tuổi với độ chính xác cao.
* Tích hợp hai mô hình để xử lý đầu vào ảnh/video: đầu tiên phát hiện gương mặt, sau đó dự đoán tuổi cho từng gương mặt.
* Đánh giá hiệu năng bằng các chỉ số như độ chính xác (accuracy), mean absolute error (MAE) cho dự đoán tuổi, và tốc độ nhận diện trên ảnh/video.

---

## Kết quả và ứng dụng

* Hệ thống nhận diện gương mặt và dự đoán tuổi đạt được độ chính xác và hiệu suất tốt, phù hợp với các ứng dụng trong thực tế.
* Có thể sử dụng trong các lĩnh vực như an ninh, kiểm soát truy cập, phân tích đối tượng khách hàng theo độ tuổi, hoặc nghiên cứu thị trường.
* Hỗ trợ xử lý video trực tiếp hoặc xử lý ảnh tĩnh với tốc độ nhanh và độ chính xác cao.

---

## Cấu trúc dự án

* `datasets/` : Chứa dữ liệu và nhãn của bộ WIDER FACE và UTKFace (không được push lên Git do dung lượng lớn).
* `resnet_model.py` : Code định nghĩa và huấn luyện mô hình ResNet34 cho dự đoán tuổi.
* `train_gpu.py`, `train_subset.py`, ... : Các script huấn luyện mô hình.
* `run.py` : Script chính để chạy nhận diện gương mặt và dự đoán tuổi trên ảnh hoặc video.
* `yolov8n.pt`, `yolov8s.pt` : Mô hình YOLOv8 được huấn luyện sẵn (không push lên Git do dung lượng lớn).
* `README.md` : File mô tả dự án.

---

