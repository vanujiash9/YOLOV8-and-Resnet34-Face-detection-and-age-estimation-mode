from ultralytics import YOLO

# Tải mô hình YOLOv8
model = YOLO('yolov8n.pt')

# Huấn luyện với tập dữ liệu nhỏ
results = model.train(
    data='D:/YOLO/datasets/wider_face/data_subset.yaml',
    epochs=10,
    imgsz=160,
    batch=1,
    workers=0,
    cache=False,
    device='cpu',
    patience=5,
    project='D:/YOLO/runs',
    name='wider_face_yolov8n_ultralight'
)

import os
import glob
from PIL import Image
import numpy as np
import shutil

def verify_and_fix_dataset():
    # Đường dẫn WIDER Face
    wider_train_labels = "D:/YOLO/datasets/wider_face/labels/train/wider_face_train_bbx_gt.txt"
    train_img_dir = "D:/YOLO/datasets/wider_face/images/WIDER/WIDER_train/images"
    yolo_label_dir = "D:/YOLO/datasets/wider_face/labels_yolo/train"
    
    os.makedirs(yolo_label_dir, exist_ok=True)
    
    # Xóa dữ liệu nhãn cũ nếu có
    for file in glob.glob(os.path.join(yolo_label_dir, "*.txt")):
        os.remove(file)
    
    # Đọc file nhãn gốc
    with open(wider_train_labels, "r") as f:
        lines = f.readlines()
    
    # Biến theo dõi
    success_count = 0
    error_count = 0
    i = 0
    
    # Quá trình chuyển đổi
    while i < len(lines):
        try:
            img_path = lines[i].strip()
            if not img_path.endswith('.jpg'):
                i += 1
                continue
                
            full_img_path = os.path.join(train_img_dir, img_path)
            if not os.path.exists(full_img_path):
                print(f"Ảnh không tồn tại: {full_img_path}")
                # Tìm đến ảnh tiếp theo
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('.jpg'):
                    i += 1
                continue
            
            # Mở ảnh để lấy kích thước
            img = Image.open(full_img_path)
            img_width, img_height = img.size
            
            # Đọc số lượng khuôn mặt
            i += 1
            face_count = int(lines[i].strip())
            
            # Tên file nhãn YOLO (thay / bằng _ để tránh tạo thư mục)
            label_name = img_path.replace('/', '_').replace('.jpg', '.txt')
            label_path = os.path.join(yolo_label_dir, label_name)
            
            # Tạo nhãn YOLO
            with open(label_path, 'w') as f_out:
                valid_boxes = 0
                for j in range(face_count):
                    i += 1
                    face_data = lines[i].strip().split()
                    
                    if len(face_data) >= 4:
                        x, y, w, h = map(float, face_data[:4])
                        
                        # Bỏ qua box có kích thước bằng 0
                        if w <= 1 or h <= 1:
                            continue
                        
                        # Chuyển đổi sang định dạng YOLO: <class> <x_center> <y_center> <width> <height>
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        # Giới hạn giá trị trong khoảng 0-1
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # Ghi vào file
                        f_out.write(f"0 {x_center} {y_center} {width} {height}\n")
                        valid_boxes += 1
                
                if valid_boxes == 0:
                    # Nếu không có box hợp lệ, xóa file nhãn
                    f_out.close()
                    os.remove(label_path)
                else:
                    success_count += 1
            
            # Di chuyển đến ảnh tiếp theo
            i += 1
            
        except Exception as e:
            error_count += 1
            print(f"Lỗi: {str(e)} tại dòng {i}")
            # Tìm đến ảnh tiếp theo
            while i < len(lines) and not lines[i].strip().endswith('.jpg'):
                i += 1
            i += 1
    
    print(f"Kết quả: Thành công: {success_count}, Lỗi: {error_count}")
    
    # Kiểm tra kết quả
    label_files = glob.glob(os.path.join(yolo_label_dir, "*.txt"))
    print(f"Số lượng file nhãn YOLO: {len(label_files)}")
    
    # Kiểm tra file ngẫu nhiên
    if label_files:
        sample_file = label_files[np.random.randint(0, len(label_files))]
        print(f"Nội dung file nhãn mẫu ({sample_file}):")
        with open(sample_file, "r") as f:
            print(f.read())

    return success_count > 0

# Chạy hàm kiểm tra và sửa lỗi
if verify_and_fix_dataset():
    print("Dữ liệu đã sẵn sàng cho huấn luyện!")
else:
    print("Vẫn có vấn đề với dữ liệu, vui lòng kiểm tra lại!")


