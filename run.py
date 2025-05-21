import os
import glob
from PIL import Image

def convert_wider_to_yolo():
    # Đường dẫn thư mục
    wider_train_labels = "D:/YOLO/datasets/wider_face/labels/train/wider_face_train_bbx_gt.txt"
    wider_val_labels = "D:/YOLO/datasets/wider_face/labels/val/wider_face_val_bbx_gt.txt"
    
    # Cấu trúc thư mục đúng (thêm "/images/" vào đường dẫn)
    train_img_dir = "D:/YOLO/datasets/wider_face/images/WIDER/WIDER_train/images"
    val_img_dir = "D:/YOLO/datasets/wider_face/images/WIDER/WIDER_val/WIDER_val/images"
    
    # Tạo thư mục đích cho nhãn YOLO
    train_labels_dir = "D:/YOLO/datasets/wider_face/labels_yolo/train"
    val_labels_dir = "D:/YOLO/datasets/wider_face/labels_yolo/val"
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Chuyển đổi nhãn
    print("Đang chuyển đổi nhãn huấn luyện...")
    convert_annotations(wider_train_labels, train_img_dir, train_labels_dir)
    
    print("Đang chuyển đổi nhãn kiểm định...")
    convert_annotations(wider_val_labels, val_img_dir, val_labels_dir)
    
    print("Đã hoàn thành chuyển đổi!")

def convert_annotations(annotation_file, img_dir, output_dir):
    error_count = 0
    success_count = 0
    skipped_count = 0
    
    with open(annotation_file, "r") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        try:
            # Đọc đường dẫn ảnh
            img_path = lines[i].strip()
            if not img_path.endswith('.jpg'):
                i += 1
                continue
                
            i += 1  # Di chuyển đến dòng số lượng khuôn mặt
            if i >= len(lines):
                break
                
            # Đọc số lượng khuôn mặt
            face_count = int(lines[i].strip())
            i += 1  # Di chuyển đến dòng đầu tiên của dữ liệu khuôn mặt
            
            # Đường dẫn đầy đủ đến ảnh
            full_img_path = os.path.join(img_dir, img_path)
            
            if os.path.exists(full_img_path):
                try:
                    # Mở ảnh và lấy kích thước
                    img = Image.open(full_img_path)
                    img_width, img_height = img.size
                    
                    # Tạo file nhãn - thay thế / bằng _ trong tên file
                    label_path = img_path.replace('/', '_').replace('\\', '_')
                    label_file = os.path.join(output_dir, label_path.replace('.jpg', '.txt'))
                    
                    # Đảm bảo thư mục tồn tại
                    os.makedirs(os.path.dirname(label_file), exist_ok=True)
                    
                    with open(label_file, 'w') as f_out:
                        # Xử lý từng khuôn mặt
                        valid_faces = 0
                        for j in range(face_count):
                            if i + j < len(lines):
                                face_data = lines[i + j].strip().split()
                                if len(face_data) >= 4:
                                    x, y, w, h = map(float, face_data[:4])
                                    
                                    # Bỏ qua các box có kích thước bằng 0
                                    if w <= 0 or h <= 0:
                                        continue
                                    
                                    # Chuyển đổi sang định dạng YOLO
                                    x_center = (x + w/2) / img_width
                                    y_center = (y + h/2) / img_height
                                    width = w / img_width
                                    height = h / img_height
                                    
                                    # Giới hạn giá trị trong khoảng 0-1
                                    x_center = max(0, min(1, x_center))
                                    y_center = max(0, min(1, y_center))
                                    width = max(0, min(1, width))
                                    height = max(0, min(1, height))
                                    
                                    f_out.write(f"0 {x_center} {y_center} {width} {height}\n")
                                    valid_faces += 1
                    
                    success_count += 1
                    # Hiển thị tiến trình
                    if success_count % 100 == 0:
                        print(f"Đã xử lý {success_count} ảnh")
                        
                except Exception as e:
                    error_count += 1
                    print(f"Lỗi xử lý ảnh {full_img_path}: {str(e)}")
            else:
                skipped_count += 1
                if skipped_count <= 5:  # Chỉ hiển thị 5 ảnh đầu tiên bị bỏ qua
                    print(f"Bỏ qua ảnh không tồn tại: {full_img_path}")
            
            # Di chuyển đến ảnh tiếp theo
            i += face_count
            
        except Exception as e:
            error_count += 1
            print(f"Lỗi khi xử lý dòng {i}: {str(e)}")
            # Tìm đến ảnh tiếp theo
            while i < len(lines) and not lines[i].strip().endswith('.jpg'):
                i += 1
    
    print(f"Kết quả: Thành công: {success_count}, Lỗi: {error_count}, Bỏ qua: {skipped_count}")

def update_data_yaml():
    # Cập nhật file data.yaml để sử dụng đường dẫn đúng và labels_yolo
    yaml_content = """
path: D:/YOLO/datasets/wider_face  # đường dẫn tới thư mục dữ liệu
train: images/WIDER/WIDER_train/images  # đường dẫn tương đối tới ảnh huấn luyện
val: images/WIDER/WIDER_val/WIDER_val/images  # đường dẫn tương đối tới ảnh kiểm định

# Các thư mục chứa nhãn YOLO
train_labels: labels_yolo/train  # thư mục chứa nhãn YOLO cho tập huấn luyện
val_labels: labels_yolo/val  # thư mục chứa nhãn YOLO cho tập kiểm định

# Số lượng lớp
nc: 1

# Tên lớp
names: ['face']
"""

    with open("D:/YOLO/datasets/wider_face/data.yaml", "w") as f:
        f.write(yaml_content)
    print("Đã cập nhật file data.yaml")

# Gọi các hàm
print("===== CHUYỂN ĐỔI NHÃN =====")
convert_wider_to_yolo()

print("\n===== CẬP NHẬT FILE DATA.YAML =====")
update_data_yaml()

print("\n===== HOÀN THÀNH =====")
print("Bây giờ bạn có thể huấn luyện mô hình YOLOv8 với dữ liệu đã chuyển đổi")

def update_data_yaml():
    # Cập nhật file data.yaml không dùng tiếng Việt
    yaml_content = """
path: D:/YOLO/datasets/wider_face
train: images/WIDER/WIDER_train/images
val: images/WIDER/WIDER_val/WIDER_val/images

train_labels: labels_yolo/train
val_labels: labels_yolo/val

nc: 1
names: ['face']
"""

    with open("D:\YOLO\datasets\wider_face\data.yaml", "w") as f:
        f.write(yaml_content)
    print("Đã cập nhật file data.yaml")


# Tạo bộ dữ liệu nhỏ (subset) từ WIDER
import os
import random
import shutil

def create_subset(source_images, source_labels, target_images, target_labels, sample_size=500):
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    # Lấy danh sách các file ảnh
    all_images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
    
    # Chọn ngẫu nhiên một tập con
    selected_images = random.sample(all_images, min(sample_size, len(all_images)))
    
    # Sao chép ảnh và nhãn
    for img_name in selected_images:
        label_name = img_name.replace('.jpg', '.txt')
        
        # Sao chép ảnh
        shutil.copy(os.path.join(source_images, img_name), os.path.join(target_images, img_name))
        
        # Sao chép nhãn nếu tồn tại
        label_path = os.path.join(source_labels, label_name)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(target_labels, label_name))

# Sử dụng cho tập train và val
create_subset(
    'D:/YOLO/datasets/wider_face/images/WIDER/WIDER_train/images', 
    'D:/YOLO/datasets/wider_face/labels_yolo/train', 
    'D:/YOLO/datasets/wider_face/images_subset/train',
    'D:/YOLO/datasets/wider_face/labels_yolo_subset/train',
    sample_size=500
)

from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Định nghĩa model ở đây
# Cấu hình huấn luyện siêu nhẹ
results = model.train(
    data='D:\YOLO\datasets\wider_face\data.yaml',
    epochs=10,           # Giảm epochs
    imgsz=160,           # Giảm kích thước ảnh
    batch=1,             # Batch size = 1 cho máy yếu
    workers=0,
    cache=False,
    device='cpu',
    patience=5,          # Early stopping
    project='D:/YOLO/runs',
    name='wider_face_yolov8n_ultralight'
)