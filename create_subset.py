import os
import glob

def explore_directory(base_path):
    print(f"Kiểm tra thư mục: {base_path}")
    print(f"Thư mục này tồn tại: {os.path.exists(base_path)}")
    
    if not os.path.exists(base_path):
        return
    
    # Liệt kê nội dung thư mục
    contents = os.listdir(base_path)
    dirs = [d for d in contents if os.path.isdir(os.path.join(base_path, d))]
    files = [f for f in contents if os.path.isfile(os.path.join(base_path, f))]
    
    print(f"Số thư mục con: {len(dirs)}")
    print(f"Số file: {len(files)}")
    
    if dirs:
        print("Các thư mục con:")
        for d in dirs:
            print(f"  - {d}")
    
    if files:
        print("Mẫu một số file:")
        for f in files[:5]:
            print(f"  - {f}")
    
    # Tìm kiếm tất cả ảnh JPG (bao gồm cả trong thư mục con)
    jpg_files = glob.glob(os.path.join(base_path, "**/*.jpg"), recursive=True)
    print(f"Tìm thấy {len(jpg_files)} ảnh JPG trong thư mục này và các thư mục con.")
    
    if jpg_files:
        print("Mẫu một số đường dẫn ảnh đầy đủ:")
        for jpg in jpg_files[:3]:
            print(f"  - {jpg}")
            # Kiểm tra kích thước file
            print(f"    Kích thước: {os.path.getsize(jpg)} bytes")

# Kiểm tra thư mục train
print("=== KIỂM TRA THƯ MỤC TRAIN ===")
explore_directory("D:/YOLO/datasets/wider_face/images/WIDER/WIDER_train")

# Kiểm tra một cấp bên dưới nếu có
train_dir = "D:/YOLO/datasets/wider_face/images/WIDER/WIDER_train"
if os.path.exists(train_dir):
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if subdirs:
        print("\n=== KIỂM TRA THƯ MỤC CON ===")
        # Kiểm tra thư mục con đầu tiên
        explore_directory(os.path.join(train_dir, subdirs[0]))

# Kiểm tra thư mục images nếu có
images_dir = os.path.join(train_dir, "images")
if os.path.exists(images_dir):
    print("\n=== KIỂM TRA THƯ MỤC IMAGES TRONG TRAIN ===")
    explore_directory(images_dir)

# Kiểm tra annotation file
anno_file = "D:/YOLO/datasets/wider_face/labels/train/wider_face_train_bbx_gt.txt"
print(f"\n=== KIỂM TRA FILE NHÃN ===")
print(f"File nhãn tồn tại: {os.path.exists(anno_file)}")

if os.path.exists(anno_file):
    with open(anno_file, 'r') as f:
        first_lines = [f.readline().strip() for _ in range(5)]
    print("5 dòng đầu tiên của file nhãn:")
    for i, line in enumerate(first_lines):
        print(f"{i+1}: {line}")

# Tìm kiếm rộng hơn
print("\n=== TÌM KIẾM JPG TRONG WIDER FACE ===")
wider_face_dir = "D:/YOLO/datasets/wider_face"
jpg_files = glob.glob(os.path.join(wider_face_dir, "**/*.jpg"), recursive=True)
print(f"Tìm thấy {len(jpg_files)} ảnh JPG trong toàn bộ thư mục wider_face.")

if jpg_files:
    print("Mẫu một số ảnh tìm thấy:")
    sample_jpgs = jpg_files[:5]
    for jpg in sample_jpgs:
        print(f"  - {jpg}")


import os
import glob
from PIL import Image
import shutil

def create_wider_dataset():
    # Đường dẫn chính xác từ kết quả kiểm tra
    train_img_dir = "D:/YOLO/datasets/wider_face/images/WIDER/WIDER_train/WIDER_train/images"
    val_img_dir = "D:/YOLO/datasets/wider_face/images/WIDER/WIDER_val/WIDER_val/images"  # Giả định tương tự cho val
    
    # File annotation
    train_anno = "D:/YOLO/datasets/wider_face/labels/train/wider_face_train_bbx_gt.txt"
    val_anno = "D:/YOLO/datasets/wider_face/labels/val/wider_face_val_bbx_gt.txt"
    
    # Thư mục đích
    output_dir = "D:/YOLO/datasets/wider_yolo_fixed"
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    # Xử lý dữ liệu train
    print("Đang xử lý tập train...")
    train_count = process_wider_annotations(train_anno, train_img_dir, 
                                          f"{output_dir}/images/train",
                                          f"{output_dir}/labels/train")
    
    # Xử lý dữ liệu val
    print("Đang xử lý tập val...")
    val_count = process_wider_annotations(val_anno, val_img_dir, 
                                        f"{output_dir}/images/val",
                                        f"{output_dir}/labels/val")
    
    # Tạo file YAML
    yaml_content = f"""
path: {output_dir}
train: images/train  
val: images/val

nc: 1
names: ['face']
"""
    
    with open(f"{output_dir}/data.yaml", "w", encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"Đã tạo xong dataset: {train_count} ảnh train, {val_count} ảnh val")
    return output_dir

def process_wider_annotations(anno_file, img_dir, out_img_dir, out_label_dir):
    """Xử lý file nhãn WIDER Face và chuyển đổi sang định dạng YOLO"""
    if not os.path.exists(anno_file):
        print(f"Không tìm thấy file nhãn: {anno_file}")
        return 0
        
    if not os.path.exists(img_dir):
        print(f"Không tìm thấy thư mục ảnh: {img_dir}")
        return 0
    
    try:
        with open(anno_file, "r") as f:
            lines = f.readlines()
        
        i = 0
        success_count = 0
        error_count = 0
        
        while i < len(lines):
            try:
                # Đọc đường dẫn ảnh
                img_path = lines[i].strip()
                if not img_path.endswith('.jpg'):
                    i += 1
                    continue
                
                # Đường dẫn đầy đủ đến ảnh - chú ý cấu trúc thư mục
                full_img_path = os.path.join(img_dir, img_path)
                
                # Kiểm tra xem file có tồn tại không
                if not os.path.exists(full_img_path):
                    print(f"Không tìm thấy ảnh: {full_img_path}")
                    i += 1
                    while i < len(lines) and not lines[i].strip().endswith('.jpg'):
                        i += 1
                    continue
                
                # Đọc số khuôn mặt
                i += 1
                if i >= len(lines):
                    break
                    
                face_count = int(lines[i].strip())
                
                # Đọc kích thước ảnh
                img = Image.open(full_img_path)
                img_width, img_height = img.size
                
                # Tạo tên file mới cho YOLO
                new_img_filename = img_path.replace('/', '_')
                dst_img_path = os.path.join(out_img_dir, new_img_filename)
                
                # Tạo thư mục đích nếu cần
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                
                # Sao chép ảnh
                shutil.copy(full_img_path, dst_img_path)
                
                # Tạo file nhãn YOLO
                label_file = os.path.join(out_label_dir, new_img_filename.replace('.jpg', '.txt'))
                
                with open(label_file, 'w') as f_out:
                    valid_faces = 0
                    
                    # Xử lý từng khuôn mặt
                    for j in range(face_count):
                        i += 1
                        if i >= len(lines):
                            break
                            
                        face_data = lines[i].strip().split()
                        
                        if len(face_data) >= 4:
                            x, y, w, h = map(float, face_data[:4])
                            
                            # Bỏ qua box quá nhỏ
                            if w <= 1 or h <= 1:
                                continue
                            
                            # Chuyển sang định dạng YOLO
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            width = w / img_width
                            height = h / img_height
                            
                            # Đảm bảo giá trị nằm trong [0,1]
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            
                            f_out.write(f"0 {x_center} {y_center} {width} {height}\n")
                            valid_faces += 1
                
                success_count += 1
                if success_count % 100 == 0:
                    print(f"Đã xử lý {success_count} ảnh...")
                
                # Di chuyển đến ảnh tiếp theo
                i += 1
                
            except Exception as e:
                error_count += 1
                print(f"Lỗi khi xử lý ảnh tại dòng {i}: {str(e)}")
                # Tìm đến ảnh tiếp theo
                while i < len(lines) and not lines[i].strip().endswith('.jpg'):
                    i += 1
        
        print(f"Kết quả xử lý: Thành công = {success_count}, Lỗi = {error_count}")
        return success_count
    
    except Exception as e:
        print(f"Lỗi tổng thể: {str(e)}")
        return 0

# Chạy hàm chính để tạo dataset
dataset_dir = create_wider_dataset()

# In thông tin để sẵn sàng huấn luyện
print(f"\nDataset đã sẵn sàng tại: {dataset_dir}")
print("Sử dụng lệnh sau để huấn luyện:")
print(f"python -c \"from ultralytics import YOLO; model = YOLO('yolov8s.pt'); model.train(data='{dataset_dir}/data.yaml', epochs=50, imgsz=640, batch=8, device=0)\"")