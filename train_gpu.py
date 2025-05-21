from ultralytics import YOLO

# Tải mô hình YOLOv8s (phù hợp với RTX 3050 4GB VRAM)
model = YOLO('yolov8s.pt')  # 'yolov8n.pt' nếu gặp vấn đề về VRAM

# Cấu hình tối ưu cho RTX 3050 và 8GB RAM
results = model.train(
    data='D:/YOLO/datasets/wider_yolo_fixed/data.yaml',  # Đường dẫn đến file data.yaml
    epochs=50,           # Huấn luyện 50 epochs
    imgsz=640,           # Kích thước ảnh tốt nhất cho face detection
    batch=16,            # Batch size phù hợp cho RTX 3050
    workers=2,           # Số workers thấp để giảm tiêu thụ RAM
    device=0,            # Sử dụng GPU (thiết bị 0)
    patience=15,         # Early stopping nếu không cải thiện sau 15 epochs
    optimizer='AdamW',   # Optimizer tốt nhất cho YOLOv8
    lr0=0.01,            # Learning rate khởi tạo
    lrf=0.01,            # Learning rate cuối
    cos_lr=True,         # Cosine annealing learning rate
    augment=True,        # Tăng cường dữ liệu
    cache=True,          # Cache images để tăng tốc huấn luyện
    project='D:/YOLO/runs',
    name='wider_face_gpu',
    exist_ok=True,      
    verbose=True,        # In chi tiết
    amp=True,            # Automatic Mixed Precision để tăng tốc trên GPU
    close_mosaic=10,     # Tắt mosaic ở 10 epochs cuối để cải thiện hiệu suất
)

# Đánh giá mô hình sau khi huấn luyện
metrics = model.val()
print(f"Metrics: {metrics}")