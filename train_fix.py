from ultralytics import YOLO
import multiprocessing

def main():
    # Tải mô hình
    model = YOLO('yolov8s.pt')
    
    # Huấn luyện với cấu hình tối ưu
    results = model.train(
        data='D:/YOLO/datasets/wider_yolo_fixed/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,           # Giảm batch size
        workers=0,         # Tắt workers để tránh lỗi multiprocessing
        device=0,          # Vẫn sử dụng GPU
        patience=15,
        optimizer='AdamW',
        project='D:/YOLO/runs',
        name='wider_face_gpu',
        cache=False,       # Tắt cache để giảm dùng RAM
        exist_ok=True
    )

if __name__ == '__main__':
    # Giải quyết lỗi multiprocessing
    multiprocessing.freeze_support()
    main()

# Tạo file face_detector_app.py
import tkinter as tk
from tkinter import filedialog, Button, Label
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

class FaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detector")
        self.root.geometry("800x600")
        
        # Tải mô hình
        self.model = YOLO('D:/YOLO/runs/wider_face_gpu/weights/best.pt')
        
        # Biến lưu trạng thái webcam
        self.is_webcam_active = False
        self.webcam_thread = None
        
        # UI Elements
        self.create_widgets()
    
    def create_widgets(self):
        # Button Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Buttons
        Button(btn_frame, text="Mở Ảnh", command=self.open_image, width=15).grid(row=0, column=0, padx=5)
        self.webcam_btn = Button(btn_frame, text="Bật Webcam", command=self.toggle_webcam, width=15)
        self.webcam_btn.grid(row=0, column=1, padx=5)
        Button(btn_frame, text="Thoát", command=self.root.quit, width=15).grid(row=0, column=2, padx=5)
        
        # Display area
        self.display_label = Label(self.root)
        self.display_label.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng")
        Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_image(self):
        # Dừng webcam nếu đang chạy
        if self.is_webcam_active:
            self.toggle_webcam()
        
        # Mở file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        # Phát hiện khuôn mặt
        self.status_var.set(f"Đang xử lý ảnh...")
        self.root.update()
        
        results = self.model(file_path, conf=0.4, device=0)
        img = results[0].plot()
        
        # Hiển thị
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.resize_image(img, 780, 500)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.display_label.config(image=img_tk)
        self.display_label.image = img_tk
        
        num_faces = len(results[0].boxes)
        self.status_var.set(f"Đã phát hiện {num_faces} khuôn mặt")
    
    def toggle_webcam(self):
        if self.is_webcam_active:
            # Dừng webcam
            self.is_webcam_active = False
            self.webcam_btn.config(text="Bật Webcam")
            self.status_var.set("Đã dừng webcam")
        else:
            # Bật webcam
            self.is_webcam_active = True
            self.webcam_btn.config(text="Tắt Webcam")
            self.status_var.set("Đang kích hoạt webcam...")
            
            # Bắt đầu thread xử lý webcam
            self.webcam_thread = threading.Thread(target=self.process_webcam)
            self.webcam_thread.daemon = True
            self.webcam_thread.start()
    
    def process_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while self.is_webcam_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Phát hiện khuôn mặt
            results = self.model(frame, conf=0.4, device=0)
            annotated_frame = results[0].plot()
            
            # Hiển thị số khuôn mặt
            num_faces = len(results[0].boxes)
            self.status_var.set(f"Đã phát hiện {num_faces} khuôn mặt")
            
            # Chuyển đổi và hiển thị
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.resize_image(img, 780, 500)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.display_label.config(image=img_tk)
            self.display_label.image = img_tk
            self.root.update()
        
        cap.release()
    
    def resize_image(self, img, max_width, max_height):
        width, height = img.size
        ratio = min(max_width/width, max_height/height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)

# Start the app
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectorApp(root)
    root.mainloop()
