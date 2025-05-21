import cv2
from ultralytics import YOLO
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv8 face detector
yolo_model = YOLO("D:/YOLO/runs/wider_face_gpu/weights/best.pt")

# Load ResNet34 age predictor
resnet_model = models.resnet34(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)
resnet_model.load_state_dict(torch.load("resnet34_utkface.pth", map_location=torch.device("cpu")))
resnet_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)

# Transform for ResNet34
max_age = 116
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_age(face_img_pil):
    face_tensor = transform(face_img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet_model(face_tensor).item()
    age = round(output * max_age)
    return age

# Nhập đường dẫn ảnh
image_path = input("Nhập đường dẫn ảnh: ").strip()
# Nếu bạn thích gán cứng thì dùng raw string: 
# image_path = r"D:\YOLO\20_Family_Group_Family_Group_20_3.jpg"

# Đọc ảnh
img = cv2.imread(image_path)
if img is None:
    print("Không đọc được ảnh, vui lòng kiểm tra lại đường dẫn.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Dò mặt bằng YOLO
results = yolo_model.predict(img_rgb, conf=0.5, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else []

# Dự đoán tuổi và vẽ khung mặt
for box in boxes:
    x1, y1, x2, y2 = box
    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        continue

    face_pil = Image.fromarray(face)
    age = predict_age(face_pil)

    # Vẽ khung và ghi tuổi
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{age} age", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Lưu ảnh kết quả
cv2.imwrite("result_with_age.jpg", img)
print("Đã lưu ảnh kết quả vào result_with_age.jpg")

# Hiển thị ảnh kết quả
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
