# import cv2
# import torch
# from ultralytics import YOLO
# from torchvision import models, transforms
# import torch.nn as nn
# from PIL import Image
# import numpy as np

# # Load YOLOv8 face detector
# yolo_model = YOLO("D:/YOLO/runs/wider_face_gpu/weights/best.pt")

# # Load ResNet34 age predictor
# resnet_model = models.resnet34(pretrained=False)
# resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)
# resnet_model.load_state_dict(torch.load("resnet34_utkface.pth", map_location=torch.device("cpu")))
# resnet_model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# resnet_model.to(device)

# # Transform for ResNet34
# max_age = 116
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # Age prediction function
# def predict_age(face_img_pil):
#     face_tensor = transform(face_img_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = resnet_model(face_tensor).item()
#     age = round(output * max_age)
#     return age

# # Start webcam
# cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to RGB for YOLO
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # YOLO face detection
#     results = yolo_model.predict(frame_rgb, conf=0.5, verbose=False)
#     boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes is not None else []

#     # Draw detections + Predict age
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         face = frame_rgb[y1:y2, x1:x2]

#         if face.size == 0:
#             continue

#         face_pil = Image.fromarray(face)
#         age = predict_age(face_pil)

#         # Draw bounding box + age label
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, f'{age} tuoi', (x1, y1 - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#     # Show frame
#     cv2.imshow("YOLOv8 + Age Estimation", frame)

#     # Exit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



