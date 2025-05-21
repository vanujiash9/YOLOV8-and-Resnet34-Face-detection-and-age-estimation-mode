import os
from PIL import Image
from ultralytics import YOLO  # Cần pip install ultralytics

def convert_annotation(annotation_file, base_image_dir, label_save_dir):
    os.makedirs(label_save_dir, exist_ok=True)
    with open(annotation_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']

    i = 0
    while i < len(lines):
        line = lines[i]
        if '/' not in line:
            i += 1
            continue

        image_name = line
        i += 1
        num_faces = int(lines[i])
        i += 1

        bboxes = lines[i:i+num_faces]
        i += num_faces

        image_path = os.path.join(base_image_dir, image_name.replace('/', os.sep))
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        with Image.open(image_path) as img:
            w_img, h_img = img.size

        label_lines = []
        for bbox in bboxes:
            parts = bbox.split()
            x, y, bw, bh = map(float, parts[:4])

            x_center = x + bw / 2
            y_center = y + bh / 2

            x_center /= w_img
            y_center /= h_img
            bw /= w_img
            bh /= h_img

            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        label_path = os.path.join(label_save_dir, os.path.splitext(os.path.basename(image_name))[0] + ".txt")
        with open(label_path, 'w') as f_label:
            f_label.write("\n".join(label_lines))
    print(f"Finished converting {annotation_file}")

def main():
    # Đường dẫn của bạn
    train_images_dir = r"D:/YOLO/datasets/wider_face/images/WIDER_train/WIDER_train/images"
    val_images_dir = r"D:/YOLO/datasets/wider_face/images/WIDER_val/WIDER_val/images"

    train_anno = r"D:/YOLO/datasets/wider_face/labels/wider_face_train_bbx_gt.txt"
    val_anno = r"D:/YOLO/datasets/wider_face/labels/wider_face_val_bbx_gt.txt"

    train_labels_dir = r"D:/YOLO/datasets/wider_face/labels/train"
    val_labels_dir = r"D:/YOLO/datasets/wider_face/labels/val"

    convert_annotation(train_anno, train_images_dir, train_labels_dir)
    convert_annotation(val_anno, val_images_dir, val_labels_dir)

    # Data.yaml chuẩn bị sẵn (hoặc tạo file data.yaml trong script cũng được)
    data_yaml = """
    train: D:/YOLO/datasets/wider_face/images/WIDER_train/WIDER_train/images
    val: D:/YOLO/datasets/wider_face/images/WIDER_val/WIDER_val/images

    nc: 1
    names: ['face']
    """
    with open("data.yaml", "w") as f:
        f.write(data_yaml)

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(data="data.yaml", epochs=50, imgsz=640)

if __name__ == "__main__":
    main()
