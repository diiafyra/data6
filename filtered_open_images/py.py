from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

# 1. Tải mô hình CNN để phân loại
print("1. Tải mô hình CNN để phân loại")
cnn_model = load_model('D:/cmcpt2/ai/Object-Detection/MoHinhCNN/ketqua/cifar10_model_01.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Tải mô hình YOLO
print("2. Tải mô hình YOLO")
path_weight = "D:/cmcpt2/ai/Object-Detection/datatrain/yolov3.weights"
path_cfg = "D:/cmcpt2/ai/Object-Detection/datatrain/yolov3.cfg"
net = cv2.dnn.readNet(path_weight, path_cfg)  # Khởi tạo YOLO

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 3. Đọc và xử lý ảnh đầu vào
test_images_dir = "D:/cmcpt2/ai/Object-Detection/filtered_open_images/test/images"
test_labels_dir = "D:/cmcpt2/ai/Object-Detection/filtered_open_images/test/labels"

# Hàm để lấy danh sách các file ảnh trong thư mục
def load_image_paths(image_dir):
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Tạo danh sách ảnh test
test_images = load_image_paths(test_images_dir)

# 4. Tính toán dự đoán và thực hiện đánh giá
y_true = []  # Nhãn thật
y_pred = []  # Dự đoán từ mô hình

for image_path in test_images:
    # Đọc ảnh và nhãn tương ứng
    img = cv2.imread(image_path)
    label_file = os.path.basename(image_path).replace('.jpg', '.txt')
    label_path = os.path.join(test_labels_dir, label_file)
    
    # Kiểm tra nếu tệp nhãn tồn tại và không rỗng
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        # Chọn nhãn của đối tượng đầu tiên trong ảnh (hoặc xử lý nhiều đối tượng nếu cần)
        true_class = int(labels[0].split()[0])  # Lấy class đầu tiên từ nhãn (YOLO format)

        # Lấy tên lớp từ class_names và chuyển nó thành chữ thường để so khớp
        true_class_name = class_names[true_class].lower()

        # 5. Phát hiện đối tượng bằng YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)  # Đảm bảo net đã được khởi tạo
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1])
                    h = int(detection[3] * img.shape[0])
                    if w != 0 and h != 0:
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        crop_img = img[y:y+h, x:x+w]
                        if crop_img.size > 0:
                            # 6. Phân loại đối tượng bằng CNN
                            cropped_image = cv2.resize(crop_img, (32, 32))
                            cropped_image = np.expand_dims(cropped_image, axis=0)
                            cropped_image = cropped_image / 255.0
                            predictions = cnn_model.predict(cropped_image)
                            predicted_class = np.argmax(predictions)
                            predicted_class_name = class_names[predicted_class].lower()  # Đảm bảo tên lớp là chữ thường
                            
                            y_pred.append(predicted_class_name)
                            y_true.append(true_class_name)
    else:
        print(f"Warning: Label file for {image_path} is missing or empty. Skipping this image.")

# 7. Đánh giá hiệu suất với các tham số như Accuracy, Precision, Recall
print("Đánh giá mô hình")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# 8. Vẽ biểu đồ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
