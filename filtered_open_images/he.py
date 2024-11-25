from ultralytics import YOLO

# Tải mô hình YOLOv8 (chọn mô hình phù hợp như 'yolov8n.pt' hoặc 'yolov8s.pt')
model = YOLO("yolov8n.pt")  # Chọn mô hình nhỏ 'yolov8n.pt'

# Huấn luyện mô hình với dữ liệu của bạn
model.train(data='D:/cmcpt2/ai/Object-Detection/filtered_open_images/data.yaml', epochs=20, imgsz=640, batch=16)
