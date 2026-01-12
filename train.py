from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/boukari/5A-TP-DeepLearning/insa-5a-deep-learning/data.yaml", epochs=100, imgsz=640)
