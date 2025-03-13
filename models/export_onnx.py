from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO("yolov11n_320_visdrone_nut_uavdt_voc.pt")

# Export the model to ONNX format
model.export(format="onnx")