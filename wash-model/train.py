from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Start training
    model.train(
        data=r"C:\Users\halde\Documents\surfbee\wash-model\dataset.yaml",
        epochs=50,
        imgsz=640,
        device=0,  # Change to "cpu" if you do not have an NVIDIA GPU
        workers=2,
        patience=20
    )

if __name__ == "__main__":
    main()