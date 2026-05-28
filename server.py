from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import base64

app = Flask(__name__)

# Allow requests from your website
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLO model once
model = YOLO(r"runs/detect/train-4/weights/best.pt")


@app.route('/predict', methods=['POST'])
def predict():

    # Safety check
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image uploaded'
        }), 400

    file = request.files['image']

    # Save uploaded image
    temp_path = 'temp.jpg'
    file.save(temp_path)

    # Run YOLO prediction
    results = model(temp_path, conf=0.25,imgsz=640)

    detections = []

    # Extract detections
    for box in results[0].boxes:

        confidence = float(box.conf[0])
        cls = int(box.cls[0])

        xyxy = box.xyxy[0].tolist()

        detections.append({
            'confidence': confidence,
            'class': cls,
            'xyxy': xyxy
        })

    # Draw boxes onto image
    plotted_image = results[0].plot()

    # Encode image as JPG
    success, buffer = cv2.imencode('.jpg', plotted_image)

    if not success:
        return jsonify({
            'error': 'Failed to encode image'
        }), 500

    # Convert image → base64 string
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    max_conf = max(
        [d['confidence'] for d in detections],
        default=0
    )

    return jsonify({

        # Detection info
        'detections': detections,

        # Highest confidence
        'max_confidence': max_conf,

        # Boolean result
        'litter_detected': max_conf > 0.25,

        # IMAGE WITH BOXES
        'image': image_base64

    })


@app.route('/')
def home():
    return "YOLO detection server is running."


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000
    )