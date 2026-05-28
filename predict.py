#from ultralytics import YOLO
 
#model = YOLO(r"C:\Users\halde\Documents\surfbee\runs\detect\train-3\weights\best.pt")
 
#results = model("resized.png")
 
#for box in results[0].boxes:
#    cls = int(box.cls[0])
#    conf = float(box.conf[0])
 
#    print("Class:", cls)
#    print("Confidence:", conf)
 

from ultralytics import YOLO
import os

# # Replace 'run1' with your actual run folder if it's named differently
MODEL_PATH = r"C:\Users\halde\Documents\surfbee\runs\detect\train-4\weights\best.pt"
TEST_IMAGES_DIR = r"C:\Users\halde\Documents\surfbee\wash-model\data\v2\test\images"
#
if not os.path.exists(MODEL_PATH):
    print(f"Error: Could not find your model file at {MODEL_PATH}")
    print("Make sure your model finished training successfully!")
    exit()

model = YOLO(MODEL_PATH)

print("Running test inference...")

results = model.predict(
     source=TEST_IMAGES_DIR,
     save=True,       # Saves the visual images with boxes drawn on them
     conf=0.25,       # Confidence threshold (only show matches > 25% sure)
     line_width=2     # Bounding box line thickness
)

print("\nTesting Complete!")
print("Check the folder 'runs/detect/predict/' to see the output images with bounding boxes!")
 