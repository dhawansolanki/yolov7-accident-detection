from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import torch
from torchvision.transforms import functional as F
import sys
sys.path.insert(0, './yolov7')
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box
import numpy as np

app = Flask(__name__)

# Load YOLOv7 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("model200.pkl", map_location=device)
model.eval()

# Define image processing function
def process_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((640, 640))  # Resize input image to match model's input size
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    return img, img_tensor

# Define detection function
class_names = {
    0: "Accident",
    1: "Non Accident"
}

# Define detection function
def detect_image(image_path):
    print("Detecting objects...")
    img, img_tensor = process_image(image_path)
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)[0]
    if pred is not None:
        img_np = np.array(img)  # Convert PIL Image to NumPy array
        for *xyxy, conf, cls in pred:
            class_name = class_names[int(cls)]  # Get class name based on class index
            plot_one_box(xyxy, img_np, label=class_name, color=(0, 255, 0))
        img_result = Image.fromarray(img_np)  # Convert NumPy array back to PIL Image
        img_result.save("result.png")  # Save result image to a static directory
    print("Detection complete.")
    return "result.png"

# Define route for index.html
@app.route('/')
def index():
    return render_template('index.html')

# Define route for image upload
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join("./yolov7", file.filename)
            file.save(image_path)
            result_image_path = detect_image(image_path)
            return render_template('result.html', result_image=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)