from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from adjust_gamma import adjust_gamma
import shutil
import os
import cv2 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
app = FastAPI()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load the model
cvNet = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/weights.caffemodel")
if cvNet is None:
    raise ValueError("Failed to load the model")

model = tf.keras.models.load_model("models/face_mask_model.keras")
if model is None:
    raise ValueError("Failed to load the model")

# Mount static directory for CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

assign = {'0':'Mask','1':"No Mask"}
gamma = 2.0
img_size = 124
@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    image = cv2.imread(file_location, 1)
    image = adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                im = cv2.resize(frame, (img_size, img_size))
                im = np.array(im) / 255.0
                im = im.reshape(1, 124, 124, 3)
                result = model.predict(im)
                label_Y = 1 if result > 0.5 else 0
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 4)
                cv2.putText(image, assign[str(label_Y)], (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 6)
        except:
            pass
    result_image_filename = "result_" + file.filename
    result_image_path = os.path.join(UPLOAD_FOLDER, result_image_filename)
    cv2.imwrite(result_image_path, image)
    result = f"Uploaded {file.filename}. (Mask detection result here)"
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "result_image": f"/uploads/{result_image_filename}"}
    )