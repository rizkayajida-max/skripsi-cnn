from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model(
    "model/model_uang.keras"
)

kelas = [
    "asli_rp100000",
    "asli_rp50000",
    "palsu_rp100000",
    "palsu_rp50000"
]

@app.get("/")
def home():
    return {
        "message": "API CNN Deteksi Uang Aktif"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

    image = image.resize(
        (256,256)
    )

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediksi = model.predict(
        img_array,
        verbose=0
    )

    kelas_idx = np.argmax(prediksi)

    confidence = float(
        np.max(prediksi) * 100
    )

    sorted_pred = np.sort(prediksi[0])

    selisih = float(
        sorted_pred[-1] - sorted_pred[-2]
    )

    if confidence < 85 or selisih < 0.30:

        return {
         "kelas": "tidak_terdeteksi",
         "confidence": confidence
        }

    hasil = kelas[kelas_idx]

    return {
        "kelas": hasil,
        "confidence": confidence
    }