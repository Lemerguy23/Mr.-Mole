import sys
sys.path.append(r'venv\Lib\site-packages') # Я НЕ ЗНАЮ, НО ПОЧЕМУ-ТО МОЙ ВЕНВ ЭТО НЕ ВИДЕЛ
import fastapi
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mako.template import Template
from mako.lookup import TemplateLookup
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uuid

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from faqdata.faq_data import FAQ_ITEMS

MODEL_PATH = r"CNN\checkpoints\model_DenseNet121_not_full150_Dp_6_AdamW_VeryBigD_Flips_We6_12_0.63.h5"
IMG_SIZE = (224, 224)
input_tensor = tf.keras.Input(shape=(*IMG_SIZE, 3))
base_model = DenseNet121(weights=None, include_top=False, input_tensor=input_tensor)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.6)(x)
output_tensor = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.load_weights(MODEL_PATH)

app = FastAPI()

app.mount("/static", StaticFiles(directory="Mr.-Mole/site/static"), name="static")
templates = TemplateLookup(directories=['Mr.-Mole/site/templates'])


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.get_template("index.mako").render()



@app.post("/check-mole")
async def check_mole(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Only images allowed")

        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data))
        
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]
        class_idx = 1 if prediction > 0.5 else 0

        return {"class": int(class_idx)}

    except Exception as e:
        return JSONResponse({
            "status": "error", 
            "message": str(e)
        }, status_code=500)


@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    return templates.get_template("faq.mako").render(
        faq_items=FAQ_ITEMS
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)