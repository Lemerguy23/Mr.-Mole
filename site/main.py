import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mako.template import Template
from mako.lookup import TemplateLookup
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uuid


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
interpreter = None


def predict_mole_tflite(image: Image.Image) -> float:
    img = image.resize((260, 260))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])
    return float(prediction[0][0])


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = TemplateLookup(directories=[TEMPLATES_DIR])


@app.on_event("startup")
async def load_model_on_startup():
    import get_model

    global interpreter
    interpreter = get_model.try_load_model()


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.get_template("main_page.mako").render()


@app.get("/analyze", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.get_template("analyze.mako").render()


@app.post("/check-mole")
async def check_mole(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "Only images allowed")

        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        prediction = predict_mole_tflite(img)
        if prediction < 0.266:
            class_idx = 0
        elif prediction < 0.316:
            class_idx = 1
        else:
            class_idx = 2

        return {"class": int(class_idx), "confidence": float(prediction)}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    from faqdata.faq_data import FAQ_ITEMS

    return templates.get_template("faq.mako").render(faq_items=FAQ_ITEMS)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2468)
