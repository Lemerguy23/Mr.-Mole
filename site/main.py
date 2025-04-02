from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from mako.template import Template
from PIL import Image
import io
import os
import time
import uuid

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    template = Template(filename='templates/index.html')
    return HTMLResponse(content=template.render())

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    os.makedirs("static", exist_ok=True)
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    #Превращение изображения в ч/б и вывод его на странице для примера, то есть ч/б в перспективе заменится на модель
    processed_image = image.convert("L")
    filename = f"processed_{uuid.uuid4().hex}.png"
    processed_image_path = f"static/{filename}"
    processed_image.save(processed_image_path)
    template = Template(filename='templates/index.html')
    html_content = template.render(
        processed_image_url=f"/{processed_image_path}"
    )

    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)