from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mako.template import Template
from PIL import Image
import io
import os
import uuid

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    template = Template(filename='templates/index.html')
    return HTMLResponse(content=template.render())

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    os.makedirs("static/uploads", exist_ok=True)
    file_ext = file.filename.split('.')[-1]
    filename = f"processed_{uuid.uuid4().hex}.{file_ext}"
    output_path = f"static/uploads/{filename}"
    try:
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = image.convert("L")
        processed_image.save(output_path)
        
        return JSONResponse({
            "status": "success",
            "image_url": f"/static/uploads/{filename}"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error", 
            "message": str(e)
        }, status_code=500)

@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    template = Template(filename='templates/faq.html')
    return HTMLResponse(content=template.render())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)