from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from model import predict

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Clasificador de Carcinoma Ductal Invasivo")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    result = predict(image)
    return result