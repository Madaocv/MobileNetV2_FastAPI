from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import logging

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
model_name = "google/mobilenet_v2_1.0_224"
preprocessor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.error(f"Invalid file format: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only jpg and png are supported."
            )
    try:
        image = Image.open(file.file)
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        raise HTTPException(
            status_code=400,
            detail="Could not open image. Make sure the file is not corrupted."
            )
    try:
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_idxs = torch.topk(probs, top_k, dim=-1)
        top_probs = top_probs.squeeze().tolist()
        top_idxs = top_idxs.squeeze().tolist()
        predictions = [{"label": model.config.id2label[idx], "probability": prob} for idx, prob in zip(top_idxs, top_probs)]
        logger.info(f"Successful prediction for file: {file.filename}")
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
            )
