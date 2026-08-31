import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.api import DepthEstimator
from src.config import logger

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    try:
        estimator = DepthEstimator(model_path="models/unet_resnet34_depth_384.onnx")
        artifacts["model"] = estimator
        logger.info("The model has been sucessfully loaded into memory!")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise RuntimeError("Failed to initialize model") from e

    yield

    logger.info("Resource cleanup on shutdown...")
    artifacts.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    is_ready = "model" in artifacts
    return {"status": "ok" if is_ready else "error", "service": "depth-estimator"}


@app.post("/predict")
async def predict_depth_map(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        logger.error(f"Expected image, got {file.content_type} instead")
        raise HTTPException(status_code=400)

    model = artifacts["model"]
    image_bytes = await file.read()
    depth_png_bytes = model.predict(image_bytes)

    return Response(content=depth_png_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
