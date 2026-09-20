from pathlib import Path
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.inference.predictor import ONNXPredictor


def create_router(predictor: ONNXPredictor) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/predict/live",
        operation_id="predict_live",
        summary="Live camera frame inference",
    )
    async def predict_live(
        file: UploadFile = File(...),
    ):
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Frame must be an image.",
            )

        suffix = Path(file.filename or "frame.jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(await file.read())
            temp_path = Path(tmp.name)

        try:
            result = predictor.predict_image(temp_path)

            return result.to_dict()

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Live inference failed: {exc}",
            )

        finally:
            if temp_path.exists():
                temp_path.unlink()

    return router