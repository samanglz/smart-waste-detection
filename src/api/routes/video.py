from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.inference.predictor import ONNXPredictor
from src.inference.video_processor import VideoProcessor


ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/avi",
    "video/quicktime",
    "video/x-msvideo",
}


def create_router(predictor: ONNXPredictor) -> APIRouter:
    router = APIRouter()
    processor = VideoProcessor(predictor)

    @router.post(
        "/predict/video",
        operation_id="predict_video",
        summary="Video inference",
    )
    async def predict_video(
        file: UploadFile = File(...),
    ):
        if file.content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Supported formats: MP4, AVI, MOV",
            )

        tmp_dir = Path(tempfile.mkdtemp())

        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"

        input_video = tmp_dir / f"input{suffix}"
        output_video = tmp_dir / "prediction.mp4"

        input_video.write_bytes(await file.read())

        processor.process(
            input_path=input_video,
            output_path=output_video,
        )

        return FileResponse(
            path=output_video,
            media_type="video/mp4",
            filename="prediction.mp4",
            background=BackgroundTask(
                lambda: shutil.rmtree(tmp_dir, ignore_errors=True)
            ),
        )

    return router