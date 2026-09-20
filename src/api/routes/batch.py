from pathlib import Path
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException

from src.inference.predictor import ONNXPredictor


def create_router(predictor: ONNXPredictor) -> APIRouter:
    router = APIRouter()

    @router.post(
    "/predict/batch",
    operation_id="predict_batch",
    summary="Batch image inference",
)
    async def predict_batch(
        files: list[UploadFile] = File(
            ...,
            description="Multiple image files for batch inference",
        )
    ):
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required.",
            )

        temp_paths = []

        try:
            for file in files:
                if (
                    not file.content_type
                    or not file.content_type.startswith("image/")
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid image file: {file.filename}",
                    )

                suffix = (
                    Path(file.filename or "image.jpg").suffix
                    or ".jpg"
                )

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as tmp:
                    tmp.write(await file.read())
                    temp_paths.append(Path(tmp.name))

            results = predictor.predict_batch(temp_paths)

            return {
                "count": len(results),
                "results": [
                    result.to_dict()
                    for result in results
                ],
            }

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Batch inference failed: {exc}",
            )

        finally:
            for temp_path in temp_paths:
                if temp_path.exists():
                    temp_path.unlink()

    return router