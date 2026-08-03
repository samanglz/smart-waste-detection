"""
YOLO inference implementation.
"""

from pathlib import Path
from typing import Union, List, Dict, Any

from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger


logger = get_logger(__name__)


class YOLOInference:
    """
    Inference handler for YOLO models.

    Provides methods for running predictions on images, batches, and videos.
    """

    def __init__(self, model: YOLOModel):
        """
        Initialize inference with a trained model.

        Args:
            model: Trained YOLOModel instance.
        """
        self.model = model
        logger.info("YOLOInference initialized with model: %s", model.weight_path)

    def predict_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Run prediction on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of predictions with class_id, confidence, and bbox.
        """
        logger.info("Predicting on image: %s", image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(source=str(image_path))
        
        return self._format_results(results, image_path)

    def predict_batch(self, image_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run prediction on all images in a directory.

        Args:
            image_dir: Directory containing images.

        Returns:
            Dictionary mapping image_path to predictions.
        """
        logger.info("Predicting on batch from: %s", image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        results = self.model.predict(source=str(image_dir))
        
        return self._format_batch_results(results)

    def predict_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """
        Run prediction on a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of frames with predictions.
        """
        logger.info("Predicting on video: %s", video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        results = self.model.predict(source=str(video_path))
        
        return self._format_video_results(results)

    def predict_with_confidence(self, image_path: Path, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Run prediction with custom confidence threshold.

        Args:
            image_path: Path to the image file.
            conf_threshold: Minimum confidence to include predictions.

        Returns:
            List of filtered predictions.
        """
        logger.info("Predicting with confidence threshold: %f", conf_threshold)
        
        results = self.model.predict(source=str(image_path), conf=conf_threshold)
        
        return self._format_results(results, image_path)

    def _format_results(self, results, image_path: Path) -> List[Dict[str, Any]]:
        """
        Format YOLO results to a standard dictionary format.

        Args:
            results: YOLO prediction results.
            image_path: Path to the image.

        Returns:
            List of formatted predictions.
        """
        formatted = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    formatted.append({
                        'class_id': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'image_path': str(image_path)
                    })
        
        logger.info("Found %d predictions in image", len(formatted))
        return formatted

    def _format_batch_results(self, results) -> Dict[str, List[Dict[str, Any]]]:
        """Format batch prediction results."""
        formatted = {}
        
        for result in results:
            image_path = str(result.path)
            formatted[image_path] = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    formatted[image_path].append({
                        'class_id': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                    })
        
        logger.info("Processed %d images in batch", len(formatted))
        return formatted

    def _format_video_results(self, results) -> List[Dict[str, Any]]:
        """Format video prediction results."""
        formatted = []
        
        for idx, result in enumerate(results):
            frame_data = {'frame': idx, 'predictions': []}
            
            if result.boxes is not None:
                for box in result.boxes:
                    frame_data['predictions'].append({
                        'class_id': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                    })
            
            formatted.append(frame_data)
        
        logger.info("Processed %d video frames", len(formatted))
        return formatted

    def save_predictions(self, results: Dict[str, Any], output_path: Path):
        """
        Save prediction results to a file.

        Args:
            results: Prediction results.
            output_path: Path to save the results.
        """
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("Predictions saved to: %s", output_path)