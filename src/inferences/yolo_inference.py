"""
YOLO inference implementation.
"""

from pathlib import Path
from typing import Union, List, Dict, Any
import time

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
        
        
        #Measure inferenve time
        start_time = time.time()

        results = self.model.predict(source=str(image_path))
        
        inference_time = time.time() - start_time
        inference_time_ms = inference_time * 1000
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        
        predictions = self._format_results(results, image_path)
        
        
        
        return {
            
            "prediction" : predictions,
            "performance" : {
                "inference_time_in_ms" : round(inference_time_ms , 2),
                "fps" : round(fps, 2),
                "num_detections" : len(predictions),
            }
            
        }

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

        # Measure inference time
        start_time = time.time()

        results = self.model.predict(source=str(image_dir))

        total_time = time.time() - start_time

        predictions = self._format_batch_results(results)

        num_images = len(predictions)
        total_time_ms = total_time * 1000
        avg_inference_time_ms = (total_time / num_images) * 1000 if num_images > 0 else 0.0
        fps = num_images / total_time if total_time > 0 else 0.0

        return {
            "predictions": predictions,
            "performance": {
                "total_time_ms": round(total_time_ms, 2),
                "avg_inference_time_ms": round(avg_inference_time_ms, 2),
                "fps": round(fps, 2),
                "num_images": num_images,
            }
        }


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

        # Measure inference time
        start_time = time.time()

        results = self.model.predict(source=str(video_path))

        predictions = self._format_video_results(results)

        total_time = time.time() - start_time

        num_frames = len(predictions)
        total_time_ms = total_time * 1000
        avg_inference_time_ms = (total_time / num_frames) * 1000 if num_frames > 0 else 0.0
        fps = num_frames / total_time if total_time > 0 else 0.0

        return {
            "predictions": predictions,
            "performance": {
                "total_time_ms": round(total_time_ms, 2),
                "avg_inference_time_ms": round(avg_inference_time_ms, 2),
                "fps": round(fps, 2),
                "num_frames": num_frames,
            }
        }

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
        
        
        
    def predict_with_confidence(
        self,
        image_path: Path,
        conf_threshold: float = 0.25,
        per_class_conf_threshold: Dict[int, float] = None
    ) -> Dict[str, Any]:
        """
        Run prediction with custom confidence thresholds per class.

        Use this method when you want to set different confidence thresholds
        for different classes (e.g., higher threshold for Glass to reduce FP).

        Args:
            image_path: Path to the image file.
            conf_threshold: Global minimum confidence (default: 0.25).
            per_class_conf_threshold: Dict mapping class_id -> confidence threshold.
                                    Example: {1: 0.5}  # Glass with higher threshold

        Returns:
            Dictionary containing:
                - predictions: List of predictions (filtered by thresholds)
                - performance: dict with inference_time_ms, fps, thresholds info

        Example:
            # Increase threshold for Glass (class_id=1) to reduce false positives
            result = inference.predict_with_confidence(
                image_path=Path("image.jpg"),
                conf_threshold=0.25,
                per_class_conf_threshold={1: 0.5}
            )
        """
        logger.info(
            "Predicting with conf_threshold=%f, per_class_thresholds=%s",
            conf_threshold,
            per_class_conf_threshold or {}
        )

        # Measure inference time
        start_time = time.time()

        # Run inference with global threshold
        results = self.model.predict(source=str(image_path), conf=conf_threshold)

        inference_time = time.time() - start_time
        inference_time_ms = inference_time * 1000
        fps = 1.0 / inference_time if inference_time > 0 else 0.0

        # Format predictions and apply per-class filtering
        raw_predictions = self._format_results(results, image_path)
        filtered_predictions = self._apply_per_class_filter(
            raw_predictions,
            conf_threshold,
            per_class_conf_threshold
        )

        return {
            "predictions": filtered_predictions,
            "performance": {
                "inference_time_ms": round(inference_time_ms, 2),
                "fps": round(fps, 2),
                "num_detections": len(filtered_predictions),
                "global_conf_threshold": conf_threshold,
                "per_class_thresholds": per_class_conf_threshold or {},
            }
        }

    def _apply_per_class_filter(
        self,
        predictions: List[Dict[str, Any]],
        global_threshold: float,
        per_class_thresholds: Dict[int, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply per-class confidence thresholds to predictions.

        Args:
            predictions: Raw predictions from YOLO.
            global_threshold: Global confidence threshold.
            per_class_thresholds: Dict mapping class_id -> confidence threshold.

        Returns:
            Filtered predictions.
        """
        if per_class_thresholds is None:
            per_class_thresholds = {}

        filtered = []

        for pred in predictions:
            class_id = pred.get('class_id', -1)
            confidence = pred.get('confidence', 0.0)

            # Use class-specific threshold if available, otherwise global
            threshold = per_class_thresholds.get(class_id, global_threshold)

            if confidence >= threshold:
                filtered.append(pred)

        logger.info(
            "Filtered: %d predictions kept out of %d (global_threshold=%.2f, per_class=%s)",
            len(filtered),
            len(predictions),
            global_threshold,
            per_class_thresholds
        )

        return filtered