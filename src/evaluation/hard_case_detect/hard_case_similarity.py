"""
Hard case similarity analysis for model error investigation.

Compares test error cases against training objects using pretrained embeddings.
Helps determine if errors (e.g., glass → plastic) are due to data similarity
or model behavior.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

from src.logging.logger import get_logger

logger = get_logger(__name__)


class HardCaseSimilarityAnalyzer:
    """
    Compare test error cases against training objects
    using pretrained image embeddings.

    For each case:
        1. Crop GT bbox from test image (query)
        2. Compare with train embeddings from target classes
        3. Calculate margin: best_sim(pred_class) - best_sim(gt_class)
        4. Determine which class the object is closer to

    This helps answer: Is the model confusing glass with plastic
    because they look similar, or because of model behavior?
    """

    def __init__(
        self,
        train_images_dir: Path,
        train_labels_dir: Path,
        class_names: List[str],
        device: str = "cuda",
    ):
        """
        Initialize the analyzer.

        Args:
            train_images_dir: Path to training images directory.
            train_labels_dir: Path to training labels directory.
            class_names: List of class names in order of indices.
            device: 'cuda' or 'cpu'.
        """
        self.train_images_dir = Path(train_images_dir)
        self.train_labels_dir = Path(train_labels_dir)
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = weights.transforms()

        # Cache for train embeddings
        self.train_embeddings = {}

    # ============================================================
    # Public API
    # ============================================================

    def build_train_index(self, target_classes: List[str]) -> None:
        """
        Extract embeddings for training objects of target classes.

        Args:
            target_classes: List of class names to index.
        """
        logger.info("Building train index for classes: %s", target_classes)

        for class_name in target_classes:
            class_id = self._get_class_id(class_name)
            samples = self._collect_class_samples(class_id)

            if not samples:
                logger.warning("No samples found for class: %s", class_name)
                continue

            embeddings = []
            for sample in samples:
                image = self._load_image(sample["image_path"])
                if image is None:
                    continue

                crop = self._crop_bbox(image, sample["bbox"])
                if crop is None:
                    continue

                embedding = self._extract_embedding(crop)
                embeddings.append({
                    "image_path": sample["image_path"],
                    "bbox": sample["bbox"],
                    "embedding": embedding,
                })

            self.train_embeddings[class_name] = embeddings
            logger.info("  Indexed %d samples for %s", len(embeddings), class_name)

    def analyze_case(
        self,
        case: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare one test error case against training objects.

        Args:
            case: Error case from ErrorAnalyzer with keys:
                - image_path: str
                - gt_class: str
                - pred_class: str
                - gt_bbox: List[float] (xyxy format)
                - pred_bbox: List[float] (xyxy format)
                - iou: float
                - confidence: float
            top_k: Number of nearest neighbors to return.

        Returns:
            Analysis result with nearest neighbors and margin.
        """
        image_path = Path(case["image_path"])
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        crop = self._crop_bbox(image, case["gt_bbox"])
        if crop is None:
            raise ValueError(f"Invalid GT bbox: {case['gt_bbox']}")

        query_embedding = self._extract_embedding(crop)

        result = {
            "image_path": str(image_path),
            "gt_class": case["gt_class"],
            "pred_class": case["pred_class"],
            "iou": case["iou"],
            "confidence": case["confidence"],
            "nearest_neighbors": {},
            "best_similarity": {},
            "margin": None,
            "nearest_class": None,
        }

        best_similarities = {}

        for class_name, samples in self.train_embeddings.items():
            if not samples:
                continue

            neighbors = []
            for sample in samples:
                similarity = self._cosine_similarity(
                    query_embedding,
                    sample["embedding"],
                )
                neighbors.append({
                    "image_path": sample["image_path"],
                    "bbox": sample["bbox"],
                    "similarity": round(similarity, 4),
                })

            neighbors.sort(key=lambda x: x["similarity"], reverse=True)
            result["nearest_neighbors"][class_name] = neighbors[:top_k]
            best_similarities[class_name] = neighbors[0]["similarity"]

        # Calculate margin
        if self.train_embeddings:
            gt_class = case["gt_class"]
            pred_class = case["pred_class"]

            best_gt = best_similarities.get(gt_class, 0.0)
            best_pred = best_similarities.get(pred_class, 0.0)

            result["best_similarity"] = {
                gt_class: best_gt,
                pred_class: best_pred,
            }

            margin = best_pred - best_gt
            result["margin"] = round(margin, 4)

            # Determine which class the object is closer to
            if margin > 0:
                result["nearest_class"] = pred_class
            else:
                result["nearest_class"] = gt_class

        return result

    def analyze_cases(
        self,
        cases: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple test error cases.

        Args:
            cases: List of error cases.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of analysis results.
        """
        results = []
        for i, case in enumerate(cases):
            logger.info("Analyzing case %d/%d", i + 1, len(cases))
            try:
                result = self.analyze_case(case, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error("Error analyzing case: %s", e)
                continue

        return results

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Path:
        """
        Generate a summary report from analysis results.

        Args:
            results: List of analysis results.
            output_dir: Directory to save the report.

        Returns:
            Path to the report file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate statistics
        total_cases = len(results)
        nearest_counts = {}

        for result in results:
            nearest = result.get("nearest_class")
            if nearest:
                nearest_counts[nearest] = nearest_counts.get(nearest, 0) + 1

        # Calculate margin statistics
        margins = [r["margin"] for r in results if r.get("margin") is not None]
        avg_margin = sum(margins) / len(margins) if margins else 0.0

        # Create report
        report = {
            "total_cases": total_cases,
            "nearest_class_counts": nearest_counts,
            "margin_stats": {
                "avg": round(avg_margin, 4),
                "min": round(min(margins), 4) if margins else None,
                "max": round(max(margins), 4) if margins else None,
            },
            "results": results,
        }

        # Save report
        report_path = output_dir / "similarity_analysis.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save summary text
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("HARD CASE SIMILARITY ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total cases analyzed: {total_cases}\n\n")
            f.write("Nearest class distribution:\n")
            for class_name, count in nearest_counts.items():
                f.write(f"  {class_name}: {count} ({count/total_cases*100:.1f}%)\n")
            f.write(f"\nAverage margin (pred - gt): {avg_margin:.4f}\n")
            f.write("  (Positive = closer to predicted class)\n")
            f.write("  (Negative = closer to ground truth class)\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("=" * 60 + "\n")
            plastic_ratio = nearest_counts.get("plastic", 0) / total_cases * 100 if total_cases > 0 else 0
            if plastic_ratio > 70:
                f.write("⚠️  Most cases are closer to PLASTIC in embedding space.\n")
                f.write("   → The confusion is likely due to DATA SIMILARITY.\n")
                f.write("   → Consider adding more diverse glass samples to training data.\n")
            elif plastic_ratio < 30:
                f.write("⚠️  Most cases are closer to GLASS in embedding space.\n")
                f.write("   → The confusion is likely due to MODEL BEHAVIOR.\n")
                f.write("   → Consider improving training or model architecture.\n")
            else:
                f.write("ℹ️  Cases are mixed. Both data and model factors may be involved.\n")

        logger.info("Report saved to: %s", report_path)
        return report_path

    # ============================================================
    # Training Data Collection
    # ============================================================

    def _collect_class_samples(self, class_id: int) -> List[Dict[str, Any]]:
        """Collect all training objects of a specific class."""
        samples = []
        label_files = list(self.train_labels_dir.glob("*.txt"))

        for label_path in label_files:
            image_path = self._find_image(label_path.stem)
            if image_path is None:
                continue

            image = self._load_image(image_path)
            if image is None:
                continue

            height, width = image.shape[:2]

            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cid = int(parts[0])
                    if cid != class_id:
                        continue

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bbox_w = float(parts[3])
                    bbox_h = float(parts[4])

                    bbox = self._yolo_to_xyxy(
                        x_center, y_center, bbox_w, bbox_h, width, height
                    )

                    samples.append({
                        "image_path": str(image_path),
                        "bbox": bbox,
                    })

        return samples

    # ============================================================
    # Image Processing
    # ============================================================

    def _load_image(self, image_path: str):
        """Load image as RGB."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _crop_bbox(self, image, bbox: List[float]):
        """Crop bounding box from image."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    def _find_image(self, stem: str) -> Optional[Path]:
        """Find image file by stem."""
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        for ext in extensions:
            img_path = self.train_images_dir / f"{stem}{ext}"
            if img_path.exists():
                return img_path
        return None

    # ============================================================
    # Embedding
    # ============================================================



    @torch.no_grad()
    def _extract_embedding(self, image) -> np.ndarray:
        """Extract image embedding using pretrained CNN."""
        # اگر numpy است، به PIL تبدیل کن
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image
        
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy()
    # ============================================================
    # Similarity
    # ============================================================

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))

    # ============================================================
    # Bounding Box Conversion
    # ============================================================

    @staticmethod
    def _yolo_to_xyxy(
        x_center: float,
        y_center: float,
        bbox_w: float,
        bbox_h: float,
        img_w: int,
        img_h: int,
    ) -> List[float]:
        """Convert YOLO normalized bbox to xyxy format."""
        x_center *= img_w
        y_center *= img_h
        bbox_w *= img_w
        bbox_h *= img_h

        x1 = x_center - bbox_w / 2
        y1 = y_center - bbox_h / 2
        x2 = x_center + bbox_w / 2
        y2 = y_center + bbox_h / 2

        return [x1, y1, x2, y2]

    # ============================================================
    # Helpers
    # ============================================================

    def _get_class_id(self, class_name: str) -> int:
        """Get class ID from class name."""
        if class_name not in self.class_to_id:
            raise ValueError(f"Unknown class: {class_name}")
        return self.class_to_id[class_name]