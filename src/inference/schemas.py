from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]

    # Production diagnostics
    area_ratio: float = 0.0
    width_ratio: float = 0.0
    height_ratio: float = 0.0
    touches_border: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass(slots=True)
class Performance:
    """
    Inference performance metrics.
    """
    inference_time_ms: float
    fps: float
    num_detections: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(slots=True)
class PredictionResult:
    """
    Standard prediction output.
    """
    predictions: List[Detection]
    performance: Performance

    def to_dict(self) -> Dict:
        return {
            "predictions": [
                det.to_dict() for det in self.predictions
            ],
            "performance": self.performance.to_dict(),
        }   