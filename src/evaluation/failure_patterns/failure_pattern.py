from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FailurePattern:
    """
    A failure pattern discovered from the validation set.

    Validation data is used only to discover the pattern.
    It is never copied into the training dataset.
    """

    pattern_id: str

    error_type: str
    # confusion | false_negative | false_positive

    source_class: str

    target_class: Optional[str] = None

    min_count: int = 1

    priority: int = 0