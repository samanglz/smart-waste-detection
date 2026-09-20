from pathlib import Path
from typing import Iterable, Set

import albumentations as A
import cv2


TARGET_CLASS_IDS: Set[int] = {1, 4}


class TargetedAugmentation:
    """
    E2 targeted augmentation.

    Applies augmentation only to images containing target classes.

    Target classes:
        1 -> glass
        4 -> plastic

    The augmentation policy is intentionally kept identical
    to the original E2 experiment.
    """

    def __init__(
        self,
        target_class_ids: Iterable[int] = TARGET_CLASS_IDS,
    ):
        self.target_class_ids = set(target_class_ids)

        self.transform = A.Compose(
            [
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.8,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.20,
                    contrast_limit=0.20,
                    p=0.7,
                ),
                A.Perspective(
                    scale=(0.02, 0.05),
                    keep_size=True,
                    p=0.4,
                ),
                A.Affine(
                    scale=(0.8, 1.2),
                    p=0.5,
                ),
                A.HorizontalFlip(
                    p=0.5,
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,
                clip=True,
            ),
        )

    def contains_target_class(
        self,
        label_path: Path,
    ) -> bool:
        """
        Check whether a YOLO label file contains
        at least one target class.
        """

        if not label_path.exists():
            return False

        for line in label_path.read_text().splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            if class_id in self.target_class_ids:
                return True

        return False

    def apply(
        self,
        image_path: Path,
        label_path: Path,
        out_image: Path,
        out_label: Path,
    ) -> bool:
        """
        Apply E2 augmentation to one image and its YOLO labels.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        bboxes = []
        classes = []

        for line in label_path.read_text().splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            c, x, y, w, h = parts

            bboxes.append(
                [
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                ]
            )

            classes.append(
                int(float(c))
            )

        result = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=classes,
        )

        out_image.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_label.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_image = cv2.cvtColor(
            result["image"],
            cv2.COLOR_RGB2BGR,
        )

        success = cv2.imwrite(
            str(out_image),
            output_image,
        )

        if not success:
            raise IOError(
                f"Could not write image: {out_image}"
            )

        lines = []

        for cls, box in zip(
            result["class_labels"],
            result["bboxes"],
        ):
            lines.append(
                f"{cls} "
                + " ".join(
                    f"{value:.6f}"
                    for value in box
                )
            )

        out_label.write_text(
            "\n".join(lines)
        )

        return True