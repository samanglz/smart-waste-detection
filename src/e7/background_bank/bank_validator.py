from __future__ import annotations

import json
from pathlib import Path

from src.e7.background_bank.validator import BackgroundValidator


class BackgroundBankValidator:
    """
    Validate all background images in the E7 Background Bank.

    Cross-image duplicate detection is handled here by maintaining
    hashes of already accepted images.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    def __init__(
        self,
        bank_dir: Path,
        validator: BackgroundValidator | None = None,
    ) -> None:
        self.bank_dir = Path(bank_dir)

        self.images_dir = self.bank_dir / "images"
        self.manifests_dir = self.bank_dir / "manifests"
        self.reports_dir = self.bank_dir / "reports"

        self.validator = (
            validator
            if validator is not None
            else BackgroundValidator()
        )

    def validate(self) -> dict:
        """
        Validate the complete background bank.

        Returns:
            Validation summary dictionary.
        """

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory does not exist: {self.images_dir}"
            )

        if not self.images_dir.is_dir():
            raise NotADirectoryError(
                f"Images path is not a directory: {self.images_dir}"
            )

        self.manifests_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_paths = self._collect_images()

        all_results = []

        accepted_content_hashes: set[str] = set()
        accepted_perceptual_hashes: list[str] = []

        for image_path in image_paths:
            result = self.validator.validate(
                image_path=image_path,
                existing_content_hashes=accepted_content_hashes,
                existing_perceptual_hashes=accepted_perceptual_hashes,
            )

            result_dict = {
                "image_path": result.image_path,
                "width": result.width,
                "height": result.height,
                "aspect_ratio": result.aspect_ratio,
                "quality_score": result.quality_score,
                "content_hash": result.content_hash,
                "perceptual_hash": result.perceptual_hash,
                "usable": result.usable,
                "reasons": list(result.reasons),
            }

            all_results.append(result_dict)

            if result.usable:
                if result.content_hash is not None:
                    accepted_content_hashes.add(
                        result.content_hash
                    )

                if result.perceptual_hash is not None:
                    accepted_perceptual_hashes.append(
                        result.perceptual_hash
                    )

        approved = [
            result
            for result in all_results
            if result["usable"]
        ]

        rejected = [
            result
            for result in all_results
            if not result["usable"]
        ]

        summary = {
            "total_images": len(all_results),
            "approved_images": len(approved),
            "rejected_images": len(rejected),
            "approval_rate": (
                len(approved) / len(all_results)
                if all_results
                else 0.0
            ),
        }

        self._write_json(
            self.manifests_dir / "all.json",
            all_results,
        )

        self._write_json(
            self.manifests_dir / "approved.json",
            approved,
        )

        report = {
            "summary": summary,
            "rejected": rejected,
        }

        self._write_json(
            self.reports_dir / "validation_report.json",
            report,
        )

        return summary

    def _collect_images(self) -> list[Path]:
        """
        Collect supported image files recursively.
        """

        return sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in self.SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def _write_json(
        path: Path,
        data,
    ) -> None:
        """
        Write JSON with readable formatting.
        """

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )