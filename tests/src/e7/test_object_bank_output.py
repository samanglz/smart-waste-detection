import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "E7_object_bank"
OBJECTS_DIR = OUTPUT_DIR / "objects"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def check_mask(mask_path: Path):
    mask = Image.open(mask_path)

    errors = []

    # 1. Mode
    if mask.mode != "L":
        errors.append(f"mask mode is {mask.mode}, expected L")

    # 2. Size
    width, height = mask.size

    if width == 0 or height == 0:
        errors.append("mask has invalid dimensions")

    # 3. Pixel values
    values = set(mask.getdata())

    if not values:
        errors.append("mask contains no pixels")
        return errors, mask

    # We expect binary mask: 0 and 255
    non_binary = values - {0, 255}

    if non_binary:
        errors.append(
            f"mask is not binary; unexpected values: "
            f"{sorted(non_binary)[:10]}"
        )

    # 4. Completely black / completely white
    if values == {0}:
        errors.append("mask is completely black")

    if values == {255}:
        errors.append("mask is completely white")

    return errors, mask


def check_object(object_path: Path, mask: Image.Image):
    errors = []

    obj = Image.open(object_path)

    # 1. RGBA
    if obj.mode != "RGBA":
        errors.append(
            f"object mode is {obj.mode}, expected RGBA"
        )

    # 2. Same dimensions
    if obj.size != mask.size:
        errors.append(
            f"size mismatch: object={obj.size}, mask={mask.size}"
        )

    # 3. Alpha channel
    if obj.mode == "RGBA":
        alpha = obj.getchannel("A")
        alpha_values = set(alpha.getdata())

        if not alpha_values:
            errors.append("alpha channel contains no pixels")

        elif alpha_values == {0}:
            errors.append("alpha channel is completely transparent")

        elif alpha_values == {255}:
            # This is not necessarily an error.
            # The object can legitimately fill the whole crop.
            pass

    return errors


def check_metadata(
    metadata_path: Path,
    object_image: Image.Image,
    mask_image: Image.Image,
):
    errors = []

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        return [f"failed to read metadata: {exc}"]

    # Crop
    crop = metadata.get("crop", {})
    crop_width = crop.get("width")
    crop_height = crop.get("height")

    if crop_width != object_image.width:
        errors.append(
            f"metadata crop width={crop_width}, "
            f"object width={object_image.width}"
        )

    if crop_height != object_image.height:
        errors.append(
            f"metadata crop height={crop_height}, "
            f"object height={object_image.height}"
        )

    # Mask
    mask_info = metadata.get("mask", {})

    area_pixels = mask_info.get("area_pixels")
    bbox_area = mask_info.get("bbox_area_pixels")
    occupancy = mask_info.get("occupancy_ratio")
    score = mask_info.get("segmentation_score")

    if area_pixels is None:
        errors.append("missing mask.area_pixels")

    if bbox_area is None:
        errors.append("missing mask.bbox_area_pixels")

    if occupancy is None:
        errors.append("missing mask.occupancy_ratio")

    elif not 0.0 <= occupancy <= 1.0:
        errors.append(
            f"occupancy_ratio={occupancy} outside [0, 1]"
        )

    if score is None:
        errors.append("missing mask.segmentation_score")

    elif not 0.0 <= score <= 1.0:
        errors.append(
            f"segmentation_score={score} outside [0, 1]"
        )

    return errors


def check_manifest():
    errors = []

    if not MANIFEST_PATH.exists():
        return ["manifest.json does not exist"]

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return [f"failed to read manifest: {exc}"]

    if not isinstance(manifest, dict):
        errors.append("manifest root is not a dictionary")
        return errors

    # Basic metadata
    if manifest.get("experiment") != "E7":
        errors.append(
            f"experiment={manifest.get('experiment')}, expected E7"
        )

    if manifest.get("stage") != "object_bank":
        errors.append(
            f"stage={manifest.get('stage')}, expected object_bank"
        )

    if manifest.get("split") != "train":
        errors.append(
            f"split={manifest.get('split')}, expected train"
        )

    # Objects
    objects = manifest.get("objects")

    if not isinstance(objects, list):
        errors.append("manifest.objects is not a list")
        return errors

    if len(objects) == 0:
        errors.append("manifest.objects is empty")

    # Count consistency
    num_objects = manifest.get("num_objects")

    if num_objects != len(objects):
        errors.append(
            f"num_objects={num_objects}, "
            f"but manifest contains {len(objects)} objects"
        )

    # Failure count
    num_failures = manifest.get("num_failures")

    if num_failures is None:
        errors.append("missing num_failures")

    # Validate every manifest record
    required_fields = {
        "object_id",
        "class_id",
        "class_name",
        "source_image",
        "source_label",
        "object_path",
        "mask_path",
        "metadata_path",
        "segmentation_score",
        "touches_image_border",
    }

    for index, record in enumerate(objects, start=1):

        if not isinstance(record, dict):
            errors.append(
                f"object record {index} is not a dictionary"
            )
            continue

        missing = required_fields - record.keys()

        if missing:
            errors.append(
                f"object record {index} missing fields: "
                f"{sorted(missing)}"
            )

        score = record.get("segmentation_score")

        if score is not None and not 0.0 <= score <= 1.0:
            errors.append(
                f"object record {index}: "
                f"segmentation_score={score} outside [0, 1]"
            )

    return errors




def main():
    print("=" * 70)
    print("E7 OBJECT BANK - OUTPUT VALIDATION")
    print("=" * 70)

    print(f"Output directory: {OUTPUT_DIR}")

    if not OUTPUT_DIR.exists():
        print("\nERROR: Output directory does not exist.")
        return

    object_dirs = sorted(
        p for p in OBJECTS_DIR.rglob("*")
        if p.is_dir()
        and (p / "object.png").exists()
        and (p / "mask.png").exists()
        and (p / "metadata.json").exists()
    )

    print(f"Objects found: {len(object_dirs)}")

    total_errors = 0

    for index, object_dir in enumerate(object_dirs, start=1):

        object_path = object_dir / "object.png"
        mask_path = object_dir / "mask.png"
        metadata_path = object_dir / "metadata.json"

        object_errors = []

        # Mask
        mask_errors, mask = check_mask(mask_path)
        object_errors.extend(
            [f"mask: {error}" for error in mask_errors]
        )

        # Object
        object_errors.extend(
            [
                f"object: {error}"
                for error in check_object(object_path, mask)
            ]
        )

        # Metadata
        object_image = Image.open(object_path)

        object_errors.extend(
            [
                f"metadata: {error}"
                for error in check_metadata(
                    metadata_path,
                    object_image,
                    mask,
                )
            ]
        )

        if object_errors:
            print(f"\n[{index}] {object_dir.name} ❌")

            for error in object_errors:
                print(f"    - {error}")

            total_errors += len(object_errors)

        else:
            print(f"[{index}] {object_dir.name} ✓")

    # Manifest
    manifest_errors = check_manifest()

    print("\n" + "-" * 70)
    print("MANIFEST")

    if manifest_errors:
        for error in manifest_errors:
            print(f"❌ {error}")
            total_errors += 1
    else:
        print("✓ manifest.json is valid")

    # Final result
    print("\n" + "=" * 70)

    if total_errors == 0:
        print("OBJECT BANK VALIDATION: PASSED")
        print(f"Validated objects: {len(object_dirs)}")
    else:
        print("OBJECT BANK VALIDATION: FAILED")
        print(f"Total errors: {total_errors}")

    print("=" * 70)


if __name__ == "__main__":
    main()