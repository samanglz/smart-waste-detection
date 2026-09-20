import json
from pathlib import Path
from collections import Counter


INPUT_PATH = Path(
    "outputs/yolo11m/E2_targeted_augmentation/"
    "hard_case_similarity_mining.json"
)

OUTPUT_PATH = Path(
    "outputs/yolo11m/E2_targeted_augmentation/"
    "E6_hard_mining_manifest.json"
)


def main():
    print("=" * 70)
    print("E6 HARD-MINING MANIFEST BUILDER")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Mining report not found:\n{INPUT_PATH}"
        )

    print("\n[1] Loading mining report...")
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    selected = data["selected_training_objects"]

    print(f"Hard cases              : {data['total_hard_cases']}")
    print(f"Top-K per case          : {data['top_k_per_case']}")
    print(f"Unique train objects    : {len(selected)}")

    # ------------------------------------------------------------
    # Build manifest
    # ------------------------------------------------------------

    print("\n[2] Building manifest...")

    manifest_objects = []

    for obj in selected:
        selection_count = int(obj["selection_count"])
        max_similarity = float(obj["max_similarity"])

        # Hard-mining priority score.
        #
        # Repeated selection is the strongest signal.
        # Similarity provides a secondary signal.
        #
        # This is a ranking/weighting signal, not a class weight.
        priority_score = (
            selection_count * max_similarity
        )

        manifest_objects.append(
            {
                "image_path": obj["image_path"],
                "bbox": obj["bbox"],
                "class": obj["class"],
                "selection_count": selection_count,
                "max_similarity": max_similarity,
                "priority_score": priority_score,
                "selected_by": obj["selected_by"],
                "global_rank": obj["global_rank"],
            }
        )

    # Highest-priority objects first.
    manifest_objects.sort(
        key=lambda x: (
            x["priority_score"],
            x["selection_count"],
            x["max_similarity"],
        ),
        reverse=True,
    )

    # Reassign rank after sorting.
    for rank, obj in enumerate(manifest_objects, 1):
        obj["hard_mining_rank"] = rank

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    class_counts = Counter(
        obj["class"] for obj in manifest_objects
    )

    selection_counts = Counter(
        obj["selection_count"]
        for obj in manifest_objects
    )

    manifest = {
        "experiment": "E6",
        "method": "embedding_based_hard_mining",
        "source": str(INPUT_PATH),

        "hard_cases": data["total_hard_cases"],
        "top_k_per_case": data["top_k_per_case"],

        "total_unique_training_objects": len(
            manifest_objects
        ),

        "class_distribution": dict(
            class_counts
        ),

        "selection_frequency": {
            str(k): v
            for k, v in sorted(
                selection_counts.items(),
                reverse=True,
            )
        },

        "ranking": {
            "primary": "selection_count",
            "secondary": "max_similarity",
            "priority_score": (
                "selection_count * max_similarity"
            ),
        },

        "objects": manifest_objects,
    }

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    print("\n[3] Saving manifest...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("E4 MANIFEST SUMMARY")
    print("=" * 70)

    print(
        f"Hard cases             : "
        f"{manifest['hard_cases']}"
    )

    print(
        f"Unique train objects   : "
        f"{manifest['total_unique_training_objects']}"
    )

    print("\nClass distribution:")
    for cls, count in class_counts.most_common():
        print(f"  {cls:10} {count}")

    print("\nSelection frequency:")
    for count, n in sorted(
        selection_counts.items(),
        reverse=True,
    ):
        print(
            f"  selected {count}x : {n}"
        )

    print("\nTop 15 hard-mining objects:")

    for obj in manifest_objects[:15]:
        filename = Path(
            obj["image_path"]
        ).name

        print(
            f"{obj['hard_mining_rank']:2}. "
            f"{obj['class']:10} "
            f"count={obj['selection_count']} "
            f"sim={obj['max_similarity']:.4f} "
            f"score={obj['priority_score']:.4f} "
            f"{filename}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()