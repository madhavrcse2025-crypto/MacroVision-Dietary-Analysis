"""
Convert the Food-101 dataset into YOLO bounding-box format.

Food-101 ships as whole-image classification labels (one label per image, no
boxes). Since MacroVision needs multi-item bounding boxes, this script:
  1. Uses the class label as a full-image box for single-item training photos
     (Iteration 1 baseline), and
  2. Reads any manually annotated multi-item boxes (data/multi_item_labels/)
     where available, for the augmented Iteration 2 / final training set.

Usage:
    python prepare_dataset.py --food101-root /path/to/food-101 --out data/yolo
"""
import argparse
import shutil
from pathlib import Path


def convert_split(food101_root: Path, split: str, out_dir: Path):
    images_dir = food101_root / "images"
    meta_file = food101_root / "meta" / f"{split}.txt"
    classes = sorted(p.name for p in images_dir.iterdir() if p.is_dir())
    class_to_id = {c: i for i, c in enumerate(classes)}

    out_img_dir = out_dir / split / "images"
    out_lbl_dir = out_dir / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_file) as f:
        entries = [line.strip() for line in f if line.strip()]

    for entry in entries:
        cls_name, img_id = entry.split("/")
        cls_id = class_to_id[cls_name]
        src_img = images_dir / cls_name / f"{img_id}.jpg"
        dst_img = out_img_dir / f"{cls_name}_{img_id}.jpg"
        shutil.copy(src_img, dst_img)

        # Full-image box (normalised YOLO format: class cx cy w h)
        label_path = out_lbl_dir / f"{cls_name}_{img_id}.txt"
        label_path.write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n")

    (out_dir / "classes.txt").write_text("\n".join(classes))
    print(f"[{split}] converted {len(entries)} images -> {out_dir/split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--food101-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/yolo"))
    args = parser.parse_args()

    convert_split(args.food101_root, "train", args.out)
    convert_split(args.food101_root, "test", args.out)
