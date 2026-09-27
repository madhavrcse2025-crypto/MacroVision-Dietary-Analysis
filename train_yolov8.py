"""
Fine-tune YOLOv8 on the prepared Food-101 subset.

Matches the training procedure in Section 4.5 of the report:
  - 50 epochs, batch size 16, AdamW, initial LR 0.001 with cosine decay,
    weight decay 0.0005, CIoU + BCE loss (YOLOv8 defaults).
  - Early stopping once validation mAP@0.5 plateaus for 5 epochs (patience=5).

Usage:
    python train_yolov8.py --data food101.yaml --epochs 50
"""
import argparse

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO dataset yaml (train/val paths + class names)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", default="yolov8n.pt", help="Starting checkpoint")
    parser.add_argument("--aug-config", default="training/augmentation.yaml")
    parser.add_argument("--out", default="backend/macrovision_yolov8.pt")
    args = parser.parse_args()

    with open(args.aug_config) as f:
        aug = yaml.safe_load(f)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        weight_decay=0.0005,
        patience=5,
        **aug,
    )
    model.export(format="pt")
    print(f"Training complete. Best weights saved under runs/detect/train/weights/best.pt")
    print(f"Copy the checkpoint to {args.out} for the backend to load it.")


if __name__ == "__main__":
    main()
