"""
data_clean.py
Convert RSNA Pneumonia Detection Challenge DICOMs to labeled PNGs organized for training/validation splits
and optionally process unlabeled test DICOMs separately.

Usage:
  # Train/Val/Test split on staged train images
  python data.py --train_dir ./rsna/stage_2_train_images --labels_csv ./rsna/stage_2_train_labels.csv --output_dir ./data/real --val_frac 0.1 --seed 42

  # Plus convert unlabeled test images
  python data.py --train_dir ./rsna/stage_2_train_images --labels_csv ./rsna/stage_2_train_labels.csv --test_dir ./rsna/stage_2_test_images --output_dir ./data/real --val_frac 0.1 --seed 42
"""

import os
import argparse
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean RSNA Pneumonia Detection data"
    )
    parser.add_argument(
        '--train_dir', type=str, required=True,
        help='Directory of staged train DICOMs'
    )
    parser.add_argument(
        '--labels_csv', type=str, required=True,
        help='Path to stage_2_train_labels.csv'
    )
    parser.add_argument(
        '--test_dir', type=str, default=None,
        help='Optional directory of unlabeled test DICOMs'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Root directory for organized output'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.1,
        help='Fraction of train for validation'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    return parser.parse_args()


def load_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    return set(df[df['Target'] == 1]['patientId'].astype(str))


def get_dicom_paths(directory):
    paths = {}
    for fname in os.listdir(directory):
        if fname.lower().endswith('.dcm'):
            pid = os.path.splitext(fname)[0]
            paths[pid] = os.path.join(directory, fname)
    return paths


def convert_and_save(dcm_path, out_path):
    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(out_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load train data and labels
    pneumonia_ids = load_labels(args.labels_csv)
    train_paths = get_dicom_paths(args.train_dir)
    train_ids = sorted(train_paths.keys())

    # Split into train/val
    train_ids, val_ids = train_test_split(
        train_ids, test_size=args.val_frac, random_state=args.seed
    )
    splits = {
        'train': train_ids,
        'val': val_ids,
    }
    # Create directories
    for split in splits:
        for lbl in ['normal', 'pneumonia']:
            os.makedirs(os.path.join(args.output_dir, split, lbl), exist_ok=True)

    # Process train/val
    for split, ids in splits.items():
        print(f"Processing {split}: {len(ids)} images")
        for pid in ids:
            label = 'pneumonia' if pid in pneumonia_ids else 'normal'
            out_dir = os.path.join(args.output_dir, split, label)
            convert_and_save(train_paths[pid], os.path.join(out_dir, f"{pid}.png"))

    # Optional test dir (unlabeled)
    if args.test_dir:
        test_paths = get_dicom_paths(args.test_dir)
        test_dir = os.path.join(args.output_dir, 'test', 'unlabeled')
        os.makedirs(test_dir, exist_ok=True)
        print(f"Processing test unlabeled: {len(test_paths)} images")
        for pid, p in test_paths.items():
            convert_and_save(p, os.path.join(test_dir, f"{pid}.png"))

    print("Done. Data organized under", args.output_dir)


if __name__ == '__main__':
    main()
