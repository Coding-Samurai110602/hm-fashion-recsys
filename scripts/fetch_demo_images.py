"""Download demo images for articles appearing in recommendations/history of sample customers.

Uses the Kaggle dataset image structure: images are at {article_id[:3]}/{article_id}.jpg
in the H&M Kaggle dataset. Downloads up to cap=2000 files, resumable, rate-limited.

Usage:
    python scripts/fetch_demo_images.py [--cap 2000] [--delay 0.1]

Requires: kaggle CLI configured (kaggle datasets download -d h-and-m-personalized-fashion-recommendations)
Output: data/images/{article_id[:3]}/{article_id}.jpg (gitignored)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUNDLE_SYNTHETIC = ROOT / "artifacts" / "bundle_synthetic"
IMAGE_DIR = ROOT / "data" / "images"


def check_kaggle() -> bool:
    """Return True if kaggle CLI is available and configured."""
    if shutil.which("kaggle") is None:
        print(
            "kaggle CLI not found. Install with: pip install kaggle\n"
            "Then configure ~/.kaggle/kaggle.json with your API key.\n"
            "Skipping image download."
        )
        return False
    return True


def load_article_ids() -> list[str]:
    """Load article IDs from the synthetic bundle's article_feats parquet."""
    try:
        import polars as pl

        feats_path = BUNDLE_SYNTHETIC / "state" / "article_feats.parquet"
        if not feats_path.exists():
            print(f"Bundle not found at {feats_path}. Run: make synthetic-bundle")
            return []
        df = pl.read_parquet(feats_path, columns=["article_id"] if "article_id" in pl.read_parquet(feats_path).columns else None)
        if "article_id" in df.columns:
            return df["article_id"].to_list()
        print("No article_id column in bundle feats. Using empty list.")
        return []
    except ImportError:
        print("polars not available — returning empty article list.")
        return []
    except Exception as exc:
        print(f"Could not load bundle article IDs: {exc}")
        return []


def download_image(article_id: str, delay: float) -> bool:
    """Download a single image using kaggle CLI. Returns True on success."""
    dest = IMAGE_DIR / article_id[:3] / f"{article_id}.jpg"
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    # The kaggle H&M dataset stores images at images/{article_id[:3]}/{article_id}.jpg
    kaggle_path = f"images/{article_id[:3]}/{article_id}.jpg"
    try:
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "h-and-m-personalized-fashion-recommendations",
                "-f",
                kaggle_path,
                "-p",
                str(IMAGE_DIR / article_id[:3]),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and dest.exists():
            time.sleep(delay)
            return True
        return False
    except (subprocess.TimeoutExpired, Exception):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download demo images from Kaggle.")
    parser.add_argument("--cap", type=int, default=2000, help="Max images to download")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between downloads")
    args = parser.parse_args()

    if not check_kaggle():
        sys.exit(0)

    article_ids = load_article_ids()
    if not article_ids:
        print("No article IDs to download.")
        sys.exit(0)

    print(f"Found {len(article_ids)} articles. Downloading up to {args.cap} images.")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for article_id in article_ids[: args.cap]:
        dest = IMAGE_DIR / article_id[:3] / f"{article_id}.jpg"
        if dest.exists():
            skipped += 1
            continue
        success = download_image(article_id, args.delay)
        if success:
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"  Downloaded {downloaded} images...")
        else:
            failed += 1

    total_size = sum(f.stat().st_size for f in IMAGE_DIR.rglob("*.jpg"))
    print(
        f"\nDone. Downloaded: {downloaded}, Skipped (already existed): {skipped}, "
        f"Failed: {failed}"
    )
    print(f"Total size: {total_size / 1e6:.1f} MB in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
