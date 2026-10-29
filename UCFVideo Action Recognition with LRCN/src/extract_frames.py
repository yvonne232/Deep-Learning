"""One-time UCF50 frame extraction using get_frames and store_frames."""

import argparse

from utils import extract_dataset_frames


def main():
    parser = argparse.ArgumentParser(
        description="Sample JPEG frames from UCF50 .avi files into clip folders."
    )
    parser.add_argument(
        "--video_dir",
        default="./UCF50",
        help="Root with class folders of raw videos (e.g. UCF50/Punch/*.avi)",
    )
    parser.add_argument(
        "--frame_dir",
        default="./UCF50_frames",
        help="Where to write ClassName/v_Action_gXX_cYY/frame*.jpg",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=16,
        help="Uniform samples per video (should match --fr_per_vid in training)",
    )
    args = parser.parse_args()
    extract_dataset_frames(args.video_dir, args.frame_dir, args.n_frames)
    print(f"Frames written to {args.frame_dir}")
    print("Train with --frame_dir pointing at that folder.")


if __name__ == "__main__":
    main()
