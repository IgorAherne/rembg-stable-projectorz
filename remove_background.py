import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import rembg

def is_image_file(path: str) -> bool:
    """Return True if 'path' ends with a typical image extension."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return path.lower().endswith(exts)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from all images in ./code/input using rembg, "
                    "then save them to ./code/output with RGBA channels."
    )
    parser.add_argument(
        "--alpha_matting",
        action="store_true",
        default=False,
        help="Enable alpha matting for sharper edges. Off by default."
    )
    parser.add_argument(
        "--foreground_thresh",
        type=int,
        default=240,
        help="Alpha matting foreground threshold (0–255). Only relevant if alpha_matting is enabled."
    )
    parser.add_argument(
        "--background_thresh",
        type=int,
        default=20,
        help="Alpha matting background threshold (0–255). Only relevant if alpha_matting is enabled."
    )
    parser.add_argument(
        "--erode_size",
        type=int,
        default=10,
        help="Alpha matting erode size. Only relevant if alpha_matting is enabled."
    )
    parser.add_argument(
        "--hard_edge",
        action="store_true",
        default=False,
        help="If set, threshold the alpha channel to produce a purely binary (hard) edge."
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    script_dir = os.path.dirname(__file__)
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Gather all image paths
    all_files = sorted(os.listdir(input_dir))
    image_files = [f for f in all_files if is_image_file(f)]

    if not image_files:
        print("No images found in code/input/. Please put .png, .jpg, etc. there.")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) in {input_dir}. Processing...")

    # Create a single rembg session (model 'u2net' on CPU). 
    # If you need GPU or a different model, change providers or the model name.
    session = rembg.new_session('u2net', providers=["CPUExecutionProvider"])

    # Show alpha matting settings if used
    if args.alpha_matting:
        print("Alpha matting is ENABLED with the following settings:")
        print(f"  foreground_thresh={args.foreground_thresh}")
        print(f"  background_thresh={args.background_thresh}")
        print(f"  erode_size={args.erode_size}")
    else:
        print("Alpha matting is DISABLED.")

    if args.hard_edge:
        print("Hard-edge thresholding of alpha channel is ENABLED.")
    else:
        print("Hard-edge thresholding of alpha channel is DISABLED.")

    for i, filename in enumerate(image_files):
        base, ext = os.path.splitext(filename)
        in_path = os.path.join(input_dir, filename)

        print(f"[{i+1}/{len(image_files)}] Removing background for: {filename}")
        image = Image.open(in_path).convert('RGBA')

        # Perform background removal
        out_image = rembg.remove(
            image,
            session=session,
            alpha_matting=args.alpha_matting,
            alpha_matting_foreground_threshold=args.foreground_thresh,
            alpha_matting_background_threshold=args.background_thresh,
            alpha_matting_erode_size=args.erode_size
        )

        # Optional hard-edge alpha thresholding
        if args.hard_edge:
            rgba = np.array(out_image)
            alpha_channel = rgba[:, :, 3]
            # For a typical 8-bit channel, threshold=128 is fairly standard
            alpha_channel = np.where(alpha_channel > 128, 255, 0).astype(np.uint8)
            rgba[:, :, 3] = alpha_channel
            out_image = Image.fromarray(rgba, mode='RGBA')

        # Save the RGBA (background-removed) result
        out_filename = f"{base}_{i}.png"
        out_path = os.path.join(output_dir, out_filename)
        out_image.save(out_path)

        print(f"   -> Saved removed BG: {out_filename}")

    print("\nAll images processed. Check the 'output' folder.")

if __name__ == '__main__':
    main()
