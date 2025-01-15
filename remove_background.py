import os
import sys
import glob
import numpy as np
from PIL import Image
import rembg

def is_image_file(path: str) -> bool:
    """Return True if 'path' ends with a typical image extension."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return path.lower().endswith(exts)

def main():
    """
    Processes every image in ./code/input/:
      - Removes background with rembg
      - Saves RGBA result to ./code/output/<basename>_<i>.png
      - Saves the alpha channel as a grayscale mask to ./code/output/<basename>_<i>_mask.png
    """
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

    # Initialize a single rembg session (U2Net on CPU to save VRAM)
    # If you prefer a different session or model, adjust here.
    session = rembg.new_session('u2net', providers=["CPUExecutionProvider"])

    for i, filename in enumerate(image_files):
        base, ext = os.path.splitext(filename)
        in_path = os.path.join(input_dir, filename)
        
        print(f"[{i+1}/{len(image_files)}] Removing background for: {filename}")
        image = Image.open(in_path).convert('RGBA')
        
        # Perform background removal
        out_image = rembg.remove(image, session=session)
        
        # Save the RGBA (background-removed) result
        out_filename = f"{base}_{i}.png"
        out_path = os.path.join(output_dir, out_filename)
        out_image.save(out_path)
        
        # Generate and save the mask (alpha channel as grayscale)
        out_mask_filename = f"{base}_{i}_mask.png"
        out_mask_path = os.path.join(output_dir, out_mask_filename)
        
        # Extract alpha channel and make a grayscale mask
        out_np = np.array(out_image)
        alpha = out_np[..., 3]  # alpha channel
        mask_img = Image.fromarray(alpha, mode='L')
        mask_img.save(out_mask_path)
        
        print(f"   -> Saved removed BG:  {out_filename}")
        print(f"   -> Saved alpha mask: {out_mask_filename}")

    print("\nAll images processed. Check the 'output' folder.")

if __name__ == '__main__':
    main()
