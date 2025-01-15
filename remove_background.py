import os
import sys
from PIL import Image
import rembg

def main():
    """
    A simple script that takes an image from code/input/ (named input.png or input.jpg)
    and removes its background. The result is saved to code/output/result.png.
    """

    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Try to locate an input image
    possible_inputs = ['input.png', 'input.jpg']
    input_path = None
    for fname in possible_inputs:
        test_path = os.path.join(input_dir, fname)
        if os.path.isfile(test_path):
            input_path = test_path
            break

    if not input_path:
        print("No input image found in code/input/. Please place 'input.png' or 'input.jpg' there.")
        sys.exit(1)

    # Load the image
    print(f"Reading image: {input_path}")
    input_image = Image.open(input_path).convert('RGB')

    # Perform background removal
    print("Removing background...")
    output_image = rembg.remove(input_image)  # Rembg returns RGBA with alpha

    # Save to code/output/result.png
    result_path = os.path.join(output_dir, 'result.png')
    output_image.save(result_path)
    print(f"Saved background-removed image as: {result_path}")

if __name__ == '__main__':
    main()
