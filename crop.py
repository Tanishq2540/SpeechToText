import cv2
import os

def split_with_overlap(image_path, output_folder, overlap_px=150):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    h, w, _ = img.shape
    mid_h, mid_w = h // 2, w // 2

    # Define boundaries with overlap
    # We add overlap_px to the "end" of the first half and subtract from the "start" of the second
    crops = {
        "top_left":     img[0 : mid_h + overlap_px, 0 : mid_w + overlap_px],
        "top_right":    img[0 : mid_h + overlap_px, mid_w - overlap_px : w],
        "bottom_left":  img[mid_h - overlap_px : h, 0 : mid_w + overlap_px],
        "bottom_right": img[mid_h - overlap_px : h, mid_w - overlap_px : w]
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for name, quad in crops.items():
        cv2.imwrite(f"{output_folder}/{name}_overlap.jpg", quad)
        print(f"Saved {name} with {overlap_px}px overlap.")

# Usage
split_with_overlap("ocr test.jpeg", "ocr_chunks", overlap_px=200)