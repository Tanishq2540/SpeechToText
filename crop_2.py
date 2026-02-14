import cv2
import os
import numpy as np

def preprocess_blackboard(img):
    """Preprocess for chalkboard: grayscale, CLAHE, denoise, binarize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, enhanced

def split_with_overlap(image_path, output_folder, overlap_px=150):
    """Your original 4-quadrant split with overlap."""
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    h, w, _ = img.shape
    mid_h, mid_w = h // 2, w // 2

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

def split_into_rows(image_path, output_folder, row_height=140, overlap=50):
    """Horizontal strips - HALF HEIGHT (140px)."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i in range(0, h, row_height - overlap):
        y_end = min(h, i + row_height)
        row = img[i:y_end, 0:w]
        name = f"row_{i//100:02d}_{y_end//100:02d}"
        cv2.imwrite(f"{output_folder}/{name}.jpg", row)
        print(f"Saved {name} ({row.shape[0]}px tall)")

# MAIN PIPELINE
if __name__ == "__main__":
    image_path = "ocr test.jpeg"
    quadrants_folder = "quadrants"
    rows_folder = "rows"
    
    # 1. Full preprocessing
    print("=== Preprocessing ===")
    img = cv2.imread(image_path)
    binary, enhanced = preprocess_blackboard(img)
    cv2.imwrite("preprocessed_binary.jpg", binary)
    cv2.imwrite("preprocessed_enhanced.jpg", enhanced)
    print("Saved preprocessed_binary.jpg and preprocessed_enhanced.jpg")
    
    # 2. Quadrant split
    print("\n=== Quadrant Split ===")
    split_with_overlap(image_path, quadrants_folder, overlap_px=200)
    
    # 3. Row strips (140px height)
    print("\n=== Row Split (140px height) ===")
    split_into_rows(image_path, rows_folder, row_height=140, overlap=50)
    
    print(f"\nDone!")
    print(f"- 4 quadrant chunks in {quadrants_folder}/")
    print(f"- ~20 row chunks in {rows_folder}/ (140px tall)")
