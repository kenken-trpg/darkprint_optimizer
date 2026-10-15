import cv2, sys
from invert_utils import generate_inversion_mask, apply_invert

import os
MODEL = os.path.join(os.path.dirname(__file__), 'dark_invert_rf.pkl')

def main():
    if len(sys.argv) != 3:
        print("Usage: python dark_invert_ml.py input.png output.png")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    img = cv2.imread(in_path)
    if img is None:
        print("画像が読み込めません")
        sys.exit(1)

    mask = generate_inversion_mask(img, MODEL)
    result = apply_invert(img.copy(), mask)

    cv2.imwrite(out_path, result)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()