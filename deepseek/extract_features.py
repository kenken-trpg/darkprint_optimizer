import cv2
import numpy as np
from skimage.segmentation import slic
import glob, os, sys
# invert_utils の extract_all_features を流用
from invert_utils import extract_all_features

def rule_based_labels(img_bgr, segments, n_segs):
    """簡易ルールで疑似ラベルを生成 (PoCのロジック)"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sat = np.sqrt(lab[:,:,1]**2 + lab[:,:,2]**2)

    orb = cv2.ORB_create(nfeatures=2000)
    kp = orb.detect(gray, None)
    kp_img = np.zeros_like(gray, dtype=np.uint8)
    if kp:
        for pt in [p.pt for p in kp]:
            cv2.circle(kp_img, (int(pt[0]), int(pt[1])), 4, 255, -1)
    density_map = cv2.GaussianBlur(kp_img.astype(np.float32), (25,25),0)/255.0

    labels = np.zeros(n_segs, dtype=np.uint8)
    for lbl in range(n_segs):
        mask = (segments == lbl)
        if np.sum(mask) == 0: continue
        L_mean = np.mean(lab[:,:,0][mask])
        L_std = np.std(lab[:,:,0][mask])
        sat_std = np.std(sat[mask])
        edge_frac = np.mean(edges[mask] > 0)
        density_avg = np.mean(density_map[mask])
        # ルール
        if (L_mean < 50 and L_std < 30 and sat_std < 35 and density_avg < 0.15) or \
           (L_mean > 190 and edge_frac > 0.3):
            labels[lbl] = 1
    return labels

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py image_folder/")
        sys.exit(1)
    folder = sys.argv[1]
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        print("画像なし")
        return

    all_X, all_y = [], []
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None: continue
        h, w = img.shape[:2]
        # リサイズ
        max_side = 800
        if max(h,w) > max_side:
            s = max_side / max(h,w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segs = slic(rgb, n_segments=200, compactness=10, start_label=0)
        n_segs = segs.max() + 1

        X = extract_all_features(img, segs, n_segs)  # invert_utilsの共通関数
        y = rule_based_labels(img, segs, n_segs)
        all_X.append(X)
        all_y.append(y)
        print(f"Processed {fname}")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'superpixel_features.npy'), X_all)
    np.save(os.path.join(data_dir, 'superpixel_labels.npy'), y_all)
    print(f"Saved {len(y_all)} samples.")


if __name__ == "__main__":
    main()