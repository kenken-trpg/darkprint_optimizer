import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern
import joblib

def extract_all_features(img_bgr, segments, n_segments):
    """スーパーピクセルごとに特徴量を抽出（学習・推論共通）"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    sat = np.sqrt(A**2 + B**2)

    edges = cv2.Canny(gray, 50, 150)
    edge_mag = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))

    # ORB特徴点密度マップ
    orb = cv2.ORB_create(nfeatures=2000)
    kp = orb.detect(gray, None)
    kp_img = np.zeros((h, w), dtype=np.uint8)
    if kp:
        for pt in [p.pt for p in kp]:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(kp_img, (x, y), 4, 255, -1)
    density_map = cv2.GaussianBlur(kp_img.astype(np.float32), (25, 25), 0) / 255.0

    # LBPテクスチャ (一様パターン)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    n_lbp_bins = 10  # uniform P=8 → 10ビン

    features = []
    for lbl in range(n_segments):
        mask = (segments == lbl)
        area = np.sum(mask)
        if area == 0:
            features.append(np.zeros(20 + n_lbp_bins))
            continue

        L_mean = np.mean(L[mask])
        L_std = np.std(L[mask])
        sat_mean = np.mean(sat[mask])
        sat_std = np.std(sat[mask])
        edge_frac = np.mean(edges[mask] > 0)
        edge_avg = np.mean(edge_mag[mask])
        density_avg = np.mean(density_map[mask])
        density_std = np.std(density_map[mask])

        lbp_vals = lbp[mask].astype(np.int32)
        lbp_hist, _ = np.histogram(lbp_vals, bins=n_lbp_bins, range=(0, n_lbp_bins-1))
        lbp_hist = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else np.zeros(n_lbp_bins)

        moments = cv2.moments(mask.astype(np.uint8))
        if moments['mu20'] + moments['mu02'] > 1e-5:
            aspect = moments['mu20'] / (moments['mu20'] + moments['mu02'])
        else:
            aspect = 0.5
        ys, xs = np.where(mask)
        cx, cy = np.mean(xs), np.mean(ys)
        dist_center = np.sqrt((cx-w/2)**2 + (cy-h/2)**2) / np.sqrt((w/2)**2 + (h/2)**2)

        feat = [
            L_mean, L_std, sat_mean, sat_std,
            edge_frac, edge_avg,
            density_avg, density_std,
            area/(h*w), aspect, dist_center
        ] + lbp_hist.tolist()
        features.append(feat)

    return np.array(features)

def build_adjacency(labels, n_labels):
    """隣接スーパーピクセル関係を構築"""
    adj = [set() for _ in range(n_labels)]
    h, w = labels.shape
    for y in range(h):
        for x in range(w):
            cur = labels[y, x]
            if x < w-1:
                nxt = labels[y, x+1]
                if nxt != cur:
                    adj[cur].add(nxt)
                    adj[nxt].add(cur)
            if y < h-1:
                nxt = labels[y+1, x]
                if nxt != cur:
                    adj[cur].add(nxt)
                    adj[nxt].add(cur)
    return [np.array(list(s)) for s in adj]

def generate_inversion_mask(img_bgr, model_path):
    """学習済みモデルを用いて反転マスクを生成"""
    # 高速化のため長辺800pxにリサイズ
    h, w = img_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    else:
        img = img_bgr.copy()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segments = slic(rgb, n_segments=200, compactness=10, start_label=0)
    n_segs = segments.max() + 1

    features = extract_all_features(img, segments, n_segs)
    rf = joblib.load(model_path)
    proba = rf.predict_proba(features)[:, 1]  # invertクラス確率

    invert_sp = (proba > 0.5).astype(np.uint8)

    # 隣接補正：周囲が反転対象なら確率0.3以上も拾う
    adj = build_adjacency(segments, n_segs)
    for i in range(n_segs):
        if invert_sp[i] == 0 and proba[i] > 0.3:
            if np.any(invert_sp[adj[i]] == 1):
                invert_sp[i] = 1

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for lbl in range(n_segs):
        if invert_sp[lbl]:
            mask[segments == lbl] = 1

    # 元の画像サイズに合わせる
    if mask.shape != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask

def apply_invert(img_bgr, mask):
    """反転マスクに従って画素を反転（元画像を直接変更）"""
    inv = 255 - img_bgr
    img_bgr[mask == 1] = inv[mask == 1]
    return img_bgr