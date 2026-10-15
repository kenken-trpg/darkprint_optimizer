import cv2
import numpy as np
import argparse
import sys

class DarkPrintOptimizer:
    def __init__(self, k_clusters=3, shadow_cut=60, gamma=1.8):
        self.k_clusters = k_clusters
        self.shadow_cut = shadow_cut
        self.gamma = gamma

    def _apply_lut(self, img, shadow_cut, gamma):
        """トーンカーブ(LUT)を用いて暗部を引き締め、コントラストを向上させる[cite: 2]"""
        lut = np.zeros((256, 1), dtype=np.uint8)
        for i in range(256):
            if i < shadow_cut:
                val = 0
            else:
                val = 255 * ((i - shadow_cut) / (255 - shadow_cut))
                val = 255 * (val / 255) ** gamma
            lut[i] = np.clip(val, 0, 255)
        return cv2.LUT(img, lut)

    def _hsv_invert(self, img):
        """色相(Hue)を維持したまま明度を反転させる"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # 明度(V)のみを反転
        hsv[:, :, 2] = 255.0 - hsv[:, :, 2]
        # 再度BGRに変換
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return res

    def process(self, input_path, output_path):
        src = cv2.imread(input_path)
        if src is None:
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        h, w = src.shape[:2]
        
        # 1. k-meansによる背景色の特定[cite: 1, 2]
        small_src = cv2.resize(src, (w//4, h//4))
        pixels = small_src.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, self.k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        bg_cluster_idx = np.argmax(np.bincount(labels.flatten()))
        bg_color = centers[bg_cluster_idx] # float32のまま保持

        # 2. 型安全な背景差分計算 (uint8 underflow 回避)
        src_float = src.astype(np.float32)
        diff = np.linalg.norm(src_float - bg_color, axis=2)
        bg_mask = (diff < 30).astype(np.uint8) * 255

        # 3. コンテンツ領域の抽出と境界のソフト化 (Feathering)[cite: 1]
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((15, 15), np.uint8)
        content_mask = cv2.dilate(edges, kernel, iterations=2)
        # 境界をぼかしてアルファチャンネルとして利用
        content_alpha = cv2.GaussianBlur(content_mask, (15, 15), 0).astype(np.float32) / 255.0
        content_alpha = np.expand_dims(content_alpha, axis=2)

        # 4. UI要素の処理 (色相保持反転 + コントラスト補正)[cite: 1, 2]
        inverted = self._hsv_invert(src)
        ui_processed = self._apply_lut(inverted, self.shadow_cut, self.gamma)
        
        # 5. UIのスケルトン強調[cite: 1, 2]
        skeleton = cv2.Laplacian(gray, cv2.CV_64F)
        skeleton = np.uint8(np.absolute(skeleton))
        _, skeleton_mask = cv2.threshold(skeleton, 30, 255, cv2.THRESH_BINARY)
        ui_processed[skeleton_mask == 255] = [30, 30, 30]

        # 6. 最終合成 (アルファブレンド)
        # 背景を白に強制
        ui_processed[bg_mask == 255] = [255, 255, 255]
        
        # アルファブレンドによる合成: Result = Content * Alpha + UI * (1 - Alpha)
        final = (src_float * content_alpha + ui_processed.astype(np.float32) * (1.0 - content_alpha))
        final = np.clip(final, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, final)
        print(f"Success: Optimized image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ダークモード画像を印刷用に最適化します[cite: 1]")
    parser.add_argument("input_path", help="入力画像のパス")
    parser.add_argument("output_path", nargs="?", default="optimized_for_print.png", help="出力画像のパス")
    parser.add_argument("--gamma", type=float, default=1.8, help="コントラスト補正のガンマ値")
    
    args = parser.parse_args()
    
    optimizer = DarkPrintOptimizer(gamma=args.gamma)
    try:
        optimizer.process(args.input_path, args.output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)