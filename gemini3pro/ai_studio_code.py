import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from sklearn.cluster import KMeans
import torch

class SmartPrintOptimizer:
    def __init__(self, use_sam=False):
        self.use_sam = use_sam
        self.sam_predictor = None
        if use_sam:
            print("Loading MobileSAM...")
            # ここでモデルの初期化（mobile_sam.ptが必要）
            # from mobile_sam import sam_model_registry, SamPredictor
            # self.sam_predictor = SamPredictor(sam_model_registry["vit_t"](checkpoint="mobile_sam.pt"))

    def get_dominant_background(self, img):
        """K-meansで画像の支配的な背景色（BGR）を特定"""
        small_img = cv2.resize(img, (100, 100))
        pixels = small_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        return kmeans.cluster_centers_[np.argmax(counts)]

    def get_text_mask(self, img):
        """OCRを用いてテキスト領域のマスクを作成"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ダークモード用に反転させてOCR精度を向上
        inv_gray = cv2.bitwise_not(gray)
        
        data = pytesseract.image_to_data(inv_gray, output_type=Output.DICT)
        for i in range(len(data['level'])):
            if int(data['conf'][i]) > 30: # 信頼度30%以上
                (x, y, w_box, h_box) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 255, -1)
        return mask

    def process(self, input_path, output_path):
        # 1. 画像読み込み
        src = cv2.imread(input_path)
        if src is None: return
        h, w = src.shape[:2]

        # 2. 解析：背景色とHSV空間
        bg_color = self.get_dominant_background(src)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # 3. 解析：テキスト領域（OCR）
        print("Analyzing layout (OCR)...")
        text_area_mask = self.get_text_mask(src)

        # 4. 解析：コンテンツ領域（Canny Edge + Saturation）
        print("Analyzing content areas...")
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        content_mask = cv2.dilate(edges, np.ones((12, 12), np.uint8), iterations=1)
        
        # 彩度が高い領域もコンテンツとみなす
        chromatic_mask = s_ch > 40 
        
        # 5. 出力画像の構築
        # 初期状態は「白」
        result = np.full(src.shape, 255, dtype=np.uint8)

        # --- 判定ロジック ---
        
        # A. 保護すべきコンテンツ領域（有彩色 OR エッジ密集地）
        # ※OCRで見つかったテキスト領域はここでは除外する
        protected_mask = ((chromatic_mask) | (content_mask > 0)) & (text_area_mask == 0)
        
        # 明度を少し底上げ（白背景で見やすくするため）
        v_boosted = np.where((v_ch < 160) & protected_mask, cv2.add(v_ch, 40), v_ch)
        hsv_final = cv2.merge([h_ch, s_ch, v_boosted])
        src_boosted = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
        
        result[protected_mask] = src_boosted[protected_mask]

        # B. テキスト領域の処理
        # OCRで文字とされた領域のうち、元の明度が高い（白い）部分を「黒」に反転
        text_pixels = (text_area_mask > 0) & (v_ch > 100)
        result[text_pixels] = [30, 30, 30] # 濃いグレー（印刷時の可読性が高い）

        # C. 背景
        # それ以外は白（初期値のまま）

        # 6. 保存
        cv2.imwrite(output_path, result)
        print(f"Success: {output_path}")

# --- 実行コード ---
if __name__ == "__main__":
    # Tesseractのパス指定（Windowsの場合など）
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    optimizer = SmartPrintOptimizer(use_sam=False)
    optimizer.process("input_dark.png", "output_optimized.png")