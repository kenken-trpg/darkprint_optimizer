import cv2
import numpy as np
import argparse
import os

def process_darkmode_screenshot(input_path, output_path=None, sensitivity=1.0):
    """
    ダークモードスクリーンショットを印刷向けに変換
    """
    # 画像読み込み
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {input_path}")
    
    original = img.copy()
    h, w = img.shape[:2]
    print(f"入力画像: {w}x{h}")

    # ==================== 前処理 ====================
    # LAB色空間（人間の知覚に近い）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]  # 明度

    # ガウスぼかし + CLAHEでコントラスト強調
    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # ==================== ダークUI領域検出 ====================
    # 適応的閾値処理（ダーク背景を抽出）
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8 * sensitivity
    )

    # 形態学演算でノイズ除去・領域結合
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # ==================== 領域フィルタリング ====================
    # 大きな領域のみ抽出（UIパネル候補）
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    min_area = (h * w) * 0.01   # 全体の1%以上
    
    ui_regions = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # 矩形近似度（UIは四角いことが多い）
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        # 面積・形状フィルタ
        if len(approx) >= 4 and area > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            ui_regions += 1

    print(f"検出されたUI領域数: {ui_regions}")

    # マスクを少し膨張させて文字周辺も含める
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # ==================== 選択的反転 ====================
    result = original.copy()
    
    # マスク領域のみ反転
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    inverted = cv2.bitwise_not(original)
    
    # マスク領域を反転画像で置き換え
    result = np.where(mask3 == 255, inverted, result)

    # ==================== 仕上げ調整 ====================
    # 全体の軽いコントラスト調整（印刷向け）
    result = cv2.convertScaleAbs(result, alpha=1.05, beta=5)

    # ==================== 保存 ====================
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_print_optimized{ext}"
    
    cv2.imwrite(output_path, result)
    print(f"出力完了: {output_path}")
    
    return result


# ======================= コマンドライン実行 =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ダークモード印刷最適化ツール (PoC)")
    parser.add_argument("input", help="入力画像パス")
    parser.add_argument("-o", "--output", help="出力画像パス", default=None)
    parser.add_argument("-s", "--sensitivity", type=float, default=1.0,
                        help="感度 (0.5=緩く, 1.5=厳しく)")
    args = parser.parse_args()

    process_darkmode_screenshot(args.input, args.output, args.sensitivity)