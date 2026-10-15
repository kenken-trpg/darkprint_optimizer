import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール v3")
st.markdown("**画像保護対応：文字と画像を分離して処理**")

# ==================== サイドバー ====================

st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", True)
protect_gray = st.sidebar.toggle("🔒 グレー文字保護", True)
use_binary = st.sidebar.toggle("🧾 二値化モード", False)
protect_images = st.sidebar.toggle("🖼️ 画像を保護（反転しない）", True)

if not use_auto:
    white_threshold = st.sidebar.slider("白抜き閾値", 0, 40, 18, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.7, 0.1)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.1, 0.1)

# ==================== 基本関数 ====================

def get_luminance(arr):
    return (0.2126 * arr[:,:,0] + 0.7152 * arr[:,:,1] + 0.0722 * arr[:,:,2])

def is_gray_pixel(arr, tol=10):
    return (
        (np.abs(arr[:,:,0]-arr[:,:,1])<tol) &
        (np.abs(arr[:,:,1]-arr[:,:,2])<tol)
    )

# ==================== 画像領域検出（重要） ====================

def detect_image_regions(arr, block_size=32):
    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=bool)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:y+block_size, x:x+block_size]

            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue

            var = np.var(block)
            unique_ratio = len(np.unique(block.reshape(-1,3), axis=0)) / (block_size*block_size)

            # ヒューリスティック
            is_image = (var > 500) and (unique_ratio > 0.08)

            if is_image:
                mask[y:y+block_size, x:x+block_size] = True

    return mask

# ==================== 処理 ====================

def make_background_pure_white(arr, threshold, protect_gray):
    brightness = get_luminance(arr)

    if protect_gray:
        gray_mask = is_gray_pixel(arr)
        mask = (brightness < threshold) & (~gray_mask)
    else:
        mask = (brightness < threshold)

    arr[mask] = [255,255,255]
    return arr

def detect_text_edges(arr):
    gray = np.mean(arr, axis=2)
    gx = np.abs(np.gradient(gray, axis=0))
    gy = np.abs(np.gradient(gray, axis=1))
    return (gx + gy) > 12

def convert_colored_to_black(arr):
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    brightness = get_luminance(arr)

    saturation = np.where(max_val==0,0,(max_val-min_val)/max_val*255)
    edges = detect_text_edges(arr)

    mask = (saturation > 35) & (brightness > 70) & (brightness < 220) & edges
    arr[mask] = [0,0,0]
    return arr

def enhance_text(img, contrast, sharpness):
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=4))
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def binarize(img):
    return img.convert('L').point(lambda x: 0 if x < 180 else 255, '1')

def auto_optimize(arr):
    brightness = get_luminance(arr).flatten()
    dark = brightness[brightness < 100]

    if len(dark) > 0:
        wt = int(np.percentile(dark, 85))
        wt = max(10, min(25, wt))
    else:
        wt = 15

    std = np.std(brightness)
    ct = round(max(1.5, min(2.2, 1.6 + std/70)),1)
    sp = round(max(0.9, min(1.7, 1.0 + arr.size/3_000_000)),1)

    return wt, ct, sp

# ==================== パイプライン ====================

def pipeline(image, config):
    arr = np.array(image.convert('RGB'))

    # 画像領域検出
    image_mask = detect_image_regions(arr) if config["protect_images"] else None

    # 反転（画像は除外）
    if config["invert"]:
        inv = 255 - arr
        if image_mask is not None:
            arr[~image_mask] = inv[~image_mask]
        else:
            arr = inv

    # パラメータ
    if config["auto"]:
        wt, ct, sp = auto_optimize(arr)
    else:
        wt, ct, sp = config["white"], config["contrast"], config["sharp"]

    # 背景白化（画像除外）
    if image_mask is not None:
        temp = arr.copy()
        temp = make_background_pure_white(temp, wt, config["gray"])
        arr[~image_mask] = temp[~image_mask]
    else:
        arr = make_background_pure_white(arr, wt, config["gray"])

    # 色→黒（画像除外）
    if config["black"]:
        if image_mask is not None:
            temp = arr.copy()
            temp = convert_colored_to_black(temp)
            arr[~image_mask] = temp[~image_mask]
        else:
            arr = convert_colored_to_black(arr)

    img = Image.fromarray(arr)
    img = enhance_text(img, ct, sp)

    if config["binary"]:
        img = binarize(img)

    return img, wt, ct, sp

# ==================== UI ====================

uploaded = st.file_uploader("画像アップロード", type=["png","jpg","jpeg","webp"])

if uploaded:
    original = Image.open(uploaded)

    st.image(original, caption="元画像", use_column_width=True)

    config = {
        "invert": True,
        "auto": use_auto,
        "white": white_threshold if not use_auto else None,
        "contrast": contrast_factor if not use_auto else None,
        "sharp": sharpness_factor if not use_auto else None,
        "gray": protect_gray,
        "black": convert_to_black,
        "binary": use_binary,
        "protect_images": protect_images
    }

    result, wt, ct, sp = pipeline(original, config)

    st.image(result, caption=f"出力（白:{wt} コントラスト:{ct}）", use_column_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    st.download_button("💾 ダウンロード", buf, "optimized.png")

else:
    st.info("画像をアップロードしてください")