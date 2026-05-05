import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール v2")
st.markdown("**精度改善版：Luma輝度・エッジ検出・二値化対応**")

# ==================== サイドバー ====================

st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", value=True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", value=True)
protect_gray = st.sidebar.toggle("🔒 グレー文字保護", value=True)
use_binary = st.sidebar.toggle("🧾 テキスト最適化（二値化）", value=False)

if not use_auto:
    white_threshold = st.sidebar.slider("白抜き閾値", 0, 40, 18, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.7, 0.1)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.1, 0.1)
else:
    st.sidebar.info("自動最適化中")

# ==================== 基本関数 ====================

def get_luminance(arr):
    return (0.2126 * arr[:,:,0] + 0.7152 * arr[:,:,1] + 0.0722 * arr[:,:,2])

def is_gray_pixel(arr, tolerance=10):
    return (
        (np.abs(arr[:,:,0] - arr[:,:,1]) < tolerance) &
        (np.abs(arr[:,:,1] - arr[:,:,2]) < tolerance)
    )

def is_dark_mode(image: Image.Image, threshold: int = 80):
    rgb = image.convert('RGB')
    stat = ImageStat.Stat(rgb)
    avg = sum(stat.mean) / 3

    w, h = rgb.size
    corners = [rgb.crop((x, y, x+w//8, y+h//8))
               for x, y in [(0,0),(w*7//8,0),(0,h*7//8),(w*7//8,h*7//8)]]

    corner_avg = sum(sum(ImageStat.Stat(c).mean)/3 for c in corners) / 4
    is_dark = corner_avg < threshold and avg < threshold + 20

    return is_dark, round(corner_avg,1), round(avg,1)

# ==================== 処理 ====================

def make_background_pure_white(arr, threshold=18, protect_gray=True):
    brightness = get_luminance(arr)

    if protect_gray:
        gray_mask = is_gray_pixel(arr)
        mask = (brightness < threshold) & (~gray_mask)
    else:
        mask = (brightness < threshold)

    arr[mask] = [255,255,255]
    return arr

def detect_text_edges(arr):
    gray = np.mean(arr, axis=2).astype(np.float32)
    gx = np.abs(np.gradient(gray, axis=0))
    gy = np.abs(np.gradient(gray, axis=1))
    edges = gx + gy
    return edges > 12

def convert_colored_to_black(arr, saturation_threshold=35):
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    brightness = get_luminance(arr)

    saturation = np.where(max_val == 0, 0, (max_val - min_val) / max_val * 255)
    text_like = detect_text_edges(arr)

    mask = (
        (saturation > saturation_threshold) &
        (brightness > 70) &
        (brightness < 220) &
        text_like
    )

    arr[mask] = [0,0,0]
    return arr

def enhance_text(image, contrast=1.7, sharpness=1.1):
    # 順序改善版
    image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=4))
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    return image

def binarize(image, threshold=180):
    gray = image.convert('L')
    return gray.point(lambda x: 0 if x < threshold else 255, '1')

def auto_optimize_parameters(arr):
    brightness = get_luminance(arr).flatten()
    dark_pixels = brightness[brightness < 100]

    if len(dark_pixels) > 0:
        white_th = int(np.percentile(dark_pixels, 85))
        white_th = max(10, min(25, white_th))
    else:
        white_th = 15

    std = np.std(brightness)
    contrast = round(max(1.5, min(2.2, 1.6 + std / 70)), 1)
    sharpness = round(max(0.9, min(1.7, 1.0 + (arr.shape[0]*arr.shape[1])/2_500_000)), 1)

    return white_th, contrast, sharpness

# ==================== パイプライン ====================

def pipeline(image, config):
    arr = np.array(image.convert('RGB'))

    if config["invert"]:
        arr = 255 - arr

    if config["auto"]:
        wt, ct, sp = auto_optimize_parameters(arr)
    else:
        wt, ct, sp = config["white_th"], config["contrast"], config["sharp"]

    arr = make_background_pure_white(arr, wt, config["protect_gray"])

    if config["black_text"]:
        arr = convert_colored_to_black(arr)

    img = Image.fromarray(arr)
    img = enhance_text(img, ct, sp)

    if config["binary"]:
        img = binarize(img)

    return img, wt, ct, sp

# ==================== UI ====================

uploaded_file = st.file_uploader("スクリーンショットをアップロード",
                                type=["png","jpg","jpeg","webp"])

if uploaded_file:
    original = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 元画像")
        st.image(original, use_container_width=True)

    is_dark, corner_b, avg_b = is_dark_mode(original)

    with col2:
        st.subheader("🔍 解析結果")
        if is_dark:
            st.success(f"ダークモード検出（四隅:{corner_b}）")
        else:
            st.warning("ライトモード")

    if is_dark:
        config = {
            "invert": True,
            "auto": use_auto,
            "white_th": white_threshold if not use_auto else None,
            "contrast": contrast_factor if not use_auto else None,
            "sharp": sharpness_factor if not use_auto else None,
            "protect_gray": protect_gray,
            "black_text": convert_to_black,
            "binary": use_binary
        }

        processed, wt, ct, sp = pipeline(original, config)

        st.success(f"最適化完了（白抜き:{wt} / コントラスト:{ct} / シャープ:{sp}）")
    else:
        processed = original

    st.subheader("🖨️ 出力")
    st.image(processed, use_container_width=True)

    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        "💾 ダウンロード",
        data=buf,
        file_name="print_ready.png",
        mime="image/png"
    )

else:
    st.info("画像をアップロードしてください")

st.caption("v2: Luma輝度 / エッジ検出 / グレー保護 / 二値化対応")