import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール")
st.markdown("**色統一＋アイコン保護 改善版**（領域サイズ考慮）")

# ==================== サイドバー ====================
st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", value=True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", value=True)
protect_gray = st.sidebar.toggle("🔒 グレー文字を保護", value=True)
protect_icon = st.sidebar.toggle("🖼️ アイコン・画像を保護", value=True)

if not use_auto:
    white_threshold = st.sidebar.slider("白抜き 閾値", 0, 80, 18, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.7, 0.1)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.1, 0.1)
else:
    st.sidebar.info("自動最適化中（白抜き上限15固定）")

# ==================== 関数定義 ====================
def is_dark_mode(image: Image.Image, threshold: int = 80) -> tuple:
    rgb = image.convert('RGB')
    stat = ImageStat.Stat(rgb)
    avg = sum(stat.mean) / 3
    w, h = rgb.size
    corners = [rgb.crop((x, y, x+w//8, y+h//8)) for x, y in [(0,0),(w*7//8,0),(0,h*7//8),(w*7//8,h*7//8)]]
    corner_avg = sum(sum(ImageStat.Stat(c).mean)/3 for c in corners) / 4
    is_dark = corner_avg < threshold and avg < threshold + 20
    return is_dark, round(corner_avg, 1), round(avg, 1)

def make_background_pure_white(image: Image.Image, threshold: int = 18, protect_gray: bool = True) -> Image.Image:
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2)
    if protect_gray:
        mask = (brightness < threshold) & ((brightness < 28) | (brightness > 110))
        arr[mask] = [255, 255, 255]
    else:
        arr[brightness < threshold] = [255, 255, 255]
    return Image.fromarray(arr)

def auto_optimize_parameters(image: Image.Image, protect_gray: bool = True) -> dict:
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2).flatten()
    dark_pixels = brightness[brightness < 90]
    if len(dark_pixels) > 0:
        percentile = 82 if protect_gray else 88
        white_th = int(np.percentile(dark_pixels, percentile))
        white_th = max(12, min(15, white_th))
    else:
        white_th = 15
    std = np.std(brightness)
    contrast = round(max(1.5, min(2.1, 1.55 + std / 75)), 1)
    sharpness = round(max(0.8, min(1.65, 0.95 + (image.size[0]*image.size[1])/2_500_000)), 1)
    return {"white_threshold": white_th, "contrast": contrast, "sharpness": sharpness}

# ====================== 新しい判定関数 ======================
def convert_colored_text_to_black(image: Image.Image, min_size: int = 80, max_size: int = 800) -> Image.Image:
    """改善版：領域サイズを考慮して色付きテキストのみ黒に統一"""
    arr = np.array(image.convert('RGB'))
    orig_arr = arr.copy()
    
    # 彩度計算
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    saturation = np.where(max_val == 0, 0, (max_val - min_val) / max_val * 255)
    brightness = max_val
    
    # 彩度が高めのピクセル
    candidate_mask = (saturation > 38) & (brightness > 70) & (brightness < 245)
    
    if not np.any(candidate_mask):
        return Image.fromarray(arr)
    
    # 簡易的な連結成分サイズフィルタ（小さい領域＝文字と判定）
    # 大きな塊（アイコン）は残す
    from scipy import ndimage
    labeled, num_features = ndimage.label(candidate_mask)
    
    for label in range(1, num_features + 1):
        component = (labeled == label)
        size = np.sum(component)
        
        if min_size <= size <= max_size:   # 文字サイズの範囲
            arr[component] = [0, 0, 0]
    
    return Image.fromarray(arr)

def protect_icons(image: Image.Image, original: Image.Image) -> Image.Image:
    """アイコン保護（大きい彩度領域を元画像に戻す）"""
    if not protect_icon:
        return image
    
    arr = np.array(image.convert('RGB'))
    orig_arr = np.array(original.convert('RGB'))
    
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    saturation = np.where(max_val == 0, 0, (max_val - min_val) / max_val)
    
    # 大きい彩度領域をアイコンとみなして元画像に戻す
    icon_mask = saturation > 0.35
    
    # 簡易サイズフィルタ（大きい領域のみ）
    from scipy import ndimage
    labeled, num = ndimage.label(icon_mask)
    for i in range(1, num+1):
        if np.sum(labeled == i) > 1200:   # 大きい領域は保護
            arr[labeled == i] = orig_arr[labeled == i]
    
    return Image.fromarray(arr)

def enhance_text(image: Image.Image, contrast: float = 1.7, sharpness: float = 1.1) -> Image.Image:
    img = ImageEnhance.Contrast(image).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=4))
    return img

# ==================== 本体 ====================
uploaded_file = st.file_uploader("スクリーンショットをアップロード", 
                               type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 元画像")
        st.image(original, use_column_width=True)
    
    is_dark, corner_b, avg_b = is_dark_mode(original)
    
    with col2:
        st.subheader("🔍 解析結果")
        if is_dark:
            st.success(f"✅ ダークモード検出")
        else:
            st.warning("⚠️ ライトモードと判定")
    
    if is_dark:
        inverted = ImageOps.invert(original)
        
        if use_auto:
            params = auto_optimize_parameters(inverted, protect_gray)
            white_threshold = params["white_threshold"]
            contrast_factor = params["contrast"]
            sharpness_factor = params["sharpness"]
        
        white_bg = make_background_pure_white(inverted, white_threshold, protect_gray)
        
        # 処理順序を改善
        processed = white_bg
        
        if protect_icon:
            processed = protect_icons(processed, original)
        
        if convert_to_black:
            processed = convert_colored_text_to_black(processed)
            st.info("⚫ 色付きテキストを黒に統一（領域サイズ考慮）")
        
        processed = enhance_text(processed, contrast_factor, sharpness_factor)
        
        caption = "✅ 印刷最適化完了"
    else:
        processed = original
        caption = "そのまま印刷できます"
    
    st.subheader(caption)
    st.image(processed, use_column_width=True)
    
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="💾 印刷用画像をダウンロード",
        data=buf,
        file_name=f"{uploaded_file.name.split('.')[0]}_print_ready.png",
        mime="image/png",
    )

else:
    st.info("👆 画像をアップロードしてください")