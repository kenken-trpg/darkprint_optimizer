import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール")
st.markdown("**グレー文字対応版** - 自動最適化を調整")

# ==================== サイドバー ====================
st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", value=True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", value=True)
protect_gray = st.sidebar.toggle("🔒 グレー文字を保護", value=True, 
                               help="ダークモードのグレー文字が潰れにくくなります")

if not use_auto:
    white_threshold = st.sidebar.slider("白抜き 閾値", 0, 80, 18, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.7, 0.1)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.1, 0.1)
else:
    st.sidebar.info("自動最適化をグレー文字に優しく調整中")

# ==================== 関数定義 ====================
def is_dark_mode(image: Image.Image, threshold: int = 80) -> tuple:
    # （省略 - 前回と同じ）
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
        # グレー文字保護：中間輝度（30〜100くらい）は白抜きしすぎない
        mask = (brightness < threshold) & ((brightness < 25) | (brightness > 100))
        arr[mask] = [255, 255, 255]
    else:
        arr[brightness < threshold] = [255, 255, 255]
    return Image.fromarray(arr)

def auto_optimize_parameters(image: Image.Image, protect_gray: bool = True) -> dict:
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2).flatten()
    dark_pixels = brightness[brightness < 90]
    
    if len(dark_pixels) > 0:
        percentile = 82 if protect_gray else 88   # グレー保護時は控えめ
        white_th = int(np.percentile(dark_pixels, percentile))
        white_th = max(12, min(15, white_th))    # 上限を28に下げて安全に
    else:
        white_th = 18
    
    std = np.std(brightness)
    contrast = round(max(1.5, min(2.1, 1.55 + std / 70)), 1)   # 控えめに
    sharpness = round(max(0.8, min(1.6, 0.95 + (image.size[0]*image.size[1])/2_500_000)), 1)
    
    return {"white_threshold": white_th, "contrast": contrast, "sharpness": sharpness}

# 他の関数（convert_colored_to_black, enhance_text）は前回と同じ
def convert_colored_to_black(image: Image.Image, saturation_threshold: int = 30) -> Image.Image:
    arr = np.array(image.convert('RGB'))
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    saturation = np.where(max_val == 0, 0, (max_val - min_val) / max_val * 255)
    colored_mask = saturation > saturation_threshold
    arr[colored_mask] = [0, 0, 0]
    return Image.fromarray(arr)

def enhance_text(image: Image.Image, contrast: float = 1.7, sharpness: float = 1.1) -> Image.Image:
    img = ImageEnhance.Contrast(image).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=4))  # 控えめシャープ
    return img

# ==================== 本体 ====================
uploaded_file = st.file_uploader("スクリーンショットをアップロード", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert('RGB')
    # 前半の表示部分は省略（前回と同じ）
    
    is_dark, corner_b, avg_b = is_dark_mode(original)
    
    if is_dark:
        inverted = ImageOps.invert(original)
        
        if use_auto:
            params = auto_optimize_parameters(inverted, protect_gray)
            white_threshold = params["white_threshold"]
            contrast_factor = params["contrast"]
            sharpness_factor = params["sharpness"]
            st.success(f"自動最適化（グレー保護: {'ON' if protect_gray else 'OFF'}） → 白抜き={white_threshold}")
        
        white_bg = make_background_pure_white(inverted, white_threshold, protect_gray)
        
        if convert_to_black:
            processed_img = convert_colored_to_black(white_bg)
            processed = enhance_text(processed_img, contrast_factor, sharpness_factor)
        else:
            processed = enhance_text(white_bg, contrast_factor, sharpness_factor)
        
        caption = "✅ 印刷最適化完了"
    else:
        processed = original
        caption = "そのまま印刷できます"
    
    st.subheader(caption)
    st.image(processed, use_column_width=True)
    
    # ダウンロード部分（前回と同じ）
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(label="💾 印刷用画像をダウンロード", data=buf,
                       file_name=f"{uploaded_file.name.split('.')[0]}_print_ready.png", mime="image/png")