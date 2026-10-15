import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール")
st.markdown("**グレー文字対応 + 色統一改善版**")

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
    st.sidebar.info("自動最適化中（白抜き上限15固定）")

# ==================== 関数定義 ====================
def is_dark_mode(image: Image.Image, threshold: int = 80) -> tuple:
    rgb = image.convert('RGB')
    stat = ImageStat.Stat(rgb)
    avg = sum(stat.mean) / 3
    w, h = rgb.size
    corners = [rgb.crop((x, y, x+w//8, y+h//8)) 
               for x, y in [(0,0),(w*7//8,0),(0,h*7//8),(w*7//8,h*7//8)]]
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
        white_th = max(12, min(15, white_th))   # ← 上限を15に固定
    else:
        white_th = 15
    
    std = np.std(brightness)
    contrast = round(max(1.5, min(2.1, 1.55 + std / 75)), 1)
    sharpness = round(max(0.8, min(1.65, 0.95 + (image.size[0]*image.size[1])/2_500_000)), 1)
    
    return {"white_threshold": white_th, "contrast": contrast, "sharpness": sharpness}

def convert_colored_to_black(image: Image.Image, saturation_threshold: int = 35) -> Image.Image:
    """改善版：色付きテキストのみを黒に統一（背景が黒くなる問題を解決）"""
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2)
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    
    saturation = np.where(max_val == 0, 0, (max_val - min_val) / max_val * 255)
    
    # 重要な変更点：
    # ① 彩度が高く、かつ「ある程度明るい」部分だけを対象（背景の黒化防止）
    # ② 白背景付近は対象から外す
    colored_mask = (saturation > saturation_threshold) & (brightness > 60) & (brightness < 240)
    
    arr[colored_mask] = [0, 0, 0]
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
            st.success(f"✅ ダークモード検出（四隅明るさ: {corner_b}）")
        else:
            st.warning("⚠️ ライトモードと判定")
    
    if is_dark:
        inverted = ImageOps.invert(original)
        
        if use_auto:
            params = auto_optimize_parameters(inverted, protect_gray)
            white_threshold = params["white_threshold"]
            contrast_factor = params["contrast"]
            sharpness_factor = params["sharpness"]
            st.success(f"🚀 自動最適化適用 → 白抜き={white_threshold}（上限15固定）")
        
        white_bg = make_background_pure_white(inverted, white_threshold, protect_gray)
        
        if convert_to_black:
            processed_img = convert_colored_to_black(white_bg)
            processed = enhance_text(processed_img, contrast_factor, sharpness_factor)
            st.info("⚫ 色付きテキストを黒に統一しました（背景黒化対策済）")
        else:
            processed = enhance_text(white_bg, contrast_factor, sharpness_factor)
        
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

st.caption("自動最適化の白抜きは上限15固定、色統一時の背景黒化問題も改善しました。")