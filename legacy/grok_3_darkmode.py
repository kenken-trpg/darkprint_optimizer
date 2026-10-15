import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール")
st.markdown("**自動検出 → 反転 → 白抜き → 文字強調 → 色付きテキスト黒一色化**")

# ==================== サイドバー ====================
st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", value=True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", value=True,
                                   help="カラーテキストやアイコンを黒文字に変換（印刷で最も綺麗）")

if not use_auto:
    threshold_dark = st.sidebar.slider("ダークモード判定", 0, 128, 80, 1)
    white_threshold = st.sidebar.slider("白抜き 閾値", 0, 80, 15, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.8, 0.1)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.2, 0.1)

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

def make_background_pure_white(image: Image.Image, threshold: int = 25) -> Image.Image:
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2)
    arr[brightness < threshold] = [255, 255, 255]
    return Image.fromarray(arr)

def convert_colored_to_black(image: Image.Image, saturation_threshold: int = 30) -> Image.Image:
    """色付き部分を黒に統一"""
    arr = np.array(image.convert('RGB'))
    
    # HSV風に簡易計算
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    max_val = np.max(arr, axis=2)
    min_val = np.min(arr, axis=2)
    saturation = np.where(max_val == 0, 0, (max_val - min_val) / max_val * 255)
    
    # 彩度が高いピクセル（色付き）を黒に
    colored_mask = saturation > saturation_threshold
    arr[colored_mask] = [0, 0, 0]
    
    return Image.fromarray(arr)

def enhance_text(image: Image.Image, contrast: float = 1.8, sharpness: float = 1.2) -> Image.Image:
    img = ImageEnhance.Contrast(image).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))
    return img

def auto_optimize_parameters(image: Image.Image) -> dict:
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2).flatten()
    dark_pixels = brightness[brightness < 100]
    
    white_th = int(np.percentile(dark_pixels, 88)) if len(dark_pixels) > 0 else 25
    white_th = max(12, min(24, white_th))
    
    std = np.std(brightness)
    contrast = round(max(1.5, min(2.4, 1.6 + std / 60)), 1)
    sharpness = round(max(0.8, min(2.0, 1.1 + (image.size[0]*image.size[1])/2_000_000)), 1)
    
    return {"white_threshold": white_th, "contrast": contrast, "sharpness": sharpness}

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
            params = auto_optimize_parameters(inverted)
            white_threshold = params["white_threshold"]
            contrast_factor = params["contrast"]
            sharpness_factor = params["sharpness"]
            st.success(f"自動最適化適用: 白抜き={white_threshold} | コントラスト={contrast_factor} | シャープ={sharpness_factor}")
        
        # 処理パイプライン
        white_bg = make_background_pure_white(inverted, white_threshold)
        
        if convert_to_black:
            black_text = convert_colored_to_black(white_bg)
            processed = enhance_text(black_text, contrast_factor, sharpness_factor)
            st.info("⚫ 色付きテキストを黒に統一しました")
        else:
            processed = enhance_text(white_bg, contrast_factor, sharpness_factor)
        
        caption = "✅ 印刷最適化完了（白背景・黒文字）"
    else:
        processed = original
        caption = "そのまま印刷できます"
    
    st.subheader(caption)
    st.image(processed, use_column_width=True)
    
    # ダウンロード
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

st.caption("このツールでダークモード画像を印刷用に最適化すると、トナー/インキ消費が大幅に削減されます。")