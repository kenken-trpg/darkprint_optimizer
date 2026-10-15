import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール")
st.markdown("**ウィンドウ残り問題対応版** - 保護ロジックを調整")

# ==================== サイドバー ====================
st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", value=True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", value=True)
protect_gray = st.sidebar.toggle("🔒 グレー文字を保護", value=True)
protect_icon = st.sidebar.toggle("🖼️ アイコン・画像を保護", value=True, help="ONにするとウィンドウが残りやすくなる可能性あり")

if not use_auto:
    white_threshold = st.sidebar.slider("白抜き 閾値", 0, 80, 20, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.75, 0.05)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.15, 0.05)
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

def make_background_pure_white(image: Image.Image, threshold: int = 20, protect_gray: bool = True) -> Image.Image:
    arr = np.array(image.convert('RGB'))
    brightness = np.max(arr, axis=2)
    if protect_gray:
        mask = (brightness < threshold) & ((brightness < 30) | (brightness > 105))
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
    contrast = round(max(1.55, min(2.1, 1.6 + std / 70)), 1)
    sharpness = round(max(0.9, min(1.7, 1.0 + (image.size[0]*image.size[1])/2_300_000)), 1)
    return {"white_threshold": white_th, "contrast": contrast, "sharpness": sharpness}

# ====================== OpenCV関数 ======================
def convert_colored_text_to_black_opencv(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert('RGB'))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, (0, 48, 65), (180, 255, 245))
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 45 < area < 1400:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.2 < w/h < 9:
                cv2.drawContours(arr, [cnt], -1, (0, 0, 0), -1)
    return Image.fromarray(arr)


def protect_icons_opencv(inverted: Image.Image, original: Image.Image) -> Image.Image:
    """ウィンドウが残りにくく調整したバージョン"""
    if not protect_icon:
        return inverted
    
    inv_arr = np.array(inverted.convert('RGB'))
    orig_arr = np.array(original.convert('RGB'))
    
    bgr = cv2.cvtColor(inv_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1]
    
    _, mask = cv2.threshold(sat, 88, 255, cv2.THRESH_BINARY)
    kernel = np.ones((6,6), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2500:                    # 以前より大きめの閾値に変更（ウィンドウを保護しにくく）
            x, y, w, h = cv2.boundingRect(cnt)
            # 極端に大きい領域（画面全体に近い）は保護しない
            if area < inv_arr.shape[0] * inv_arr.shape[1] * 0.6:
                inv_arr[y:y+h, x:x+w] = orig_arr[y:y+h, x:x+w]
    
    return Image.fromarray(inv_arr)


def enhance_text(image: Image.Image, contrast: float = 1.75, sharpness: float = 1.15) -> Image.Image:
    img = ImageEnhance.Contrast(image).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.9, percent=140, threshold=3))
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
            st.warning("⚠️ ライトモード")
    
    if is_dark:
        inverted = ImageOps.invert(original)
        
        if use_auto:
            params = auto_optimize_parameters(inverted, protect_gray)
            white_threshold = params["white_threshold"]
            contrast_factor = params["contrast"]
            sharpness_factor = params["sharpness"]
        
        white_bg = make_background_pure_white(inverted, white_threshold, protect_gray)
        
        processed = white_bg
        
        if protect_icon:
            processed = protect_icons_opencv(processed, original)
        
        if convert_to_black:
            processed = convert_colored_text_to_black_opencv(processed)
        
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

st.caption("ウィンドウが残る問題に対処するため、保護の閾値を上げました。「アイコン保護」をOFFにするとより白抜きが強くなります。")