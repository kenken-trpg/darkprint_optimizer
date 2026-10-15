import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモード印刷最適化ツール")
st.caption("ダークモードのスクリーンショットを、印刷しやすい白背景画像へ変換します。")

# =========================
# 設定
# =========================
st.sidebar.header("変換設定")

use_auto = st.sidebar.toggle("🚀 自動パラメータ最適化", value=True)
convert_to_black = st.sidebar.toggle("⚫ 色付きテキストを黒に統一", value=True)
protect_gray = st.sidebar.toggle("🔒 グレー文字を保護", value=True)

if use_auto:
    st.sidebar.info("自動最適化を使用します。")
    white_threshold = 12
    contrast_factor = 1.7
    sharpness_factor = 1.1
else:
    white_threshold = st.sidebar.slider("白抜き閾値", 0, 60, 18, 1)
    contrast_factor = st.sidebar.slider("コントラスト", 1.0, 3.0, 1.7, 0.1)
    sharpness_factor = st.sidebar.slider("シャープネス", 0.0, 3.0, 1.1, 0.1)

# =========================
# ヘルパー
# =========================
def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        return image.convert("RGBA")
    return image.convert("RGB")

def preserve_alpha_invert(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        rgb = image.convert("RGB")
        inv = ImageOps.invert(rgb)
        inv.putalpha(alpha)
        return inv
    if image.mode in ("L", "RGB"):
        return ImageOps.invert(image)
    return ImageOps.invert(image.convert("RGB"))

def analyze_image(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    h, w = arr.shape[:2]

    brightness = arr.max(axis=2)
    mean_brightness = float(brightness.mean())

    c = max(1, min(h, w) // 8)
    corners = [
        arr[0:c, 0:c],
        arr[0:c, w-c:w],
        arr[h-c:h, 0:c],
        arr[h-c:h, w-c:w],
    ]
    corner_means = [float(x.max(axis=2).mean()) for x in corners]
    corner_brightness = float(np.mean(corner_means))

    return {
        "mean_brightness": round(mean_brightness, 1),
        "corner_brightness": round(corner_brightness, 1),
        "is_dark": corner_brightness < 80 and mean_brightness < 100,
    }

def auto_parameters(image: Image.Image, protect_gray: bool = True) -> dict:
    arr = np.array(image.convert("RGB"))
    brightness = arr.max(axis=2).flatten()
    dark_pixels = brightness[brightness < 120]

    if len(dark_pixels) == 0:
        white_threshold = 12
    else:
        p = 78 if protect_gray else 86
        white_threshold = int(np.percentile(dark_pixels, p))
        white_threshold = max(8, min(12, white_threshold))

    std = float(np.std(brightness))
    contrast = round(max(1.3, min(2.2, 1.45 + std / 90.0)), 1)

    pixels = image.size[0] * image.size[1]
    sharpness = round(max(0.8, min(1.6, 0.95 + pixels / 3_000_000)), 1)

    return {
        "white_threshold": white_threshold,
        "contrast": contrast,
        "sharpness": sharpness,
    }

def whiten_background(image: Image.Image, threshold: int = 18, protect_gray: bool = True) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.array(rgb)

    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    sat = max_c - min_c

    if protect_gray:
        mask = (max_c < threshold) & ((sat < 16) | (max_c < 28))
    else:
        mask = max_c < threshold

    arr[mask] = [255, 255, 255]
    return Image.fromarray(arr)

def blacken_colored_text(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.array(rgb)

    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    sat = np.where(max_c == 0, 0, (max_c - min_c) / max_c * 255)

    text_like = (sat > 38) & (max_c > 40) & (max_c < 235)
    arr[text_like] = [0, 0, 0]

    return Image.fromarray(arr)

def enhance_printability(image: Image.Image, contrast: float, sharpness: float) -> Image.Image:
    img = ImageEnhance.Contrast(image).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=4))
    return img

def process_image(original: Image.Image, use_auto: bool, white_threshold: int,
                  contrast_factor: float, sharpness_factor: float, protect_gray: bool,
                  convert_to_black: bool) -> tuple[Image.Image, dict]:
    analysis = analyze_image(original)

    if not analysis["is_dark"]:
        return original, {"analysis": analysis, "applied": False}

    inverted = preserve_alpha_invert(original)

    if use_auto:
        params = auto_parameters(inverted, protect_gray=protect_gray)
        white_threshold = params["white_threshold"]
        contrast_factor = params["contrast"]
        sharpness_factor = params["sharpness"]
    else:
        params = {
            "white_threshold": white_threshold,
            "contrast": contrast_factor,
            "sharpness": sharpness_factor,
        }

    processed = whiten_background(inverted, threshold=white_threshold, protect_gray=protect_gray)

    if convert_to_black:
        processed = blacken_colored_text(processed)

    processed = enhance_printability(processed, contrast_factor, sharpness_factor)

    return processed, {"analysis": analysis, "applied": True, "params": params}

def to_png_bytes(image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    if image.mode == "RGBA":
        image.save(buf, format="PNG")
    else:
        image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf

# =========================
# UI
# =========================
uploaded_file = st.file_uploader(
    "スクリーンショットをアップロード",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is None:
    st.info("👆 画像をアップロードしてください。")
    st.stop()

original = Image.open(uploaded_file)
processed, meta = process_image(
    original=original,
    use_auto=use_auto,
    white_threshold=white_threshold,
    contrast_factor=contrast_factor,
    sharpness_factor=sharpness_factor,
    protect_gray=protect_gray,
    convert_to_black=convert_to_black,
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 元画像")
    st.image(original, use_container_width=True)

with col2:
    st.subheader("🔍 解析結果")
    a = meta["analysis"]
    if a["is_dark"]:
        st.success(f"ダークモード検出。四隅: {a['corner_brightness']} / 平均: {a['mean_brightness']}")
    else:
        st.warning(f"ライトモード判定。四隅: {a['corner_brightness']} / 平均: {a['mean_brightness']}")

if meta["applied"]:
    st.subheader("✅ 印刷用プレビュー")
    st.image(processed, use_container_width=True)

    if "params" in meta:
        p = meta["params"]
        st.write(
            f"使用パラメータ: 白抜き={p['white_threshold']}, "
            f"コントラスト={p['contrast']}, シャープネス={p['sharpness']}"
        )

    buf = to_png_bytes(processed)
    base = uploaded_file.name.rsplit(".", 1)[0]

    st.download_button(
        label="💾 印刷用画像をダウンロード",
        data=buf,
        file_name=f"{base}_print_ready.png",
        mime="image/png",
    )
else:
    st.subheader("ℹ️ 変換なし")
    st.image(original, use_container_width=True)