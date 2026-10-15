"""DarkPrint Optimizer — Streamlit UI。

起動: streamlit run app.py
"""
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from darkprint import Options, convert, ink_coverage, output_name

st.set_page_config(page_title="DarkPrint Optimizer", page_icon="🖨️", layout="wide")
st.title("🖨️ DarkPrint Optimizer")
st.caption("ダークモードのスクリーンショットを、色味を保ったまま白背景の印刷向け画像に変換します")

# ==================== サイドバー ====================

st.sidebar.header("変換設定")
protect = st.sidebar.toggle("🖼️ 写真・アイコンを反転しない", value=True)
grayscale = st.sidebar.toggle("⚫ 白黒で出力", value=False)
chroma = st.sidebar.slider("文字色の鮮やかさ", 0.0, 1.5, 1.0, 0.1, disabled=grayscale)
force = st.sidebar.toggle("判定に関わらず反転する", value=False,
                          help="ダークモードなのにスキップされた画像用。白背景の画像に使うと黒背景になります")
show_mask = st.sidebar.toggle("保護した領域を表示", value=False)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png(bgr: np.ndarray) -> bytes:
    return cv2.imencode(".png", bgr)[1].tobytes()


@st.cache_data(max_entries=64, show_spinner="変換中…")
def process(data: bytes, protect: bool, grayscale: bool, chroma: float, force: bool):
    """設定変更やダウンロードのたびに再変換しないようキャッシュする。"""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    opt = Options(protect_photos=protect, grayscale=grayscale, chroma=chroma, force=force)
    return bgr, convert(bgr, opt)


# ==================== 本体 ====================

files = st.file_uploader("スクリーンショット（複数可）", type=["png", "jpg", "jpeg", "webp"],
                         accept_multiple_files=True)
if not files:
    st.info("画像をアップロードしてください")
    st.stop()

outputs: list[tuple[str, bytes]] = []
used_names: set[str] = set()
for f in files:
    processed = process(f.getvalue(), protect, grayscale, chroma, force)
    if processed is None:
        st.error(f"読み込めません: {f.name}")
        continue
    bgr, res = processed

    st.subheader(f.name)
    if not (res.is_dark or force):
        st.warning("ライトモードの画像と判定したため変換していません"
                   "（ダークモードの画像なら「判定に関わらず反転する」をオンに）")
        continue

    shown = res.image.copy()
    if show_mask and res.photo_mask.any():
        shown[res.photo_mask] = (shown[res.photo_mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    col1, col2 = st.columns(2)
    col1.image(to_rgb(bgr), caption="元画像")
    col2.image(to_rgb(shown), caption="印刷用")
    st.caption(f"インク使用量の目安: {ink_coverage(bgr):.1f}% → {ink_coverage(res.image):.1f}%"
               f"　／　画像保護: {res.photo_mask.mean() * 100:.1f}%")

    name = output_name(Path(f.name).stem, used_names)
    data = encode_png(res.image)
    outputs.append((name, data))
    st.download_button("💾 ダウンロード", data, name, "image/png", key=f"dl-{f.file_id}")
    st.divider()

if len(outputs) > 1:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in outputs:
            z.writestr(name, data)
    st.sidebar.download_button("📦 まとめてダウンロード (zip)", buf.getvalue(),
                               "darkprint.zip", "application/zip")
