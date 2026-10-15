import streamlit as st
from PIL import Image, ImageOps
import io

# ページ設定
st.set_page_config(
    page_title="DarkPrint Optimizer - Streamlit",
    page_icon="🌓",
    layout="centered"
)

# カスタムCSSでデザインを洗練
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    .upload-text {
        text-align: center;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌓 DarkPrint Optimizer")
st.subheader("ダークモード画像を印刷用に反転（ライトモード化）")

st.markdown("""
ダークモードのスクリーンショットをそのまま印刷するとインクを大量に消費します。
このツールは色を反転させ、白背景の読みやすい画像に変換します。
""")

# ファイルアップローダー
uploaded_file = st.file_uploader("画像を選択してください (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file)
    
    # レイアウト作成
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 変換前")
        st.image(image, use_container_width=True)
    
    # 変換処理
    with st.spinner('処理中...'):
        # RGBモードに変換
        rgb_image = image.convert('RGB')
        # 反転処理
        inverted_image = ImageOps.invert(rgb_image)
        
        # メモリ上に保存
        buf = io.BytesIO()
        inverted_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

    with col2:
        st.write("### 変換後")
        st.image(inverted_image, use_container_width=True)
    
    # ダウンロードボタン
    st.divider()
    file_name = f"inverted_{uploaded_file.name}"
    st.download_button(
        label="✨ 変換した画像をダウンロード",
        data=byte_im,
        file_name=file_name,
        mime="image/png"
    )

st.divider()
st.caption("Powered by DarkPrint Optimizer Core Logic")
