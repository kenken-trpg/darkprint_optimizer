import streamlit as st
from PIL import Image, ImageOps, ImageStat
import numpy as np
import io

st.set_page_config(page_title="ダークモード印刷最適化", layout="wide")
st.title("🖼️ ダークモードスクリーンショット 印刷用変換ツール")
st.markdown("**自動でダークモードを検出 → ネガポジ反転**してトナー/インキを節約します")

# サイドバー設定
st.sidebar.header("設定")
threshold = st.sidebar.slider("ダークモード判定の閾値", 0, 128, 80, 1,
                            help="小さいほど厳しく判定（背景がかなり暗くないと反転しない）")

def is_dark_mode(image: Image.Image, threshold: int = 80) -> tuple:
    """背景が暗いかどうかを判定（複数方法で総合判定）"""
    # RGBに変換
    rgb_img = image.convert('RGB')
    
    # 方法1: 画像全体の平均明るさ
    stat = ImageStat.Stat(rgb_img)
    avg_brightness = sum(stat.mean) / 3
    
    # 方法2: 四隅（背景が多い領域）の平均
    width, height = rgb_img.size
    corners = [
        rgb_img.crop((0, 0, width//8, height//8)),           # 左上
        rgb_img.crop((width*7//8, 0, width, height//8)),     # 右上
        rgb_img.crop((0, height*7//8, width//8, height)),    # 左下
        rgb_img.crop((width*7//8, height*7//8, width, height))  # 右下
    ]
    
    corner_brightness = []
    for corner in corners:
        c_stat = ImageStat.Stat(corner)
        corner_brightness.append(sum(c_stat.mean) / 3)
    
    avg_corner = sum(corner_brightness) / 4
    
    # 最終判定（全体と四隅の両方を考慮）
    is_dark = avg_corner < threshold and avg_brightness < threshold + 20
    
    return is_dark, round(avg_corner, 1), round(avg_brightness, 1)

# ファイルアップローダー
uploaded_file = st.file_uploader("スクリーンショットをアップロードしてください（PNG/JPG）", 
                               type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 元画像")
        st.image(original, use_column_width=True)
    
    # 自動判定
    is_dark, corner_bright, avg_bright = is_dark_mode(original, threshold)
    
    with col2:
        st.subheader("🔍 解析結果")
        if is_dark:
            st.success(f"**ダークモードを検出しました** (背景明るさ: {corner_bright})")
            st.info("→ ネガポジ反転を適用します")
        else:
            st.warning(f"**ライトモードと判定** (背景明るさ: {corner_bright})")
        
        st.caption(f"全体平均明るさ: {avg_bright} | 四隅平均: {corner_bright}")
    
    # 変換処理
    if is_dark:
        processed = ImageOps.invert(original.convert('RGB'))
        caption = "✅ 反転後画像（印刷推奨）"
    else:
        processed = original
        caption = "そのまま印刷できます"
    
    st.subheader(caption)
    st.image(processed, use_column_width=True)
    
    # ダウンロードボタン
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="💾 変換後画像をダウンロード",
        data=buf,
        file_name=f"{uploaded_file.name.split('.')[0]}_inverted.png",
        mime="image/png",
    )
    
    # 印刷Tips
    st.info("**印刷Tips**: 反転後は白背景になるため、トナー/インキ消費が大幅に削減されます。")

else:
    st.info("👆 画像をアップロードすると自動で処理します")

st.caption("開発中：今後「明るさ微調整」「コントラスト強化」「バッチ処理」などを追加予定")