import argparse
from PIL import Image, ImageOps
import os
import sys

def convert_dark_to_light(input_path, output_path):
    try:
        # 入力ファイルの存在確認
        if not os.path.exists(input_path):
            print(f"エラー: 入力ファイル '{input_path}' が見つかりません。")
            return

        # 画像を開く
        with Image.open(input_path) as img:
            # RGBモードに変換（PNGなどのRGBA（透過）を扱うため）
            img = img.convert('RGB')
            
            # ネガポジ反転を実行
            inverted_img = ImageOps.invert(img)
            
            # 結果を保存
            inverted_img.save(output_path, quality=95)
            print(f"変換完了: {output_path}")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ダークモードの画像を印刷用に反転（ライトモード化）します。")
    parser.add_argument("input", help="入力画像ファイル名（例: screenshot.png）")
    parser.add_argument("-o", "--output", help="出力画像ファイル名（省略時は '入力名_inverted.拡張子'）")

    args = parser.parse_args()

    input_file = args.input
    
    # 出力ファイル名が指定されていない場合のデフォルト設定
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_inverted{ext}"

    convert_dark_to_light(input_file, output_file)