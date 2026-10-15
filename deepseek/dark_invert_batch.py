import argparse, os, cv2
from invert_utils import generate_inversion_mask, apply_invert

MODEL_NAME = 'dark_invert_rf.pkl'
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), MODEL_NAME)

def main():
    parser = argparse.ArgumentParser(description="Dark mode screenshot inverter (batch)")
    parser.add_argument("input_dir", help="入力フォルダ")
    parser.add_argument("-o", "--output_dir", default="inverted", help="出力フォルダ")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="モデルファイル")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("モデルファイルがありません:", args.model)
        return

    exts = ('.png','.jpg','.jpeg','.bmp','.tiff')
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    if not files:
        print("画像がありません")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(files)} 枚を処理します...")
    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"スキップ: {fname}")
            continue
        try:
            mask = generate_inversion_mask(img, args.model)
            result = apply_invert(img.copy(), mask)
            out_name = os.path.splitext(fname)[0] + "_inverted.png"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, result)
            print(f"変換: {fname} -> {out_path}")
        except Exception as e:
            print(f"エラー: {fname} - {e}")
    print("完了")

if __name__ == "__main__":
    main()