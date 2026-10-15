# DarkPrint Optimizer

ダークモードのスクリーンショットを、色味を保ったまま白背景の印刷向け画像に変換するツールです。
黒背景のまま印刷するとインクを大量に消費するうえ読みにくいため、ライトモード相当の見た目に変換します。

## 特徴

- **色相を保った反転** — Lab 色空間で明度だけを反転するため、青いリンクは青のまま、緑のバッジは緑のまま残ります（単純な RGB 反転のように反対色になりません）
- **背景を純白に** — 最頻の明度を背景とみなしてレベル補正し、背景付近の淡い色味も除去します
- **写真・アイコンを保護** — 画像領域を検出し、反転せずに元のまま残します
- **ダークモード自動判定** — ライトモードの画像は変換せずにスキップします
- **白黒出力** — 色を使わずに印刷したい場合に

## セットアップ

```sh
pip install -r requirements.txt
```

## 使い方

### Web UI（Streamlit）

```sh
streamlit run app.py
```

ブラウザで画像をアップロードすると、変換前後を並べて表示します。複数枚をまとめて変換し、zip で一括ダウンロードできます。

| 設定 | 内容 |
|---|---|
| 写真・アイコンを反転しない | 画像領域を検出して元のまま残す（既定: オン） |
| 白黒で出力 | グレースケールで出力 |
| 文字色の鮮やかさ | 色付き文字の彩度倍率（0 で白黒相当） |
| 判定に関わらず反転する | ダークモードなのに誤ってスキップされた画像用。白背景の画像に使うと黒背景になるので注意 |
| 保護した領域を表示 | 画像として保護した範囲を赤くハイライト（確認用） |

### コマンドライン

```sh
python darkprint.py screenshot.png                 # → screenshot_print.png
python darkprint.py *.png -o out/                  # まとめて out/ に出力
python darkprint.py screenshot.png --gray          # 白黒で出力
python darkprint.py screenshot.png --no-protect    # 画像領域も反転
python darkprint.py screenshot.png --force         # 判定を無視して反転（ダークなのにスキップされた時用）
```

実行時に、変換前後のインク使用量の目安（平均の暗さ）と、保護した画像領域の割合を表示します。

### Python から使う

```python
import cv2
from darkprint import Options, convert

res = convert(cv2.imread("screenshot.png"), Options(grayscale=True))
cv2.imwrite("out.png", res.image)
```

## 仕組み

1. 画像を Lab 色空間に変換し、最頻の明度から背景色を推定
2. 背景と外周がどちらも暗ければダークモードと判定
3. 明度 L だけを `100 - L` で反転し、背景 → 白・最暗部 → 黒 にレベル補正
4. 背景付近の明度では色味 (a/b) を徐々に 0 にして紙の白を残す
5. 背景と異なる画素の塊ごとに「背景色の割合」「色数」「中間調の割合」を調べ、写真らしい矩形は元画像のまま戻す

## ファイル構成

| パス | 内容 |
|---|---|
| `darkprint.py` | 変換処理の本体と CLI |
| `app.py` | Streamlit UI |
| `legacy/` | 以前の実装（各種 AI に生成させた比較用の試作。メンテナンス対象外。依存は `legacy/requirements.txt`） |

## ライセンス

Apache License 2.0 — [LICENSE](LICENSE) を参照してください。
