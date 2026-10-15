"""ダークモードのスクリーンショットを印刷向け（白背景）に変換するコア処理。

方式:
  1. Lab 色空間で明度 L だけを反転し、色相 (a/b) は維持する
     → 青いリンクは青のまま、緑のバッジは緑のまま
  2. 最頻の明度を背景とみなして純白へ、最暗部を黒へレベル補正
  3. 背景に近い明度の淡い色味を除去して紙の白を残す
  4. 写真・アイコンなどの画像領域は反転せず元のまま残す

CLI:
  python darkprint.py input.png [input2.png ...] [-o OUTDIR] [--gray] [--no-protect]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Options:
    protect_photos: bool = True   # 画像領域を反転しない
    grayscale: bool = False       # 色を捨てて白黒にする
    chroma: float = 1.0           # 文字色の彩度倍率（0 で白黒相当）
    bg_margin: float = 3.0        # 背景明度からこの幅だけ暗い所まで純白に寄せる
    force: bool = False           # ダークモード判定を無視して常に変換


@dataclass
class Result:
    image: np.ndarray             # BGR uint8
    is_dark: bool
    background_l: float           # 元画像の背景明度 (L*, 0-100)
    photo_mask: np.ndarray        # True = 保護した画像領域


# ==================== 解析 ====================

def to_lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def background_lightness(L: np.ndarray) -> float:
    """最頻の明度（= 背景）を返す。"""
    hist = np.histogram(L, bins=100, range=(0, 100))[0]
    return float(hist.argmax()) + 0.5


def is_dark_mode(L: np.ndarray, threshold: float = 40.0) -> bool:
    """背景明度と外周の明度がどちらも暗ければダークモードとみなす。"""
    border = np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]])
    return background_lightness(L) < threshold and float(np.median(border)) < threshold + 15


def detect_photo_regions(bgr: np.ndarray, L: np.ndarray, bg_l: float,
                         min_side: int = 40) -> np.ndarray:
    """写真・アイコンなどの画像領域を矩形単位で検出する。

    文字や UI 部品の領域は「背景色の画素が大半」で明度が二極化しているのに対し、
    写真は背景色の画素が少なく、量子化した色数が多い点で見分ける。
    """
    h, w = L.shape
    mask = np.zeros((h, w), bool)

    # 背景と異なる画素をまとめて塊にする
    fg = (np.abs(L - bg_l) > 4).astype(np.uint8)
    k = max(3, min(h, w) // 150) | 1
    blob = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)

    quant = (bgr // 32).astype(np.int32)
    codes = quant[..., 0] * 64 + quant[..., 1] * 8 + quant[..., 2]

    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bw < min_side or bh < min_side:
            continue
        region_l = L[y:y + bh, x:x + bw]
        bg_ratio = float((np.abs(region_l - bg_l) <= 4).mean())
        colors = np.bincount(codes[y:y + bh, x:x + bw].ravel(), minlength=512)
        # 全体の 0.2% 以上を占める色の数
        n_colors = int((colors > colors.sum() * 0.002).sum())
        # 明度分布の広がり（文字は二極化、写真は連続的）
        mid = float(((region_l > 15) & (region_l < 85)).mean())
        if bg_ratio < 0.35 and n_colors >= 12 and mid > 0.3:
            mask[y:y + bh, x:x + bw] = True
    return mask


# ==================== 変換 ====================

def invert_lightness(lab: np.ndarray, bg_l: float, opt: Options) -> np.ndarray:
    out = lab.copy()
    L = 100 - lab[..., 0]
    paper = 100 - bg_l                       # 反転後の背景明度
    black = float(np.percentile(L, 0.5))
    span = max(paper - opt.bg_margin - black, 1.0)
    L = np.clip((L - black) / span, 0, 1) * 100
    out[..., 0] = L

    chroma = 0.0 if opt.grayscale else opt.chroma
    # 背景付近（L>97）は色味を消して純白に、それ以外は徐々に色を戻す
    fade = np.clip((97 - L) / 5, 0, 1)[..., None]
    out[..., 1:] = lab[..., 1:] * chroma * fade
    return out


def convert(bgr: np.ndarray, opt: Options | None = None) -> Result:
    opt = opt or Options()
    lab = to_lab(bgr)
    L = lab[..., 0]
    bg_l = background_lightness(L)
    dark = is_dark_mode(L)
    photo = np.zeros(L.shape, bool)

    if not (dark or opt.force):
        return Result(bgr.copy(), dark, bg_l, photo)

    out = cv2.cvtColor(invert_lightness(lab, bg_l, opt), cv2.COLOR_LAB2BGR)
    out = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)

    if opt.protect_photos:
        photo = detect_photo_regions(bgr, L, bg_l)
        out[photo] = bgr[photo]
    if opt.grayscale:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return Result(out, dark, bg_l, photo)


def ink_coverage(bgr: np.ndarray) -> float:
    """平均の暗さ (%) — インク使用量の目安。"""
    return float((255 - bgr.astype(np.float32)).mean() / 255 * 100)


# ==================== CLI ====================

def main() -> None:
    ap = argparse.ArgumentParser(description="ダークモード画像を印刷向けに変換します")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, help="出力先（省略時は入力と同じ場所）")
    ap.add_argument("--gray", action="store_true", help="白黒で出力")
    ap.add_argument("--no-protect", action="store_true", help="画像領域も反転する")
    ap.add_argument("--force", action="store_true", help="ライトモード画像も変換する")
    args = ap.parse_args()

    opt = Options(protect_photos=not args.no_protect, grayscale=args.gray, force=args.force)
    for src in args.inputs:
        bgr = cv2.imread(str(src))
        if bgr is None:
            print(f"読み込めません: {src}")
            continue
        res = convert(bgr, opt)
        if not (res.is_dark or opt.force):
            print(f"スキップ（ライトモード）: {src}")
            continue
        outdir = args.outdir or src.parent
        outdir.mkdir(parents=True, exist_ok=True)
        dst = outdir / f"{src.stem}_print.png"
        cv2.imwrite(str(dst), res.image)
        print(f"{src} -> {dst}  インク {ink_coverage(bgr):.1f}% -> {ink_coverage(res.image):.1f}%"
              f"  画像保護 {res.photo_mask.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
