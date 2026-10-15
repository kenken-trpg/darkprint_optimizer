"""
ダークモード・PDF印刷最適化変換システム 参照実装（v1.2）
レビュー指摘事項対応: 色空間統合, Pixmap API準拠, ゼロ除算制御, パーサー堅牢化, DBSCAN最適化
"""

from __future__ import annotations
import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDFが必要です: pip install PyMuPDF")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 1. ルールエンジン定義・検証
# =============================================================================
class RuleEngine:
    """YAML定義に基づくレイヤー/領域変換ルール管理エンジン"""
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.rules = sorted(
            self.config.get("layer_categories", []),
            key=lambda r: r.get("transform", {}).get("priority", 0),
            reverse=True
        )

    @staticmethod
    def _parse_condition(condition: Union[str, float, dict]) -> Tuple[str, float]:
        """YAML条件式の堅牢なパース（数値リテラル・文字列・構造化形式に対応）"""
        if isinstance(condition, dict):
            return condition.get("operator", ">="), float(condition.get("value", 0))
        # 数値リテラルまたは数値文字列のフォールバック
        try:
            return "==", float(condition)
        except (ValueError, TypeError):
            pass
        # 演算子付き文字列のパース
        match = re.match(r"^([<>]=?)\s*(\d+(?:\.\d+)?)$", str(condition).strip())
        if match:
            return match.group(1), float(match.group(2))
        raise ValueError(f"不正な条件式フォーマット: {condition}")

    @staticmethod
    def _evaluate_condition(op: str, threshold: float, value: float) -> bool:
        if op == "<": return value < threshold
        if op == "<=": return value <= threshold
        if op == ">": return value > threshold
        if op == ">=": return value >= threshold
        if op == "==": return abs(value - threshold) < 1e-3
        return False

    def match_and_transform(self, region_meta: Dict) -> Dict:
        for rule in self.rules:
            if self._check_trigger(rule["trigger"], region_meta):
                return self._apply_transform(rule["transform"], region_meta)
        # 修正: ルールにマッチしない場合でもデフォルトのポリシーを適用して返す
        return self._apply_transform({}, region_meta)

    def _check_trigger(self, trigger: Dict, meta: Dict) -> bool:
        if "ocg_keywords" in trigger:
            if any(kw in meta.get("role", "").lower() for kw in trigger["ocg_keywords"]):
                return True
        if "geometry" in trigger:
            if trigger["geometry"].get("min_stroke", 0) > meta.get("stroke_width", 0):
                return False
        if "color" in trigger:
            if "avg_luminance" in trigger["color"]:
                op, th = self._parse_condition(trigger["color"]["avg_luminance"])
                if not self._evaluate_condition(op, th, meta.get("luminance", 255)):
                    return False
        return True

    def _apply_transform(self, transform: Dict, meta: Dict) -> Dict:
        meta["transform_policy"] = transform.get("ink_policy", "preserve")
        meta["gamma"] = transform.get("gamma_correction", 1.0)
        meta["target_k"] = transform.get("target_gray_k", None)
        meta["preserve_geometry"] = transform.get("geometry_tolerance_mm", 0.0) > 0
        return meta


# =============================================================================
# 2. 画像処理モジュール（BGR基準・HSV反転・KD-Tree最適化）
# =============================================================================
class ImageProcessor:
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
        self.orb = cv2.ORB_create(nfeatures=5000)

    def extract_and_cluster(self, img: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        kp, des = self.orb.detectAndCompute(gray, None)
        if not kp:
            return []

        points = np.array([p.pt for p in kp], dtype=np.float32)
        # 画像サイズに応じた動的epsの計算 (対角線の約2%をクラスタリングの閾値とする)
        diag = math.sqrt(img.shape[0]**2 + img.shape[1]**2)
        eps_dynamic = max(10.0, diag * 0.02)
        
        clustering = DBSCAN(eps=eps_dynamic, min_samples=5, metric="euclidean", algorithm="kd_tree").fit(points)
        labels = clustering.labels_

        regions = []
        for lbl in set(labels):
            if lbl == -1: continue
            pts = points[labels == lbl]
            x, y, w, h = cv2.boundingRect(pts)
            roi = gray[y:y+h, x:x+w]
            regions.append({
                "bbox": (x, y, w, h),
                "luminance": float(roi.mean()),
                "role": "dark_ui_bg" if roi.mean() < 60 else "content_image",
                "stroke_width": 0.0,
                "type": "raster_region"
            })
        return regions

    def selective_inversion(self, img: np.ndarray, regions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        output = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for r in regions:
            r_meta = self.rule_engine.match_and_transform(r)
            if r_meta["transform_policy"] == "invert_selective":
                x, y, w, h = r_meta["bbox"]
                roi = output[y:y+h, x:x+w].copy()

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
                v_norm = hsv[:, :, 2] / 255.0
                gamma = r_meta.get("gamma", 1.2)
                inv_v = (1.0 - v_norm) ** (1.0 / gamma)
                hsv[:, :, 2] = np.clip(inv_v * 255.0, 0, 255).astype(np.uint8)

                output[y:y+h, x:x+w] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                mask[y:y+h, x:x+w] = 255
        return output, mask


# =============================================================================
# 3. PDF/OCG解析・インク最適化モジュール（RGB/BGR統合・TAC厳密計算）
# =============================================================================
class PDFProcessor:
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine

    def estimate_tac(self, img: np.ndarray) -> Dict:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb = img.astype(np.float32) / 255.0
        # RGB -> CMY 変換のチャンネル対応修正 (R:0->C, G:1->M, B:2->Y)
        c = 1.0 - rgb[:, :, 0]
        m = 1.0 - rgb[:, :, 1]
        y = 1.0 - rgb[:, :, 2]
        k = np.minimum(np.minimum(c, m), y)
        k_inv = 1.0 - k
        # ゼロ除算厳密制御 (k_inv == 0 の場合 CMY=0)
        c = np.divide(c - k, k_inv, out=np.zeros_like(c), where=k_inv > 1e-6)
        m = np.divide(m - k, k_inv, out=np.zeros_like(m), where=k_inv > 1e-6)
        y = np.divide(y - k, k_inv, out=np.zeros_like(y), where=k_inv > 1e-6)
        
        tac_map = (c + m + y + k) * 100.0
        return {
            "max_tac": float(np.max(tac_map)),
            "overflow_ratio": float(np.mean(tac_map > 300.0)) * 100.0,
            "avg_tac": float(np.mean(tac_map))
        }

    def optimize_pdf(self, input_path: str, output_path: str):
        logger.warning("PDF処理はPoC段階のラスタライズフォールバックを採用しています。"
                       "テキスト検索性は維持されず、ベクター構造は平坦化されます。")
        doc = fitz.open(input_path)
        tac_summary = {"max": 0.0, "overflow": 0.0, "avg": 0.0}

        for page in doc:
            pix = page.get_pixmap(dpi=150)
            # PyMuPDF (RGB) → OpenCV (BGR) 変換
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            proc = ImageProcessor(self.rule_engine)
            regions = proc.extract_and_cluster(img_bgr)
            optimized_bgr, _ = proc.selective_inversion(img_bgr, regions)

            # OpenCV (BGR) → PyMuPDF (RGB) 変換
            optimized_rgb = cv2.cvtColor(optimized_bgr, cv2.COLOR_BGR2RGB)
            
            tac = self.estimate_tac(optimized_rgb)
            tac_summary["max"] = max(tac_summary["max"], tac["max_tac"])
            tac_summary["overflow"] += tac["overflow_ratio"]

            # Pixmap生成（APIシグネチャ厳密準拠）
            new_pixmap = fitz.Pixmap(fitz.csRGB, optimized_rgb.shape[1], optimized_rgb.shape[0], optimized_rgb.tobytes(), False)
            
            # ページコンテンツ初期化 → 最適化画像挿入
            page.clean_contents()
            page.insert_image(page.rect, pixmap=new_pixmap)

        doc.save(output_path, garbage=4, deflate=True, clean=True, pdf_version=1.7)
        logger.info(f"PDF出力完了: {output_path} | 推定最大TAC: {tac_summary['max']:.1f}%")


# =============================================================================
# 4. メインパイプライン
# =============================================================================
class ConversionPipeline:
    def __init__(self, config_path: str):
        self.rule_engine = RuleEngine(config_path)
        self.img_proc = ImageProcessor(self.rule_engine)
        self.pdf_proc = PDFProcessor(self.rule_engine)

    def run(self, input_path: str, output_path: str):
        p = Path(input_path)
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            self._process_image(p, Path(output_path))
        elif p.suffix.lower() == ".pdf":
            self.pdf_proc.optimize_pdf(input_path, output_path)
        else:
            raise ValueError("未対応フォーマットです。PNG/JPG/PDFを入力してください。")

    def _process_image(self, in_path: Path, out_path: Path):
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"画像の読み込みに失敗: {in_path}")
        regions = self.img_proc.extract_and_cluster(img)
        optimized, mask = self.img_proc.selective_inversion(img, regions)
        cv2.imwrite(str(out_path), optimized)
        tac = self.pdf_proc.estimate_tac(optimized)
        logger.info(f"画像変換完了: {out_path} | 検出領域: {len(regions)} | MaxTAC: {tac['max_tac']:.1f}%")


# =============================================================================
# CLI エントリーポイント
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ダークモード印刷最適化変換エンジン v1.2")
    parser.add_argument("-i", "--input", required=True, help="入力ファイル (PNG/JPG/PDF)")
    parser.add_argument("-o", "--output", required=True, help="出力ファイルパス")
    parser.add_argument("-c", "--config", default="config/rules.yaml", help="ルール設定ファイル (YAML)")
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.info("デフォルトルール設定を生成します。")
        Path("config/rules.yaml").write_text("""
schema_version: "1.0"
layer_categories:
  - name: dark_ui_background
    trigger: { color: { avg_luminance: "< 80" } }
    transform: { ink_policy: "invert_selective", gamma_correction: 1.2, priority: 3 }
  - name: structural_lines
    trigger: { geometry: { min_stroke: 0.5 } }
    transform: { ink_policy: "preserve", priority: 10 }
  - name: auxiliary_grid
    trigger: { color: { avg_luminance: { operator: ">", value: 200 } } }
    transform: { ink_policy: "reduce", target_gray_k: 12, priority: 1 }
""", encoding="utf-8")

    pipeline = ConversionPipeline(args.config)
    pipeline.run(args.input, args.output)

if __name__ == "__main__":
    main()