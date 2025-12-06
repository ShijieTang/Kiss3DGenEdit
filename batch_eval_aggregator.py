"""
Batch Evaluation Aggregator - P2P 实验批量评估汇总工具

遍历 results/p2p_batch_results/ 下所有实验文件夹，
使用 Gemini + LPIPS + MVClip 指标评估每个 bundle，并生成跨实验对比结果。

Usage Examples:
  # 评估所有实验（完整评估）
  python batch_eval_aggregator.py --root results/p2p_batch_results/

  # 仅评估特定 category
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --category diff_tau

  # 跳过 Gemini（只计算 LPIPS + MVClip）
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --skip-gemini

  # 跳过 MVClip（只计算 Gemini + LPIPS）
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --skip-mvclip

  # 只计算 LPIPS
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --skip-gemini --skip-mvclip

  # 指定输出目录
  python batch_eval_aggregator.py --root results/p2p_batch_results/ -o custom_output/
"""

import os
import json
import csv
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from PIL import Image

from bundle_evaluator import BundleEvaluator
from mvclip import MVClipEvaluator


# ============================================================================
# 辅助函数
# ============================================================================

def extract_rgb_half(bundle_path: str) -> Image.Image:
    """
    从 Bundle 中提取上半部分的 RGB 视角 (2048x512)

    Bundle 格式: 2048x1024
    - 上半部分 (2048x512): 4个RGB视角
    - 下半部分 (2048x512): 4个法线视角
    """
    img = Image.open(bundle_path).convert("RGB")
    w, h = img.size
    # 只取上半部分
    rgb_half = img.crop((0, 0, w, h // 2))
    return rgb_half


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class EntryEvalResult:
    """单个 entry 的评估结果"""
    entry_name: str
    src_prompt: str
    tgt_prompt: str

    # Gemini 评估 (分别评估 src 和 tgt，不交叉)
    gemini_src_consistency: float = 0.0      # 原图视角一致性
    gemini_src_semantic: float = 0.0         # 原图语义匹配度
    gemini_tgt_consistency: float = 0.0      # 编辑后视角一致性
    gemini_tgt_semantic: float = 0.0         # 编辑后语义匹配度
    gemini_src_analysis: str = ""
    gemini_tgt_analysis: str = ""

    # LPIPS 评估 (越小越好)
    lpips_distance: float = 0.0              # 平均 LPIPS
    lpips_per_view: List[float] = field(default_factory=list)  # 4个视角
    lpips_best_view: int = -1                # 最低 LPIPS 的视角索引
    lpips_worst_view: int = -1               # 最高 LPIPS 的视角索引

    # MVClip 交叉评估 (2 prompts x 2 images = 4 values)
    mvclip_src_src: float = 0.0              # src_prompt + src_img (基准)
    mvclip_src_tgt: float = 0.0              # src_prompt + tgt_img (编辑后与原描述匹配度)
    mvclip_tgt_src: float = 0.0              # tgt_prompt + src_img (原图与目标描述匹配度)
    mvclip_tgt_tgt: float = 0.0              # tgt_prompt + tgt_img (关键指标)

    # MVClip 派生指标
    mvclip_tgt_improvement: float = 0.0      # tgt_tgt - tgt_src (越大越好)
    mvclip_src_preservation: float = 0.0     # src_src - src_tgt (越大越好)

    # MVClip per-view 分数 (只保存关键的两个)
    mvclip_src_src_per_view: List[float] = field(default_factory=list)
    mvclip_tgt_tgt_per_view: List[float] = field(default_factory=list)

    # 路径
    src_bundle_path: str = ""
    tgt_bundle_path: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """单个实验（参数配置）的评估结果"""
    experiment_path: str
    experiment_name: str
    category: str                          # e.g., "diff_editmode_same_identity"
    p2p_tau: float
    p2p_edit_mode: str
    entries: List[EntryEvalResult] = field(default_factory=list)

    # Gemini 平均值 (分别评估 src 和 tgt)
    avg_src_consistency: float = 0.0
    avg_src_semantic: float = 0.0
    avg_tgt_consistency: float = 0.0
    avg_tgt_semantic: float = 0.0

    # LPIPS 平均值
    avg_lpips: float = 0.0

    # MVClip 交叉评估平均值
    avg_mvclip_src_src: float = 0.0
    avg_mvclip_src_tgt: float = 0.0
    avg_mvclip_tgt_src: float = 0.0
    avg_mvclip_tgt_tgt: float = 0.0
    avg_mvclip_tgt_improvement: float = 0.0
    avg_mvclip_src_preservation: float = 0.0

    num_entries: int = 0
    num_success: int = 0
    num_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entries"] = [e.to_dict() for e in self.entries]
        return d

    def compute_averages(self):
        """计算平均值"""
        successful = [e for e in self.entries if e.error is None]
        self.num_entries = len(self.entries)
        self.num_success = len(successful)
        self.num_failed = self.num_entries - self.num_success

        if successful:
            n = len(successful)
            # Gemini (分别评估)
            self.avg_src_consistency = sum(e.gemini_src_consistency for e in successful) / n
            self.avg_src_semantic = sum(e.gemini_src_semantic for e in successful) / n
            self.avg_tgt_consistency = sum(e.gemini_tgt_consistency for e in successful) / n
            self.avg_tgt_semantic = sum(e.gemini_tgt_semantic for e in successful) / n

            # LPIPS
            self.avg_lpips = sum(e.lpips_distance for e in successful) / n

            # MVClip 交叉评估
            self.avg_mvclip_src_src = sum(e.mvclip_src_src for e in successful) / n
            self.avg_mvclip_src_tgt = sum(e.mvclip_src_tgt for e in successful) / n
            self.avg_mvclip_tgt_src = sum(e.mvclip_tgt_src for e in successful) / n
            self.avg_mvclip_tgt_tgt = sum(e.mvclip_tgt_tgt for e in successful) / n
            self.avg_mvclip_tgt_improvement = sum(e.mvclip_tgt_improvement for e in successful) / n
            self.avg_mvclip_src_preservation = sum(e.mvclip_src_preservation for e in successful) / n


# ============================================================================
# 实验发现逻辑
# ============================================================================

def discover_experiments(root_dir: str, category_filter: Optional[str] = None) -> List[Dict]:
    """
    遍历 root_dir，找到所有包含 config.json 的实验文件夹

    Args:
        root_dir: 根目录 (e.g., results/p2p_batch_results/)
        category_filter: 可选，只返回指定 category 的实验

    Returns:
        List[Dict]: [{path, category, experiment_name, config}, ...]
    """
    experiments = []
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"根目录不存在: {root_dir}")

    # 遍历 category 目录 (diff_tau, diff_editmode_same_identity, etc.)
    for category_dir in root_path.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        # 应用 category 过滤
        if category_filter and category_name != category_filter:
            continue

        # 遍历每个实验目录 (mode_qk_img_tau0.1, etc.)
        for exp_dir in category_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            config_path = exp_dir / "config.json"
            if not config_path.exists():
                continue

            # 读取 config
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                warnings.warn(f"无法解析 config.json: {config_path}, 错误: {e}")
                continue

            experiments.append({
                "path": str(exp_dir),
                "category": category_name,
                "experiment_name": exp_dir.name,
                "config": config,
            })

    return experiments


def find_bundle_paths(exp_dir: str, entry_name: str) -> tuple:
    """
    在实验目录中查找 entry 对应的 src/tgt bundle 路径

    Args:
        exp_dir: 实验目录路径
        entry_name: entry 名称 (e.g., "clay->crystal")

    Returns:
        (src_bundle_path, tgt_bundle_path) or (None, None) if not found
    """
    entry_dir = Path(exp_dir) / entry_name
    if not entry_dir.exists():
        return None, None

    src_bundle = None
    tgt_bundle = None

    for file in entry_dir.iterdir():
        if file.suffix.lower() == ".png":
            if "_src_bundle" in file.name:
                src_bundle = str(file)
            elif "_tgt_bundle" in file.name:
                tgt_bundle = str(file)

    return src_bundle, tgt_bundle


# ============================================================================
# 评估逻辑
# ============================================================================

def evaluate_experiment(
    exp_info: Dict,
    evaluator: BundleEvaluator,
    mvclip_evaluator: Optional[MVClipEvaluator] = None,
    skip_gemini: bool = False,
    skip_mvclip: bool = False,
    verbose: bool = False,
) -> ExperimentResult:
    """
    评估单个实验的所有 entries

    Args:
        exp_info: discover_experiments 返回的实验信息字典
        evaluator: BundleEvaluator 实例
        mvclip_evaluator: MVClipEvaluator 实例（可选）
        skip_gemini: 是否跳过 Gemini 评估
        skip_mvclip: 是否跳过 MVClip 评估
        verbose: 是否打印详细信息

    Returns:
        ExperimentResult
    """
    config = exp_info["config"]
    exp_path = exp_info["path"]

    result = ExperimentResult(
        experiment_path=exp_path,
        experiment_name=exp_info["experiment_name"],
        category=exp_info["category"],
        p2p_tau=config.get("p2p_tau", 0.0),
        p2p_edit_mode=config.get("p2p_edit_mode", ""),
    )

    entries = config.get("entries", [])

    for entry in entries:
        entry_name = entry.get("name", "")
        src_prompt = entry.get("src_prompt", "")
        tgt_prompt = entry.get("tgt_prompt", "")

        if verbose:
            print(f"    评估 entry: {entry_name}")

        # 查找 bundle 文件
        src_bundle, tgt_bundle = find_bundle_paths(exp_path, entry_name)

        entry_result = EntryEvalResult(
            entry_name=entry_name,
            src_prompt=src_prompt,
            tgt_prompt=tgt_prompt,
            src_bundle_path=src_bundle or "",
            tgt_bundle_path=tgt_bundle or "",
        )

        if not src_bundle or not tgt_bundle:
            entry_result.error = f"Bundle 文件缺失: src={src_bundle}, tgt={tgt_bundle}"
            result.entries.append(entry_result)
            continue

        try:
            # 1. BundleEvaluator: 分别评估 src 和 tgt (Gemini + LPIPS)

            # 1.1 评估 tgt_bundle (包含 LPIPS: src vs tgt)
            tgt_eval = evaluator.evaluate_bundle(
                bundle=tgt_bundle,
                prompt=tgt_prompt,
                reference_bundle=src_bundle,
                skip_gemini=skip_gemini,
            )
            entry_result.gemini_tgt_consistency = tgt_eval.gemini_consistency_score
            entry_result.gemini_tgt_semantic = tgt_eval.gemini_semantic_score
            entry_result.gemini_tgt_analysis = (
                f"Consistency: {tgt_eval.gemini_consistency_analysis}\n"
                f"Semantic: {tgt_eval.gemini_semantic_analysis}"
            )

            # LPIPS (src vs tgt)
            entry_result.lpips_distance = tgt_eval.lpips_distance or 0.0
            entry_result.lpips_per_view = tgt_eval.lpips_distances_per_view or []

            # 计算 LPIPS 最佳/最差视角 (LPIPS 越小越好)
            if entry_result.lpips_per_view:
                entry_result.lpips_best_view = int(
                    min(range(len(entry_result.lpips_per_view)),
                        key=lambda i: entry_result.lpips_per_view[i])
                )
                entry_result.lpips_worst_view = int(
                    max(range(len(entry_result.lpips_per_view)),
                        key=lambda i: entry_result.lpips_per_view[i])
                )

            # 1.2 评估 src_bundle (不需要 reference)
            if not skip_gemini:
                src_eval = evaluator.evaluate_bundle(
                    bundle=src_bundle,
                    prompt=src_prompt,
                    reference_bundle=None,
                    skip_gemini=False,
                )
                entry_result.gemini_src_consistency = src_eval.gemini_consistency_score
                entry_result.gemini_src_semantic = src_eval.gemini_semantic_score
                entry_result.gemini_src_analysis = (
                    f"Consistency: {src_eval.gemini_consistency_analysis}\n"
                    f"Semantic: {src_eval.gemini_semantic_analysis}"
                )

            # 2. MVClip 交叉评估 (2 prompts x 2 images = 4 values)
            if not skip_mvclip and mvclip_evaluator is not None:
                try:
                    # 提取上半部分 RGB 视角
                    src_rgb = extract_rgb_half(src_bundle)
                    tgt_rgb = extract_rgb_half(tgt_bundle)

                    # 4 个交叉评估
                    score_src_src, per_view_src_src = mvclip_evaluator.evaluate_bundle(
                        bundle=src_rgb,
                        base_prompt=src_prompt,
                        verbose_scores=False,
                        save_plot=False,
                    )
                    score_src_tgt, _ = mvclip_evaluator.evaluate_bundle(
                        bundle=tgt_rgb,
                        base_prompt=src_prompt,
                        verbose_scores=False,
                        save_plot=False,
                    )
                    score_tgt_src, _ = mvclip_evaluator.evaluate_bundle(
                        bundle=src_rgb,
                        base_prompt=tgt_prompt,
                        verbose_scores=False,
                        save_plot=False,
                    )
                    score_tgt_tgt, per_view_tgt_tgt = mvclip_evaluator.evaluate_bundle(
                        bundle=tgt_rgb,
                        base_prompt=tgt_prompt,
                        verbose_scores=False,
                        save_plot=False,
                    )

                    # 保存 4 个交叉评估结果
                    entry_result.mvclip_src_src = score_src_src
                    entry_result.mvclip_src_tgt = score_src_tgt
                    entry_result.mvclip_tgt_src = score_tgt_src
                    entry_result.mvclip_tgt_tgt = score_tgt_tgt

                    # 保存 per-view 分数
                    entry_result.mvclip_src_src_per_view = per_view_src_src
                    entry_result.mvclip_tgt_tgt_per_view = per_view_tgt_tgt

                    # 计算派生指标
                    entry_result.mvclip_tgt_improvement = score_tgt_tgt - score_tgt_src
                    entry_result.mvclip_src_preservation = score_src_src - score_src_tgt

                except Exception as e:
                    warnings.warn(f"MVClip 评估失败 {entry_name}: {e}")

        except Exception as e:
            entry_result.error = str(e)
            warnings.warn(f"评估失败 {entry_name}: {e}")

        result.entries.append(entry_result)

    # 计算平均值
    result.compute_averages()
    return result


def evaluate_all_experiments(
    root_dir: str,
    category_filter: Optional[str] = None,
    skip_gemini: bool = False,
    skip_mvclip: bool = False,
    verbose: bool = True,
) -> List[ExperimentResult]:
    """
    评估所有实验

    Args:
        root_dir: 根目录
        category_filter: 可选的 category 过滤
        skip_gemini: 是否跳过 Gemini
        skip_mvclip: 是否跳过 MVClip
        verbose: 是否打印进度

    Returns:
        List[ExperimentResult]
    """
    experiments = discover_experiments(root_dir, category_filter)

    if not experiments:
        print("未找到任何实验")
        return []

    if verbose:
        print(f"发现 {len(experiments)} 个实验")

    # 创建共享的 evaluator 实例（模型只加载一次）
    evaluator = BundleEvaluator(lazy_load=True)

    # 创建 MVClip evaluator（如果需要）
    # 使用 rows=1, cols=4 只评估上半部分的4个RGB视角
    mvclip_evaluator = None
    if not skip_mvclip:
        if verbose:
            print("初始化 MVClip 评估器 (4 RGB views only)...")
        mvclip_evaluator = MVClipEvaluator(
            model_type="ImageReward",
            rows=1,
            cols=4,
            save_dir=None
        )

    results = []

    for i, exp_info in enumerate(experiments, 1):
        if verbose:
            print(f"\n[{i}/{len(experiments)}] 评估: {exp_info['category']}/{exp_info['experiment_name']}")

        result = evaluate_experiment(
            exp_info, evaluator, mvclip_evaluator,
            skip_gemini=skip_gemini, skip_mvclip=skip_mvclip, verbose=verbose
        )
        results.append(result)

        if verbose:
            mvclip_str = ""
            if not skip_mvclip:
                mvclip_str = (f", MVClip(tgt_tgt): {result.avg_mvclip_tgt_tgt:.4f}, "
                             f"Improvement: {result.avg_mvclip_tgt_improvement:+.4f}")
            print(f"  -> LPIPS: {result.avg_lpips:.4f}, "
                  f"Gemini(tgt): {result.avg_tgt_consistency:.1f}/{result.avg_tgt_semantic:.1f}"
                  f"{mvclip_str}")

    # 释放模型内存
    evaluator.release_models()

    return results


# ============================================================================
# 跨实验对比
# ============================================================================

def compare_experiments(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    跨实验对比分析

    Args:
        results: 所有实验的评估结果

    Returns:
        包含对比分析的字典
    """
    if not results:
        return {}

    comparison = {
        "by_category": {},
        "best_params": {},
        "entry_comparison": [],
    }

    # 按 category 分组
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    # 每个 category 内的最优参数
    for cat_name, cat_results in categories.items():
        cat_info = {
            "experiments": [],
            "best_by_lpips": None,
            "best_by_tgt_consistency": None,
            "best_by_tgt_semantic": None,
            "best_by_mvclip_tgt_tgt": None,
            "best_by_mvclip_improvement": None,
        }

        for r in cat_results:
            cat_info["experiments"].append({
                "name": r.experiment_name,
                "edit_mode": r.p2p_edit_mode,
                "tau": r.p2p_tau,
                "avg_lpips": r.avg_lpips,
                # Gemini (分别评估)
                "avg_src_consistency": r.avg_src_consistency,
                "avg_src_semantic": r.avg_src_semantic,
                "avg_tgt_consistency": r.avg_tgt_consistency,
                "avg_tgt_semantic": r.avg_tgt_semantic,
                # MVClip 交叉评估
                "avg_mvclip_src_src": r.avg_mvclip_src_src,
                "avg_mvclip_src_tgt": r.avg_mvclip_src_tgt,
                "avg_mvclip_tgt_src": r.avg_mvclip_tgt_src,
                "avg_mvclip_tgt_tgt": r.avg_mvclip_tgt_tgt,
                "avg_mvclip_tgt_improvement": r.avg_mvclip_tgt_improvement,
                "avg_mvclip_src_preservation": r.avg_mvclip_src_preservation,
            })

        # 找最优
        if cat_results:
            # LPIPS 越小越好
            best_lpips = min(cat_results, key=lambda x: x.avg_lpips if x.avg_lpips > 0 else float('inf'))
            cat_info["best_by_lpips"] = {
                "name": best_lpips.experiment_name,
                "value": best_lpips.avg_lpips,
            }

            # tgt_consistency 越高越好
            best_consistency = max(cat_results, key=lambda x: x.avg_tgt_consistency)
            cat_info["best_by_tgt_consistency"] = {
                "name": best_consistency.experiment_name,
                "value": best_consistency.avg_tgt_consistency,
            }

            # tgt_semantic 越高越好
            best_semantic = max(cat_results, key=lambda x: x.avg_tgt_semantic)
            cat_info["best_by_tgt_semantic"] = {
                "name": best_semantic.experiment_name,
                "value": best_semantic.avg_tgt_semantic,
            }

            # MVClip tgt_tgt 越高越好
            best_mvclip = max(cat_results, key=lambda x: x.avg_mvclip_tgt_tgt)
            cat_info["best_by_mvclip_tgt_tgt"] = {
                "name": best_mvclip.experiment_name,
                "value": best_mvclip.avg_mvclip_tgt_tgt,
            }

            # MVClip improvement 越高越好
            best_improvement = max(cat_results, key=lambda x: x.avg_mvclip_tgt_improvement)
            cat_info["best_by_mvclip_improvement"] = {
                "name": best_improvement.experiment_name,
                "value": best_improvement.avg_mvclip_tgt_improvement,
            }

        comparison["by_category"][cat_name] = cat_info

    # 全局最优参数
    if results:
        best_lpips = min(results, key=lambda x: x.avg_lpips if x.avg_lpips > 0 else float('inf'))
        best_tgt_consistency = max(results, key=lambda x: x.avg_tgt_consistency)
        best_tgt_semantic = max(results, key=lambda x: x.avg_tgt_semantic)
        best_mvclip_tgt_tgt = max(results, key=lambda x: x.avg_mvclip_tgt_tgt)
        best_mvclip_improvement = max(results, key=lambda x: x.avg_mvclip_tgt_improvement)

        comparison["best_params"] = {
            "best_lpips": {
                "experiment": f"{best_lpips.category}/{best_lpips.experiment_name}",
                "edit_mode": best_lpips.p2p_edit_mode,
                "tau": best_lpips.p2p_tau,
                "value": best_lpips.avg_lpips,
            },
            "best_tgt_consistency": {
                "experiment": f"{best_tgt_consistency.category}/{best_tgt_consistency.experiment_name}",
                "edit_mode": best_tgt_consistency.p2p_edit_mode,
                "tau": best_tgt_consistency.p2p_tau,
                "value": best_tgt_consistency.avg_tgt_consistency,
            },
            "best_tgt_semantic": {
                "experiment": f"{best_tgt_semantic.category}/{best_tgt_semantic.experiment_name}",
                "edit_mode": best_tgt_semantic.p2p_edit_mode,
                "tau": best_tgt_semantic.p2p_tau,
                "value": best_tgt_semantic.avg_tgt_semantic,
            },
            "best_mvclip_tgt_tgt": {
                "experiment": f"{best_mvclip_tgt_tgt.category}/{best_mvclip_tgt_tgt.experiment_name}",
                "edit_mode": best_mvclip_tgt_tgt.p2p_edit_mode,
                "tau": best_mvclip_tgt_tgt.p2p_tau,
                "value": best_mvclip_tgt_tgt.avg_mvclip_tgt_tgt,
            },
            "best_mvclip_improvement": {
                "experiment": f"{best_mvclip_improvement.category}/{best_mvclip_improvement.experiment_name}",
                "edit_mode": best_mvclip_improvement.p2p_edit_mode,
                "tau": best_mvclip_improvement.p2p_tau,
                "value": best_mvclip_improvement.avg_mvclip_tgt_improvement,
            },
        }

    # Entry 级别对比
    entry_data = {}
    for r in results:
        for e in r.entries:
            if e.entry_name not in entry_data:
                entry_data[e.entry_name] = []
            entry_data[e.entry_name].append({
                "category": r.category,
                "experiment": r.experiment_name,
                "edit_mode": r.p2p_edit_mode,
                "tau": r.p2p_tau,
                # Gemini (分别评估)
                "src_consistency": e.gemini_src_consistency,
                "src_semantic": e.gemini_src_semantic,
                "tgt_consistency": e.gemini_tgt_consistency,
                "tgt_semantic": e.gemini_tgt_semantic,
                # LPIPS
                "lpips": e.lpips_distance,
                "lpips_per_view": e.lpips_per_view,
                "lpips_best_view": e.lpips_best_view,
                "lpips_worst_view": e.lpips_worst_view,
                # MVClip 交叉评估
                "mvclip_src_src": e.mvclip_src_src,
                "mvclip_src_tgt": e.mvclip_src_tgt,
                "mvclip_tgt_src": e.mvclip_tgt_src,
                "mvclip_tgt_tgt": e.mvclip_tgt_tgt,
                "mvclip_tgt_improvement": e.mvclip_tgt_improvement,
                "mvclip_src_preservation": e.mvclip_src_preservation,
                "error": e.error,
            })

    comparison["entry_comparison"] = entry_data

    return comparison


# ============================================================================
# 输出生成
# ============================================================================

def save_results(
    results: List[ExperimentResult],
    comparison: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """
    保存评估结果到文件

    Args:
        results: 所有实验结果
        comparison: 对比分析结果
        output_dir: 输出目录

    Returns:
        生成的文件路径字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = {}

    # 1. 详细结果 JSON
    detailed_path = output_path / f"detailed_results_{timestamp}.json"
    detailed_data = {
        "timestamp": timestamp,
        "num_experiments": len(results),
        "experiments": [r.to_dict() for r in results],
    }
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, ensure_ascii=False, indent=2)
    files["detailed_results"] = str(detailed_path)

    # 2. 汇总 CSV
    summary_path = output_path / f"summary_{timestamp}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "experiment", "edit_mode", "tau",
            # Gemini (分别评估)
            "avg_src_consistency", "avg_src_semantic",
            "avg_tgt_consistency", "avg_tgt_semantic",
            # LPIPS
            "avg_lpips",
            # MVClip 交叉评估
            "avg_mvclip_src_src", "avg_mvclip_src_tgt",
            "avg_mvclip_tgt_src", "avg_mvclip_tgt_tgt",
            "avg_mvclip_improvement", "avg_mvclip_preservation",
            # 统计
            "num_entries", "num_success", "num_failed"
        ])
        for r in results:
            writer.writerow([
                r.category, r.experiment_name, r.p2p_edit_mode, r.p2p_tau,
                # Gemini
                f"{r.avg_src_consistency:.2f}", f"{r.avg_src_semantic:.2f}",
                f"{r.avg_tgt_consistency:.2f}", f"{r.avg_tgt_semantic:.2f}",
                # LPIPS
                f"{r.avg_lpips:.4f}",
                # MVClip
                f"{r.avg_mvclip_src_src:.4f}", f"{r.avg_mvclip_src_tgt:.4f}",
                f"{r.avg_mvclip_tgt_src:.4f}", f"{r.avg_mvclip_tgt_tgt:.4f}",
                f"{r.avg_mvclip_tgt_improvement:+.4f}", f"{r.avg_mvclip_src_preservation:+.4f}",
                # 统计
                r.num_entries, r.num_success, r.num_failed
            ])
    files["summary"] = str(summary_path)

    # 3. Entry 对比 CSV
    entry_path = output_path / f"comparison_by_entry_{timestamp}.csv"
    with open(entry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "entry_name", "category", "experiment", "edit_mode", "tau",
            # Gemini (分别评估)
            "src_consistency", "src_semantic",
            "tgt_consistency", "tgt_semantic",
            # LPIPS
            "lpips", "lpips_best_view", "lpips_worst_view",
            # MVClip 交叉评估
            "mvclip_src_src", "mvclip_src_tgt",
            "mvclip_tgt_src", "mvclip_tgt_tgt",
            "mvclip_improvement", "mvclip_preservation",
            "error"
        ])
        for entry_name, entries in comparison.get("entry_comparison", {}).items():
            for e in entries:
                writer.writerow([
                    entry_name, e["category"], e["experiment"],
                    e["edit_mode"], e["tau"],
                    # Gemini
                    f"{e['src_consistency']:.2f}" if e.get("src_consistency") else "",
                    f"{e['src_semantic']:.2f}" if e.get("src_semantic") else "",
                    f"{e['tgt_consistency']:.2f}" if e.get("tgt_consistency") else "",
                    f"{e['tgt_semantic']:.2f}" if e.get("tgt_semantic") else "",
                    # LPIPS
                    f"{e['lpips']:.4f}" if e.get("lpips") else "",
                    e.get("lpips_best_view", ""),
                    e.get("lpips_worst_view", ""),
                    # MVClip
                    f"{e['mvclip_src_src']:.4f}" if e.get("mvclip_src_src") else "",
                    f"{e['mvclip_src_tgt']:.4f}" if e.get("mvclip_src_tgt") else "",
                    f"{e['mvclip_tgt_src']:.4f}" if e.get("mvclip_tgt_src") else "",
                    f"{e['mvclip_tgt_tgt']:.4f}" if e.get("mvclip_tgt_tgt") else "",
                    f"{e['mvclip_tgt_improvement']:+.4f}" if e.get("mvclip_tgt_improvement") else "",
                    f"{e['mvclip_src_preservation']:+.4f}" if e.get("mvclip_src_preservation") else "",
                    e.get("error", "")
                ])
    files["entry_comparison"] = str(entry_path)

    # 4. 最优参数 JSON
    best_path = output_path / f"best_params_{timestamp}.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    files["best_params"] = str(best_path)

    return files


def print_summary(results: List[ExperimentResult], comparison: Dict[str, Any]):
    """打印汇总信息"""
    print("\n" + "=" * 70)
    print("评估汇总 (交叉评估)")
    print("=" * 70)

    print(f"\n总实验数: {len(results)}")

    # 按 category 打印
    for cat_name, cat_info in comparison.get("by_category", {}).items():
        print(f"\n--- {cat_name} ---")
        for exp in sorted(cat_info["experiments"], key=lambda x: x["tau"]):
            mvclip_str = ""
            if exp.get('avg_mvclip_tgt_tgt'):
                mvclip_str = (f"\n      MVClip: src_src={exp['avg_mvclip_src_src']:.4f}, "
                             f"tgt_tgt={exp['avg_mvclip_tgt_tgt']:.4f}, "
                             f"improve={exp['avg_mvclip_tgt_improvement']:+.4f}")
            print(f"  {exp['name']}: LPIPS={exp['avg_lpips']:.4f}, "
                  f"Gemini(tgt)={exp['avg_tgt_consistency']:.1f}/{exp['avg_tgt_semantic']:.1f}"
                  f"{mvclip_str}")

        if cat_info.get("best_by_lpips"):
            print(f"  最优 LPIPS: {cat_info['best_by_lpips']['name']} ({cat_info['best_by_lpips']['value']:.4f})")
        if cat_info.get("best_by_mvclip_improvement") and cat_info["best_by_mvclip_improvement"]["value"] != 0:
            print(f"  最优 MVClip Improvement: {cat_info['best_by_mvclip_improvement']['name']} ({cat_info['best_by_mvclip_improvement']['value']:+.4f})")

    # 全局最优
    best = comparison.get("best_params", {})
    if best:
        print("\n--- 全局最优参数 ---")
        if "best_lpips" in best:
            b = best["best_lpips"]
            print(f"  最低 LPIPS: {b['experiment']} (tau={b['tau']}, mode={b['edit_mode']}) -> {b['value']:.4f}")
        if "best_tgt_consistency" in best:
            b = best["best_tgt_consistency"]
            print(f"  最高 tgt_Consistency: {b['experiment']} (tau={b['tau']}, mode={b['edit_mode']}) -> {b['value']:.1f}")
        if "best_tgt_semantic" in best:
            b = best["best_tgt_semantic"]
            print(f"  最高 tgt_Semantic: {b['experiment']} (tau={b['tau']}, mode={b['edit_mode']}) -> {b['value']:.1f}")
        if "best_mvclip_tgt_tgt" in best and best["best_mvclip_tgt_tgt"]["value"] != 0:
            b = best["best_mvclip_tgt_tgt"]
            print(f"  最高 MVClip(tgt_tgt): {b['experiment']} (tau={b['tau']}, mode={b['edit_mode']}) -> {b['value']:.4f}")
        if "best_mvclip_improvement" in best and best["best_mvclip_improvement"]["value"] != 0:
            b = best["best_mvclip_improvement"]
            print(f"  最高 MVClip Improvement: {b['experiment']} (tau={b['tau']}, mode={b['edit_mode']}) -> {b['value']:+.4f}")


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Evaluation Aggregator - P2P 实验批量评估汇总",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估所有实验（完整: Gemini + LPIPS + MVClip）
  python batch_eval_aggregator.py --root results/p2p_batch_results/

  # 仅评估特定 category
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --category diff_tau

  # 跳过 Gemini（LPIPS + MVClip）
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --skip-gemini

  # 跳过 MVClip（Gemini + LPIPS）
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --skip-mvclip

  # 只计算 LPIPS
  python batch_eval_aggregator.py --root results/p2p_batch_results/ --skip-gemini --skip-mvclip
        """
    )

    parser.add_argument(
        "--root", "-r",
        type=str,
        required=True,
        help="实验结果根目录 (e.g., results/p2p_batch_results/)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录，默认为 {root}/evaluation_summary/"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="只评估指定 category (e.g., diff_tau)"
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="跳过 Gemini 评估"
    )
    parser.add_argument(
        "--skip-mvclip",
        action="store_true",
        help="跳过 MVClip 评估"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式，减少输出"
    )

    args = parser.parse_args()

    # 设置输出目录
    output_dir = args.output or os.path.join(args.root, "evaluation_summary")

    # 执行评估
    print(f"开始评估: {args.root}")
    if args.category:
        print(f"过滤 category: {args.category}")
    if args.skip_gemini:
        print("跳过 Gemini 评估")
    if args.skip_mvclip:
        print("跳过 MVClip 评估")

    results = evaluate_all_experiments(
        root_dir=args.root,
        category_filter=args.category,
        skip_gemini=args.skip_gemini,
        skip_mvclip=args.skip_mvclip,
        verbose=not args.quiet,
    )

    if not results:
        print("没有评估结果")
        return

    # 对比分析
    comparison = compare_experiments(results)

    # 保存结果
    files = save_results(results, comparison, output_dir)

    # 打印汇总
    if not args.quiet:
        print_summary(results, comparison)

    print("\n" + "=" * 70)
    print("输出文件:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
