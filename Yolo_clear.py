"""
YOLO 格式数据集小目标自动分析与清洗脚本（增强版）

功能：
  1. 计算数据集中每个目标在缩放到 YOLO 训练分辨率后的实际像素大小
  2. 统计小于设定最小像素阈值的目标占比、数量
  3. 自动删除过小目标对应的标签行，生成干净数据集
  4. 生成小目标分布可视化图（散点图 + 直方图）
  5. 支持批量处理多个数据集 & YAML 配置文件
  6. 完整的日志记录 + CSV 清洗明细 + 统计报告

用法：
  python Yolo_clear.py                           # 仅统计分析，不修改文件
  python Yolo_clear.py -d                        # 统计 + 直接删除小目标标签行
  python Yolo_clear.py --config config.yaml      # 使用自定义配置文件
  python Yolo_clear.py --datasets ./ds1 ./ds2    # 批量处理多个数据集
  python Yolo_clear.py -d --output ./my_reports  # 指定报告输出目录
  python Yolo_clear.py --no-plot                 # 跳过生成可视化图表
"""

import argparse
import csv
import io
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 修复 Windows 终端 GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ======================== 可选依赖（优雅降级） ========================
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，无需 GUI
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ======================== 内部默认配置 ========================
DEFAULT_CONFIG = {
    "yolo_train_res": 640,
    "min_area_threshold": 128,      # 安全面积：训练分辨率下至少有 128 像素 (对标 640x640 下的长宽)
    "min_single_side": 8,           # 安全单边：宽或高中最细的一边至少 8 像素
    "max_aspect_ratio": 6.0,       # 安全长宽比：长边与短边比例不应超过 6:1 (例如 5x50 = 10倍)
    "label_dirs": [
        "train/labels",
        "Val/labels",
    ],
}


# ======================== 日志系统 ========================
def setup_logging(output_dir: Path) -> logging.Logger:
    """
    配置日志系统：同时输出到终端和日志文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "yolo_clear.log"

    logger = logging.getLogger("yolo_clear")
    logger.setLevel(logging.DEBUG)

    # 清除旧的 handler（防止重复添加）
    logger.handlers.clear()

    # 文件 handler —— 记录所有级别
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # 终端 handler —— 仅记录 INFO 及以上
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


# ======================== 配置加载 ========================
def load_config(config_path: str | None) -> dict:
    """
    加载 YAML 配置文件，与默认配置合并。
    优先级：配置文件 > 代码内默认值
    """
    config = DEFAULT_CONFIG.copy()

    if config_path is None:
        return config

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[警告] 配置文件不存在: {config_path}，使用默认配置")
        return config

    if not HAS_YAML:
        print("[警告] 未安装 pyyaml，无法解析配置文件。请运行: pip install pyyaml")
        return config

    with open(config_file, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    # 合并：用户配置覆盖默认值
    for key in config:
        if key in user_config:
            config[key] = user_config[key]

    return config


# ======================== 核心分析函数 ========================
def analyze_labels(
    label_dir: Path,
    delete_mode: bool,
    train_res: int,
    config: dict,
    logger: logging.Logger,
) -> dict:
    """
    分析单个标签目录，根据多维度安全线筛选无效目标。
    依据：
      1. Area (面积) >= 128px^2
      2. MinSide (短边) >= 8px
      3. AspectRatio (比例) <= 6:1

    返回字典包含：
      - scanned_files: 扫描的文件数
      - total: 总目标数
      - small: 小目标数
      - modified_files: 修改的文件数
      - targets: 所有目标的详细信息列表
      - deleted: 被删除/标记的目标详细信息列表
    """
    min_area = config.get("min_area_threshold", 128)
    min_side = config.get("min_single_side", 8)
    max_ratio = config.get("max_aspect_ratio", 6.0)
    total = 0
    small = 0
    modified_files = 0
    scanned_files = 0

    all_targets = []     # 所有目标详细信息
    deleted_targets = [] # 被删除/标记的小目标信息

    for txt_file in sorted(label_dir.glob("*.txt")):
        # 跳过 classes.txt
        if txt_file.name == "classes.txt":
            continue

        scanned_files += 1
        logger.debug(f"扫描文件: {txt_file.name}")

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        keep_lines = []
        file_changed = False

        for line_idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                keep_lines.append(line)
                continue

            # 归一化宽高 × 训练分辨率 = 实际像素大小
            pixel_w = w * train_res
            pixel_h = h * train_res

            # 依据专业安全基准进行判定
            area = pixel_w * pixel_h
            short_side = min(pixel_w, pixel_h)
            long_side = max(pixel_w, pixel_h)
            aspect_ratio = long_side / max(short_side, 1e-6)

            # 判定原因汇总
            reasons = []
            if area < min_area:
                reasons.append(f"面积不足({area:.0f}<{min_area})")
            if short_side < min_side:
                reasons.append(f"单边太细({short_side:.1f}<{min_side})")
            if aspect_ratio > max_ratio:
                reasons.append(f"比例畸形({aspect_ratio:.1f}>{max_ratio})")

            is_unsafe = len(reasons) > 0
            reason_str = "|".join(reasons) if is_unsafe else "安全"

            total += 1

            # 记录每个目标的详细信息
            target_info = {
                "file": txt_file.name,
                "line": line_idx,
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
                "norm_w": w,
                "norm_h": h,
                "pixel_w": round(pixel_w, 2),
                "pixel_h": round(pixel_h, 2),
                "area": round(area, 2),
                "aspect_ratio": round(aspect_ratio, 2),
                "is_small": is_unsafe,
                "reason": reason_str,
            }
            all_targets.append(target_info)

            if is_unsafe:
                small += 1
                file_changed = True
                deleted_targets.append(target_info)
                logger.warning(
                    f"亚健康目标 → {txt_file.name}:L{line_idx}  "
                    f"类别={class_id}  原因={reason_str}"
                )
                # 仅统计模式下仍保留该行（不写回）
                if not delete_mode:
                    keep_lines.append(line)
            else:
                keep_lines.append(line)

        # 删除模式下，有变动才写回
        if delete_mode and file_changed:
            # 在删除前自动备份原始文件
            shutil.copy2(txt_file, txt_file.with_suffix('.txt.bak'))
            with open(txt_file, "w", encoding="utf-8") as f:
                f.writelines(keep_lines)
            modified_files += 1
            logger.info(f"  [已修改] {txt_file.name} (备份 → {txt_file.name}.bak)")

    return {
        "scanned_files": scanned_files,
        "total": total,
        "small": small,
        "modified_files": modified_files,
        "targets": all_targets,
        "deleted": deleted_targets,
    }


# ======================== 可视化输出 ========================
def generate_distribution_plot(
    all_targets: list[dict],
    min_threshold: int,
    train_res: int,
    output_dir: Path,
    dataset_name: str = "",
):
    """
    生成小目标分布可视化图：左侧散点图 + 右侧直方图。
    使用高对比度配色方案，清晰区分小目标与正常目标。
    """
    if not HAS_MATPLOTLIB:
        print("[提示] 未安装 matplotlib，跳过可视化图表生成。请运行: pip install matplotlib")
        return

    if not all_targets:
        print("[提示] 无目标数据，跳过可视化图表生成。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 分离正常目标和小目标
    normal_pw = [t["pixel_w"] for t in all_targets if not t["is_small"]]
    normal_ph = [t["pixel_h"] for t in all_targets if not t["is_small"]]
    small_pw = [t["pixel_w"] for t in all_targets if t["is_small"]]
    small_ph = [t["pixel_h"] for t in all_targets if t["is_small"]]

    # ---- 自动寻找系统支持的中文核心字体 ----
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
    chosen_font = None
    for f in font_list:
        try:
            from matplotlib.font_manager import FontProperties
            if any(f.lower() in font.name.lower() for font in matplotlib.font_manager.fontManager.ttflist):
                chosen_font = f
                break
        except:
            continue
    
    if chosen_font:
        plt.rcParams['font.sans-serif'] = [chosen_font]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框
    
    # ---- 图表样式：深色主题 + 高对比度 ----
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.5,
        "font.size": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    title_suffix = f" — {dataset_name}" if dataset_name else ""

    # ========== 左图：散点图 ==========
    ax1 = axes[0]

    # 先画正常目标（绿色），再画小目标（红色），确保小目标在上层
    if normal_pw:
        ax1.scatter(
            normal_pw, normal_ph,
            c="#00e676", alpha=0.5, s=12, edgecolors="none",
            label=f"正常目标 ({len(normal_pw)})",
            zorder=2,
        )
    if small_pw:
        ax1.scatter(
            small_pw, small_ph,
            c="#ff1744", alpha=0.7, s=18, edgecolors="none",
            label=f"小目标 ({len(small_pw)})",
            zorder=3,
        )

    # 阈值区域：用半透明红色矩形标记小面积区（作为示意）
    # 这里的 min_threshold 取单边最小值用于绘图参考
    min_side = min_threshold
    danger_rect = Rectangle(
        (0, 0), min_side, min_side,
        linewidth=0, facecolor="#ff1744", alpha=0.08, zorder=1,
    )
    ax1.add_patch(danger_rect)

    # 阈值线
    ax1.axvline(x=min_side, color="#ff6e40", linewidth=1.5, linestyle="--", alpha=0.8, label=f"单边线 ({min_side}px)")
    ax1.axhline(y=min_side, color="#ff6e40", linewidth=1.5, linestyle="--", alpha=0.8)

    ax1.set_xlabel("像素宽度 (px)", fontsize=12)
    ax1.set_ylabel("像素高度 (px)", fontsize=12)
    ax1.set_title(f"目标尺寸分布 (散点图){title_suffix}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # ========== 右图：直方图（面积分布） ==========
    ax2 = axes[1]

    normal_areas = [t["pixel_w"] * t["pixel_h"] for t in all_targets if not t["is_small"]]
    small_areas = [t["pixel_w"] * t["pixel_h"] for t in all_targets if t["is_small"]]

    # 确定合理的 bin 范围
    all_areas = normal_areas + small_areas
    if all_areas:
        max_area = min(max(all_areas), train_res * train_res * 0.25)  # 限制范围避免极端值
        bins = 50

        if normal_areas:
            ax2.hist(
                normal_areas, bins=bins, range=(0, max_area),
                color="#00e676", alpha=0.6, edgecolor="#1a1a1a",
                label=f"正常目标 ({len(normal_areas)})",
            )
        if small_areas:
            ax2.hist(
                small_areas, bins=bins, range=(0, max_area),
                color="#ff1744", alpha=0.7, edgecolor="#1a1a1a",
                label=f"小目标 ({len(small_areas)})",
            )

        # 阈值线：面积 = threshold^2
        threshold_area = min_threshold * min_threshold
        ax2.axvline(
            x=threshold_area, color="#ff6e40", linewidth=1.5, linestyle="--",
            alpha=0.8, label=f"阈值面积 ({threshold_area}px²)"
        )

    ax2.set_xlabel("目标面积 (px²)", fontsize=12)
    ax2.set_ylabel("数量", fontsize=12)
    ax2.set_title(f"目标面积分布 (直方图){title_suffix}", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)

    # 保存
    safe_name = dataset_name.replace("/", "_").replace("\\", "_").replace(" ", "_") if dataset_name else "all"
    output_file = output_dir / f"target_distribution_{safe_name}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [图表] 分布图已保存: {output_file}")


# ======================== 统计报告 ========================
def generate_text_report(
    results_by_dir: dict,
    config: dict,
    delete_mode: bool,
    output_dir: Path,
    dataset_root: str,
):
    """
    生成结构化文本统计报告。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "cleaning_report.txt"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_text = "分析 + 清洗（已修改文件）" if delete_mode else "仅统计分析（未修改文件）"

    lines = []
    lines.append("=" * 60)
    lines.append("  YOLO 数据集小目标分析与清洗报告")
    lines.append("=" * 60)
    lines.append(f"  生成时间    : {now}")
    lines.append(f"  数据集根目录: {dataset_root}")
    lines.append(f"  执行模式    : {mode_text}")
    lines.append(f"  训练分辨率  : {config['yolo_train_res']}×{config['yolo_train_res']}")
    lines.append(f"  安全阈值    : 面积>{config['min_area_threshold']} | 短边>{config['min_single_side']} | 比例<{config['max_aspect_ratio']}:1")
    lines.append("=" * 60)

    grand_scanned = 0
    grand_total = 0
    grand_small = 0
    grand_modified = 0

    for dir_name, result in results_by_dir.items():
        scanned = result["scanned_files"]
        total = result["total"]
        small_count = result["small"]
        modified = result["modified_files"]
        ratio = (small_count / total * 100) if total > 0 else 0.0

        lines.append(f"\n┌─ 目录: {dir_name}")
        lines.append(f"│  文件数      : {scanned}")
        lines.append(f"│  总目标数    : {total}")
        lines.append(f"│  小目标数    : {small_count}  ({ratio:.2f}%)")
        if delete_mode:
            lines.append(f"│  已修改文件  : {modified}")
        lines.append(f"└{'─' * 40}")

        grand_scanned += scanned
        grand_total += total
        grand_small += small_count
        grand_modified += modified

    # 汇总
    grand_ratio = (grand_small / grand_total * 100) if grand_total > 0 else 0.0
    lines.append("\n" + "=" * 60)
    lines.append("  汇总统计")
    lines.append("=" * 60)
    lines.append(f"  扫描文件总数 : {grand_scanned}")
    lines.append(f"  目标总数     : {grand_total}")
    lines.append(f"  小目标总数   : {grand_small}  ({grand_ratio:.2f}%)")
    if delete_mode:
        lines.append(f"  已修改文件数 : {grand_modified}")
    lines.append("=" * 60)

    # 写入文件
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [报告] 统计报告已保存: {report_file}")


# ======================== CSV 清洗明细 ========================
def generate_csv_report(all_deleted: list[dict], output_dir: Path):
    """
    生成 CSV 格式的删除/标记目标明细表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "deleted_targets.csv"

    headers = ["文件名", "行号", "类别ID", "像素宽", "像素高", "面积", "长宽比", "清洗原因", "状态"]

    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t in all_deleted:
            writer.writerow([
                t["file"],
                t["line"],
                t["class_id"],
                t["pixel_w"],
                t["pixel_h"],
                t["area"],
                t["aspect_ratio"],
                t["reason"],
                "已删除" if t.get("deleted", True) else "仅标记",
            ])

    print(f"  [报告] 清洗明细已保存: {csv_file}  ({len(all_deleted)} 条记录)")


# ======================== 主入口 ========================
def main():
    parser = argparse.ArgumentParser(
        description="YOLO 数据集小目标分析与清洗工具（增强版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d", action="store_true",
        help="启用删除模式：直接从标签文件中移除小目标标注行",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="指定 YAML 配置文件路径",
    )
    parser.add_argument(
        "--datasets", nargs="+", type=str, default=None,
        help="指定一个或多个数据集根目录路径",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="指定报告输出目录 (默认: ./reports)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="跳过生成可视化图表",
    )
    args = parser.parse_args()

    # ---------- 加载配置 ----------
    config = load_config(args.config)
    delete_mode = args.d
    train_res = config["yolo_train_res"]
    label_dirs = config["label_dirs"]

    # 确定数据集根目录列表
    if args.datasets:
        dataset_roots = [Path(ds) for ds in args.datasets]
    elif "datasets" in config:
        dataset_roots = [Path(ds) for ds in config["datasets"]]
    else:
        dataset_roots = [Path(__file__).parent]

    # 确定输出目录
    output_dir = Path(args.output) if args.output else (dataset_roots[0] / "reports")

    # ---------- 初始化日志 ----------
    logger = setup_logging(output_dir)

    mode_text = "分析 + 清洗（将修改文件）" if delete_mode else "仅统计分析（不修改文件）"
    logger.info(f"执行模式  : {mode_text}")
    logger.info(f"训练分辨率: {train_res}×{train_res}")
    logger.info(f"安全红线  : 面积>{config['min_area_threshold']} | 短边>{config['min_single_side']} | 比例<{config['max_aspect_ratio']}")
    logger.info(f"数据集数量: {len(dataset_roots)}")
    logger.info("=" * 50)

    # ---------- 逐数据集处理 ----------
    global_all_targets = []
    global_all_deleted = []
    global_results = {}

    for ds_idx, ds_root in enumerate(dataset_roots, start=1):
        ds_name = str(ds_root)
        logger.info(f"\n{'━' * 50}")
        logger.info(f"数据集 [{ds_idx}/{len(dataset_roots)}]: {ds_name}")
        logger.info(f"{'━' * 50}")

        if not ds_root.exists():
            logger.warning(f"[跳过] 数据集目录不存在: {ds_name}")
            continue

        ds_results = {}
        ds_targets = []
        ds_deleted = []

        for sub in label_dirs:
            label_path = ds_root / sub
            if not label_path.exists():
                logger.info(f"  [跳过] 目录不存在: {sub}")
                continue

            logger.info(f"\n  正在扫描: {sub} ...")
            result = analyze_labels(label_path, delete_mode, train_res, config, logger)

            ratio = (result["small"] / result["total"] * 100) if result["total"] > 0 else 0.0

            logger.info(f"    文件数  : {result['scanned_files']}")
            logger.info(f"    总目标  : {result['total']}")
            logger.info(f"    小目标  : {result['small']}  ({ratio:.2f}%)")
            if delete_mode:
                logger.info(f"    已修改  : {result['modified_files']} 个文件")

            dir_key = f"{ds_name}/{sub}"
            ds_results[dir_key] = result
            ds_targets.extend(result["targets"])
            ds_deleted.extend(result["deleted"])

        # 为每个数据集生成可视化（如果未禁用）
        if not args.no_plot and ds_targets:
            generate_distribution_plot(
                ds_targets, config["min_single_side"], train_res, output_dir,
                dataset_name=Path(ds_name).name,
            )

        global_results.update(ds_results)
        global_all_targets.extend(ds_targets)
        global_all_deleted.extend(ds_deleted)

    # ---------- 汇总统计输出 ----------
    grand_scanned = sum(r["scanned_files"] for r in global_results.values())
    grand_total = sum(r["total"] for r in global_results.values())
    grand_small = sum(r["small"] for r in global_results.values())
    grand_modified = sum(r["modified_files"] for r in global_results.values())
    grand_ratio = (grand_small / grand_total * 100) if grand_total > 0 else 0.0

    logger.info(f"\n{'=' * 50}")
    logger.info("汇总统计:")
    logger.info(f"  扫描文件总数 : {grand_scanned}")
    logger.info(f"  目标总数     : {grand_total}")
    logger.info(f"  小目标总数   : {grand_small}  ({grand_ratio:.2f}%)")
    if delete_mode:
        logger.info(f"  已修改文件数 : {grand_modified}")
        logger.info("  状态: [OK] 标签文件已直接修改完成")
    else:
        logger.info("  状态: [INFO] 未修改任何文件（加 -d 参数可执行删除）")
    logger.info("=" * 50)

    # ---------- 生成报告 ----------
    logger.info(f"\n正在生成报告 → {output_dir}")

    # 如果处理了多个数据集，额外生成一张总体分布图
    if not args.no_plot and len(dataset_roots) > 1 and global_all_targets:
        generate_distribution_plot(
            global_all_targets, config["min_single_side"], train_res, output_dir,
            dataset_name="汇总",
        )

    # 统计报告
    generate_text_report(
        global_results, config, delete_mode, output_dir,
        dataset_root=", ".join(str(r) for r in dataset_roots),
    )

    # CSV 清洗明细
    if global_all_deleted:
        generate_csv_report(global_all_deleted, output_dir)
    else:
        logger.info("  [提示] 无小目标记录，跳过 CSV 明细生成。")

    logger.info(f"\n全部完成！报告目录: {output_dir}")


if __name__ == "__main__":
    main()
