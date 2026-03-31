# 🚀 YOLO Dataset Cleaner (Expert Pro)

专业的 YOLO 格式数据集清洗、深度分析与质量审计工具。专为工业级计算机视觉项目设计，旨在剔除亚健康小目标（Invalid Targets），从标注源头提升模型 mAP。

---

## 🌟 核心专家特性

- **✅ 类别敏感阈值**: 支持为不同 `class_id` 设置独立过滤红线。例如：对“行人”类别保持高敏感，对“背景杂物”类别执行严苛过滤。
- **✅ 分类健康度统计**: 自动生成包含类别名称、总数、亚健康数及健康率的专业报表，数据质量一目了然。
- **✅ 视觉抽样审计**: 随机抽取被删除的目标并生成 `audit_samples` 预览图，带红框标注及原因说明，解决“黑盒清洗”痛点。
- **✅ 工业级并行性能**: 基于 `ProcessPoolExecutor` 的多进程架构，扫描速度提升 4-8 倍。
- **✅ 精确像素分析**: 支持通过 `Pillow` 读取原图真实分辨率，消除坐标归一化带来的计算偏差。

---

## 🛠️ 安装依赖

```bash
pip install numpy pillow tqdm matplotlib pyyaml
```

---

## 📖 使用实例 (Best Practices)

### 场景 A：安全审计模式 (分析但不修改)
适合在正式清洗前，评估当前数据集的质量现状。
```bash
python Yolo_clear.py \
  --config config_example.yaml \
  --datasets "./data/train" \
  --output "./audit_report_v1" \
  --precise --workers 8
```

### 场景 B：专业清洗模式 (执行删除 + 视觉抽样)
执行清洗并生成 50 张审计图用于人工抽检。
```bash
python Yolo_clear.py \
  -d --config config_example.yaml \
  --datasets "./data/train" \
  --garbage "./trash_bin" \
  --audit-n 50 --workers 8
```

### 场景 C：极致清理模式 (删除标签 + 同步删除空标注图片)
```bash
python Yolo_clear.py -d --clean-img --precise
```

---

## 📝 配置文件 (`config.yaml`) 详解

```yaml
# --- 基础过滤阈值 (默认值) ---
min_area_threshold: 128    # 最小面积 (训练分辨率下的像素平方)
min_single_side: 8         # 最小短边 (像素)
max_aspect_ratio: 6.0      # 最大长宽比 (如 6:1)

# --- 类别敏感过滤 (高级) ---
class_thresholds:
  '0': { min_area: 50, min_side: 5 } # 对 ID 0 放宽限制 (可能为极小目标)
  '7': { min_area: 256, max_ratio: 3.0 } # 对 ID 7 严苛限制 (可能为误标)

# --- 执行设置 ---
yolo_train_res: 640         # 训练阶段的目标分辨率
use_precise_resolution: true # 自动读取原图分辨率 (推荐使用)
max_workers: 8               # 并行进程数
audit_n: 20                  # 每次清洗随机生成的复核图数量
```

---

## 📊 报告产物说明

执行完成后，`output` 目录下将生成：
1. `audit_samples/`: **[关键]** 包含带红框标注的采样图，用于人工确认清洗策略是否过于激进。
2. `cleaning_report.txt`: 包含全局及分类统计的纯文本报告。
3. `target_distribution.png`: 目标尺寸分布可视化图表。
4. `deleted_targets.csv`: 所有被剔除目标的详细坐标与原因清单。

---
*Powered by Antigravity AI Engineering Tools*
