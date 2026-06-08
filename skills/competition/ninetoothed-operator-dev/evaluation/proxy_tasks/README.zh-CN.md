# proxy_tasks —— 离线评测任务集

[English](README.md) | **中文**

赛题的 8 个隐藏评测任务不可见。proxy 任务集是一组镜像隐藏分布的替身任务,用于在
缺少隐藏任务的情况下提供可重复的离线评测基准:度量 skill 安装前后的增益(A/B),
并为 Stage 3 的 MOO 迭代提供 reward 信号。

## 规模与划分

24 题,覆盖赛题四类,每类 6 题;每类 4 题为 train、2 题为 holdout(合计 train 16、
holdout 8)。

| 类别 | train | holdout |
|---|---|---|
| 逐元素 / 广播 | add, mul_broadcast, relu, gelu | silu, masked_add |
| 归约 / 分块 | sum_last, mean_last, softmax, rms_norm | max_last, l2_norm |
| 布局敏感 | contig_transpose, flip_last, narrow_half, strided_gather | pixel_unshuffle, pixel_shuffle |
| 性能 / 诊断 | softmax_no_maxsub, mean_no_upcast, add_bench_memorybound, inspect_tile_config | noncontig_regression, aot_numwarps_mismatch |

holdout 集刻意采用 train 集没有的算子(如 pixel_unshuffle / pixel_shuffle),通过
holdout 即说明 skill 的增益可泛化到调参集合之外,而非对公开样例过拟合。

## 两种 kind

- **operator**(18 题):要求实现一个 NineToothed 算子。携带 PyTorch 参考实现与
  输入生成器;正确性按 MERE/MARE 与参考对比(阈值见
  `../../scripts/run_correctness_matrix.py`:fp32 1.22e-4、fp16 9.77e-4、
  bf16 7.81e-3)。
- **diagnosis**(6 题):给定一个出错或偏慢的 kernel,要求定位根因并给出最小修复。
  携带 `scenario` 与 `expected_findings` 清单,按命中的 finding 数量打分,需修复的
  另判正确性。

任务的 `prompt` / `scenario` / `expected_findings` 字符串特意保留中文:这是面向
中文赛题的任务内容,非代码。

## 任务来源

- InfiniTensor/ninetoothed 已合并 PR 与 issue 中出现的算子开发场景
- 仓库 `examples/` 与 `ntops` 已有算子
- 为布局敏感与 scatter 覆盖缺口手工补充的用例(公开工作在这两类较弱)

## 文件结构

```
proxy_tasks/
  schema.py       # TaskSpec 定义与校验;randn_inputs 工厂
  elementwise.py  # 6 题
  reduction.py    # 6 题
  layout.py       # 6 题
  perf_diag.py    # 6 题(diagnosis)
  loader.py       # 加载、校验、生成 manifest、可选 torch CPU 自检
  manifest.json   # 由 loader 从模块生成,保证与代码不漂移
  README.md
```

## 使用

```bash
# 校验 schema 与结构不变量(24 题 / 每类 4+2),重建 manifest
python loader.py

# 额外在 CPU 上跑每个 operator 任务的参考实现做数值自检(需 torch)
python loader.py --check
```

模块在未安装 torch 的机器上也可导入(torch 在 reference / make_inputs 内部惰性
导入),因此结构校验与 manifest 生成不依赖 GPU 环境;数值自检在装有 torch 的机器
上运行。

## 与评测闭环的衔接

evaluator(`../skill_eval/`)消费本任务集:对每题在 no-skill 与 v0 两档下让 agent
求解,operator 任务用 MERE/MARE 判正确性、`robust_bench` 测性能并查 reward-hacking,
diagnosis 任务按 `expected_findings` 命中率打分,失败用 `failure_classifier` 归因。
train 集驱动 Stage 3 优化,holdout 集仅用于最终泛化评估。
