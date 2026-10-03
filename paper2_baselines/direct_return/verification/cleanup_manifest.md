# 项目清理清单（2026-09-28）

清理根目录固定为 `D:\信息回传\code`。本次先检查入口、导入、测试、输入来源和结果读取关系，再执行删除。删除目标均使用绝对路径解析并检查：必须位于项目根目录内，且目标自身及其父路径不能是目录联接或符号链接。

## 分类与保留决定

| 类别 | 路径 | 用途与决定 |
|---|---|---|
| 当前运行必需 | `paper2_model/`、`paper2_baselines/direct_return/` | 公共模型与直接回传基准分离；全部保留 |
| 输入重新生成必需 | `paper1/run_pipeline.py`、`paper1/outputs/paper2_input_10mps/`、`paper2_baselines/direct_return/prepare_input.py` | 生成堆场、轨迹、采集事件和冻结数据量；全部保留 |
| 测试、审计和基准复现必需 | `paper1/tests/`、`paper2_model/tests/`、`paper2_baselines/direct_return/tests/`、`paper2_baselines/direct_return/verification/`、`requirements.txt`、`.runtime/` | 40项测试、12项独立审计、输入校验、基准重跑和环境；全部保留 |
| 当前完整结果 | `paper2_baselines/direct_return/results/direct_scheduler_comparison_seed2026_data2027/` | 三种直接回传策略的统一完整结果，与基准代码同目录；保留 |
| 历史对照记录 | `paper2_baselines/direct_return/results/history_metadata/` | 已淘汰负载实验的摘要、配置和校验；保留 |
| 用户资料或用途不宜推断 | `paper1/figures/` | 可能用于论文插图；保留，不按“可再生成”擅自删除 |

`paper1/outputs/paper2_input_10mps/` 虽含源导出时的数据量字段，但它仍是当前10 m/s轨迹、环境、起降时刻和采集事件的来源，因此保留。`.runtime/` 是随项目提供的Python运行环境；其中的第三方缓存不单独清理。

## 确认删除

| 确切相对路径 | 类型 | 删除理由 | 替代位置或恢复方式 | 实际结果 |
|---|---|---|---|---|
| `__pycache__/` | 可再生缓存 | 根入口字节码缓存，无运行依赖 | Python运行时自动重建 | 已删除 |
| `paper1/__pycache__/` | 可再生缓存 | 第一阶段模块字节码缓存 | Python运行时自动重建 | 已删除 |
| `paper1/tests/__pycache__/` | 可再生缓存 | 第一阶段测试字节码缓存 | 测试运行时自动重建 | 已删除 |
| `paper2/__pycache__/` | 可再生缓存 | 第二阶段模块字节码缓存，含已删除旧模块的残留字节码 | Python运行时自动重建 | 已删除 |
| `paper2/tests/__pycache__/` | 可再生缓存 | 第二阶段测试字节码缓存 | 测试运行时自动重建 | 已删除 |
| `paper2_results/direct_ref_5to10mbit_v10_seed2026_data2027/`（重组前路径） | 重复结果 | 速率贪心结果已完整包含在统一三策略结果中 | 完整结果：`paper2_baselines/direct_return/results/direct_scheduler_comparison_seed2026_data2027/rate_greedy/`；冻结证据：`paper2_baselines/direct_return/verification/` | 已删除 |

删除单策略目录前，现 `paper2_baselines/direct_return/run_comparison.py` 的回归基准读取位置已改为冻结记录，不再依赖该目录。输入生成入口仍可按需重新产生单策略输出，但当前推荐结果以统一三策略目录为准。

## 此前已清理且仅保留摘要

以下大型历史输入或结果已在前一轮确认无当前依赖后删除，其摘要和配置集中在 `paper2_baselines/direct_return/results/history_metadata/`：

- 固定 103671 byte 结果；
- 固定 2 MB 输入与结果；
- 固定 1 MB、5 m/s 输入与结果；
- 固定 1 MB、10 m/s 独立输入与结果；
- 仅服务旧固定1 MB实验的入口及过时测试文件。

## 暂不清理

| 路径 | 暂留原因 |
|---|---|
| `paper1/figures/` | 可能已被论文或用户文档引用，无法从代码调用关系确认 |
| `.runtime/` | 用户明确要求保留，且用于锁定当前可运行Python环境 |
| `paper1/outputs/paper2_input_10mps/` | 当前输入重建所依赖的10 m/s源轨迹与环境 |
| `paper2_model/demo_channel.py` | README中的确定性信道检查和实际堆场信道演示入口 |
| `paper1/legacy_planner.py` | 当前Stage 1随机序列兼容与Paper 1测试仍引用 |
| `paper1/plot_environment_multiscale.py` | Paper 1多尺度场景测试仍引用 |

## 目录重组记录

本轮只移动文件和修正导入/入口，没有改变模型参数或算法：

- `paper2/channel.py`、容量诊断、信道演示和信道测试移动到 `paper2_channel/`；
- 队列仿真器、三种直接基准、当前输入、结果、测试和验证移动到 `paper2_baselines/`；
- `paper2_input/` 移动到 `paper1/generated_input/`，明确它是Paper 1生成的10 m/s源输入；
- `paper2_experiments/input_ref_5to10mbit_v10/` 移动为 `paper2_baselines/input/`；
- `paper2_results/` 移动为 `paper2_baselines/results/`；
- `verification/` 移动为 `paper2_baselines/verification/`；
- 原 `paper2/` 在内容全部归类后删除。

最终结构进一步按共享边界整理：

- `paper1/run_pipeline.py` 与 `paper1/outputs/paper2_input_10mps/` 集中Paper 1入口和输出；
- `paper2_model/` 集中信道、DataUnit、队列、缓存、容量诊断和结果写出等全部公共模型；
- `paper2_baselines/direct_return/` 集中直接回传基准代码、固定输入、结果、测试和验证；
- `paper2_algorithms/` 只用于后续新增算法，每个算法单独建立含代码和 `results/` 的子目录。

## 清理后验证

- 全部测试：40/40通过；
- 项目内独立审计：12/12通过；
- 当前正式输入：8000条事件、60,034,387,504 bit，完整校验通过；
- 速率贪心回归：100%完整交付，平均时延63.58476096944617 s，无缓存超限，数据守恒通过；
- 清理后运行不依赖项目外临时审计脚本或已删除结果目录。
