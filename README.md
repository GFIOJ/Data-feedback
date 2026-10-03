# 复杂堆场多无人机巡检与数据回传仿真

## 1. 项目状态

本项目研究固定巡检轨迹下的多无人机数据回传。第一阶段生成复杂集装箱堆场、侧面巡检目标、连续扫描任务以及无碰撞带时间戳轨迹；第二阶段把轨迹与首次有效采集事件作为固定输入，逐时隙生成无线信道并模拟数据在无人机队列、中继节点和中心基站之间的传输。

当前已完成并验证：

- 堆场、轨迹、采集事件和统一时间轴的生成与导出；
- 基于距离与扫描角度的连续巡检判定；
- LoS/NLoS遮挡、路径损耗、阴影衰落、莱斯/瑞利小尺度衰落、同频干扰、SINR和速率；
- 给定调度下的同步队列更新，支持直接回传、人工指定中继、分片汇合和部分传输；
- 速率贪心、轮询、最老数据优先三种直接回传基线；
- 容量上界诊断、输入校验、40项测试和12项独立审计。

项目尚未实现论文拟研究的协同中继优化算法。现有多跳能力是仿真执行接口和手工验证案例，不能表述为已经完成中继选择或路由优化。下一阶段方案见 [`paper2_algorithms/algorithm_design.md`](paper2_algorithms/algorithm_design.md)。

## 2. 当前冻结基准

当前唯一推荐通信输入为 `paper2_baselines/direct_return/input/input_ref_5to10mbit_v10/`，冻结记录为 `paper2_baselines/direct_return/verification/baseline_frozen.json`，输入和源代码SHA-256分别保存在冻结记录及同目录的 `source_sha256.json`。

| 项目 | 当前值 |
|---|---:|
| 无人机 / 指定巡检目标 | 20 / 8000 |
| 轨迹速度 | 10 m/s |
| 场景 / 通信时隙 | 2000 m × 2000 m / 1 s |
| 轨迹结束时间 / 时隙数 | 925.3682229706367 s / 926 |
| 每目标数据量 | 离散均匀抽样并保存，625,000～1,250,000 byte，即5～10 Mbit |
| 数据量种子 / 信道种子 | 2027 / 2026 |
| 实际总数据量 | 7,504,298,438 byte = 60,034,387,504 bit |
| 子信道 | 4个，每个5,000,000 Hz，总带宽20,000,000 Hz |
| 活动发送功率 | 每架活动无人机固定0.1 W，不再除以子信道数 |
| 噪声功率谱密度 | 4×10⁻²¹ W/Hz；单子信道噪声为2×10⁻¹⁴ W |
| 单机缓存 | 1,000,000,000 byte = 8,000,000,000 bit |
| 基站天线 / 起降点 | (1000,1000,30) m / (1000,1000,15) m |

十进制单位统一采用 `1 MB = 1,000,000 byte`、`1 byte = 8 bit`、`1 Mbit = 1,000,000 bit`。每个目标只产生一个完整数据单元，不再叠加RFID、图像或视频大小。

参数来源分为三类：

- **文献参考与场景适配**：2025年文献给出1 MB传输包和5 MHz高频数据链路，2022年文献给出每链路5 MHz、20～30 dBm和1～10 Mbit包长。本项目采用每子信道5 MHz，并把5～10 Mbit映射为每目标完整数据单元。映射方式和子区间选择属于场景适配。
- **项目设定**：20架无人机、8000个目标、10 m/s、4个子信道、总带宽20 MHz、固定0.1 W、1 GB缓存、中心基站位置、场景和数据/信道种子。不能把节点度上限4当作子信道数4的来源。
- **待实测校准**：2.4 GHz、参考距离1 m、参考损耗40 dB、路径损耗指数2.2、遮挡附加损耗20 dB、阴影标准差4 dB、LoS莱斯因子4（线性值）、有效距离1～5000 m。这些值是当前仿真参数，不是已经验证的真实堆场空地信道测量结果。

第一版固定 `p_i(t)=P=0.1 W`，功率不参与优化。

## 3. 时间、在线和队列规则

所有无人机使用从0 s开始的同一全局时间轴，保留错时起飞、避碰等待和各自返航时刻。无人机只在 `[takeoff_time_s, return_time_s)` 内通信在线；只有整个时隙均在线的节点才有 `scheduling_eligible=True`，边界时隙不能按完整时隙容量参与调度。基站在整个通信仿真期间在线。

零起始时隙 `k` 表示 `[kΔt,(k+1)Δt)`，论文一起始时隙编号为 `k+1`。恰好位于右边界 `(k+1)Δt` 的采集事件属于下一时隙。首次满足8 m距离和120°全扫描角判据时生成一个完整数据单元；事件在采集所在时隙末入队，最早从下一时隙发送。

每个时隙按固定顺序执行：

1. 读取时隙开始队列和缓存信道快照；
2. 根据时隙开始状态产生调度；
3. 由 `evaluate_schedule` 计算干扰、SINR和速率；
4. 只根据时隙开始存量确定各数据单元传输量；
5. 时隙末统一扣除发送量、加入中继接收量或基站累计量；
6. 加入本时隙采集完成的数据；
7. 判断基站是否完整接收每个数据单元。

新采集或新接收的数据不能在同一时隙继续转发。返航后的残留数据保留但不再发送，不自动交付、删除或通过未建模地面链路上传。缓存超限时保留全部数据，记录违规并令 `performance_metrics_valid=false`；该运行不能作为可行调度结果。

完整接收时刻取完成时隙末。首次发送时刻取首次获得正传输量的时隙起点。首次发送前等待包含FIFO前序数据和未被调度造成的等待；首次发送后至完成时间仍可能包含后续排队与等待，不能称为纯物理传输时长。

## 4. 信道模型

每个时隙生成一次只读 `ChannelSnapshot`，同一时隙比较不同调度时复用同一快照。位置、在线状态和全部信道矩阵均为不共享可写输入内存的只读数组；调用方需要修改时必须显式 `.copy()`。

节点间三维线段与原始聚合障碍物边界相交时为NLoS，相切按遮挡处理；越过障碍物顶部不遮挡。大尺度模型为：

```text
L_ij = L0 + 10 n log10(d_ij/d0) + s_ij A_obs + X_ij
beta_ij = 10^(-L_ij/10)
```

归一化莱斯系数为：

```text
h_ij = sqrt(K/(K+1)) exp(j phi) + sqrt(1/(K+1)) w
w = (Z1 + j Z2)/sqrt(2)
g_ij = beta_ij |h_ij|^2
```

LoS取线性 `K=4`，NLoS取 `K=0`。接收功率直接使用 `p_i g_ij`，不再对 `g_ij` 平方。每子信道速率为：

```text
I_ij^f = sum_{m != i, same f} p_m g_mj
SINR_ij^f = p_i g_ij / (N0 b + I_ij^f)
R_ij^f = b log2(1 + SINR_ij^f)
R_ij = sum_f x_ij^f R_ij^f
```

第一版补充假设为：阴影和散射项在不同时隙独立抽样；同一时隙正反向链路理想互易；直射相位由载频和距离计算；各子信道共享该时隙平坦衰落增益。项目没有实现时间相关阴影、多普勒相关衰落或信道预测模型。

## 5. 数据流

```mermaid
flowchart LR
  A[堆场、障碍物与侧面目标] --> B[无碰撞带时间戳轨迹]
  B --> C[首次有效采集事件]
  C --> D[固定Paper 2输入]
  D --> E[逐时隙信道快照]
  E --> F[调度与子信道安排]
  F --> G[源端/中继队列同步更新]
  G --> H[基站累计完整接收]
  H --> I[交付率、时延、积压、缓存与守恒指标]
```

## 6. 目录结构

```text
code/
├─ README.md
├─ requirements.txt
├─ paper1/                  # 堆场、扫描任务、轨迹与Paper 2输入生成
│  ├─ run_pipeline.py
│  ├─ outputs/
│  │  └─ paper2_input_10mps/ # 10 m/s原始轨迹和环境来源
│  ├─ figures/              # 当前Paper 1图，因可能被论文引用而保留
│  └─ tests/
├─ paper2_model/            # 所有算法共享的信道、队列、缓存和仿真模型
│  └─ tests/
├─ paper2_baselines/
│  └─ direct_return/        # 直接回传基准代码、输入、结果和验证
│     ├─ prepare_input.py
│     ├─ run_comparison.py
│     ├─ input/input_ref_5to10mbit_v10/
│     ├─ results/
│     ├─ tests/
│     └─ verification/
├─ paper2_algorithms/
│  ├─ README.md             # 新算法目录规范
│  └─ algorithm_design.md   # 下一阶段设计草案
└─ .runtime/               # 随项目提供的Python运行环境，不逐项列第三方文件
```

后续每个新算法统一建立在 `paper2_algorithms/<algorithm_name>/`，目录内放README、`run.py`、可选专属配置和 `results/`。新算法共享 `paper2_model`，不复制公共模型，也不向 `paper2_model/` 或 `paper2_baselines/` 添加实验脚本。

## 7. 项目代码、配置和文档职责

### 根目录与文档

| 相对路径 | 类型与职责 | 主要输入 → 输出 | 通常是否修改 |
|---|---|---|---|
| `README.md` | 当前项目、数据和运行说明 | 项目实际状态 → 研究者使用说明 | 规则、入口或结果变化时更新 |
| `requirements.txt` | 最小依赖声明 | pip → NumPy/SciPy/Matplotlib | 依赖真正变化时修改 |
| `.gitignore` | 忽略运行环境、缓存和可生成输出 | Git工作区规则 | 通常不改 |
| `paper2_algorithms/README.md` | 新算法目录和共享模块规则 | 目录约定 → 后续算法开发规范 | 新增公共约定时修改 |
| `paper2_algorithms/algorithm_design.md` | 下一阶段协同回传设计草案 | 已验证仿真约束 → 待审阅求解流程 | 算法方案评审时修改 |

### `paper1/`

| 相对路径 | 类型与职责 | 主要输入 → 输出 | 通常是否修改 |
|---|---|---|---|
| `paper1/__init__.py` | 包标识 | 无 | 否 |
| `paper1/run_pipeline.py` | Paper 1总入口 | `CONFIG` → `outputs/paper2_input_10mps/`及可视化图 | 重新生成场景时修改配置；日常通信实验不改 |
| `paper1/common.py` | 被调用模块：结构体、参数、障碍物线段相交和数值辅助 | 数组/配置 → 通用计算结果 | 否 |
| `paper1/plot_environment.py` | 被调用模块：生成聚合集装箱堆场和侧面目标 | 场景参数 → 障碍物、目标、场景元数据 | 修改场景模型时 |
| `paper1/plot_environment_multiscale.py` | 辅助模块：多尺度场景生成兼容入口 | 多尺度参数 → 场景 | 一般不改 |
| `paper1/rfid_module.py` | 被调用模块：仅按8 m距离和120°扫描角判断读取 | 读卡器/目标几何 → 稀疏覆盖矩阵 | 判据变化时 |
| `paper1/game_hover_solver.py` | 被调用模块：连续扫描候选、覆盖和任务筛选 | 场景与覆盖矩阵 → 扫描任务 | 规划方法研究时 |
| `paper1/legacy_planner.py` | 被调用兼容模块：历史随机序列与规划兼容实现 | 种子/任务 → 可复现序列与兼容规划 | 一般不改；Stage 1和测试仍引用 |
| `paper1/uav_mission_planner.py` | 核心被调用模块：任务分配、路径库、避障避碰和时序 | 扫描任务/无人机参数 → 多机带时间轨迹 | Paper 1规划研究时 |
| `paper1/main_stage1_hover_planner.py` | Stage 1入口 | 场景/RFID参数 → 覆盖任务与扫描段 | 独立调试Stage 1时 |
| `paper1/main_stage2_uav_planner.py` | Stage 2入口 | Stage 1结果/飞行参数 → 无碰撞任务轨迹 | 独立调试Stage 2时 |
| `paper1/paper2_export.py` | 被调用模块：导出并完整验证Paper 2输入 | 场景与轨迹 → CSV/JSON/NPZ | 导出格式变化时 |
| `paper1/paper2_dataset.py` | 公共数据接口：读取轨迹、状态和事件，重算数据量派生统计 | Paper 2输入目录 → `Paper2Dataset` | 数据格式变化时 |
| `paper1/visualization.py` | 辅助模块：场景和轨迹绘图 | Stage 1/2结果 → Matplotlib图 | 绘图需求变化时 |
| `paper1/tests/__init__.py` | 测试包标识 | 无 | 否 |
| `paper1/tests/test_python.py` | 15项Paper 1与导出回归测试 | 合成场景 → unittest结果 | 功能变化时同步测试 |

### `paper2_model/`

| 相对路径 | 类型与职责 | 主要输入 → 输出 | 通常是否修改 |
|---|---|---|---|
| `paper2_model/__init__.py` | 公共模型包接口 | 无 | 否 |
| `paper2_model/channel.py` | 只读信道快照、遮挡、衰落、干扰、SINR和速率 | 数据集/配置/调度 → 快照与链路速率 | 信道公式确认后不改 |
| `paper2_model/return_simulation.py` | DataUnit、队列、守恒、缓存、给定调度执行和结果写出 | 固定输入/快照/安排 → 逐单元和汇总结果 | 新算法调用，不复制 |
| `paper2_model/experiment_config.py` | 当前共享通信与缓存配置及输入一致性辅助 | 无 → 已验证配置对象 | 公共参数确认后不改 |
| `paper2_model/capacity_diagnostic.py` | 基站接收容量必要条件上界与探索负载表 | 固定输入/信道配置 → 上界报告 | 通常不改 |
| `paper2_model/demo_channel.py` | 确定性手算案例和实际堆场同/异频演示 | 合成或当前输入 → 控制台中间量 | 一般不改 |
| `paper2_model/tests/test_channel.py` | 8项信道、容量、快照与调度合法性测试 | 合成快照 → unittest结果 | 信道接口变化时 |
| `paper2_model/tests/test_return_simulation.py` | 15项队列、时序、中继、分片、调度和输入校验测试 | 合成数据 → unittest结果 | 队列规则变化时 |

### `paper2_baselines/`

| 相对路径 | 类型与职责 | 主要输入 → 输出 | 通常是否修改 |
|---|---|---|---|
| `paper2_baselines/__init__.py` | 基准实验包标识 | 无 | 否 |
| `paper2_baselines/direct_return/prepare_input.py` | 由10 m/s源输入准备并验证5～10 Mbit固定输入 | Paper 1输出 → `input/input_ref_5to10mbit_v10/` | 仅重建输入时使用 |
| `paper2_baselines/direct_return/run_comparison.py` | 速率贪心、轮询、最老数据优先三策略入口 | 固定输入/冻结基准 → `results/` | 直接回传基准变化时 |
| `paper2_baselines/direct_return/tests/test_experiment_config.py` | 2项带宽、功率、缓存和数据抽样配置测试 | 当前配置 → unittest结果 | 配置变化时 |
| `paper2_baselines/direct_return/verification/simulator_validation.py` | 正式输入校验并独立重跑速率贪心 | 固定输入/冻结值 → 回归证据 | 校验口径变化时 |
| `paper2_baselines/direct_return/verification/simulator_audit.py` | 项目内12项独立审计 | 合成案例和固定输入 → `audit_results.json` | 不能弱化断言 |
| `paper2_baselines/direct_return/verification/freeze_baseline.py` | 记录环境、配置、结果及输入/源代码SHA-256 | 当前证据 → 两份冻结JSON | 重新正式冻结时运行 |
| `paper2_baselines/direct_return/verification/cleanup_manifest.md` | 历史清理与目录迁移记录 | 依赖审计 → 清理证据 | 再次清理时更新 |

## 8. 输入数据文件

`paper1/outputs/paper2_input_10mps/` 和 `paper2_baselines/direct_return/input/input_ref_5to10mbit_v10/` 具有同一基本结构。前者保留10 m/s源轨迹和环境；后者是通信算法必须读取的冻结输入，并额外包含 `data_unit_sizes.csv`。所有CSV/NPZ中的节点、目标、任务和数据ID均从0开始；论文时隙编号字段从1开始。

| 文件 | 内容、关键字段或数组 | 单位与角色 |
|---|---|---|
| `scenario.json` | 场景、速度、种子、目标/障碍物数量、时隙数、时间规则、数据量分布 | m、s、bit；配置元数据 |
| `environment.npz` | `scene_bounds_m`、原始障碍物三维边界、目标坐标/所属堆垛/侧面法向、扫描段、起降点和基站坐标 | m；信道遮挡与场景恢复的源输入 |
| `trajectories.npz` | `uav_0...uav_19`，每行为 `[x,y,z,time]`；重复坐标且时间增加表示等待 | m、s；固定轨迹源输入 |
| `node_states.npz` | 时隙起止/中点、20机中点位置、在线、整时隙资格、边界时隙、起飞/返航/退出时间 | s、m；信道和调度源输入 |
| `collection_events.csv` | 数据/目标/源机/扫描任务ID，首次读取与完成时间，入队和可发送时隙，数据byte/bit，目标与无人机坐标、扫描窗口 | s、m、byte、bit；每目标一行的源输入 |
| `data_unit_sizes.csv` | `data_id,data_bytes,data_bits` | 当前实验保存的一次抽样结果；各算法共同读取 |
| `slot_arrivals.npz` | 每机/全系统新增、下一时隙可发送和累计数据量，首次读取映射 | byte、bit；由事件表可重新生成 |
| `system_arrivals.csv` | 每时隙新增单元、数据量及累计量 | s、byte、bit；由事件表可重新生成 |
| `uav_metadata.csv` | 每机起飞、返航、通信区间、采集数和生成量 | s、byte、bit；由轨迹和事件可重新生成 |
| `validation.json` | 输入跨文件一致性检查 | 可重新生成的校验报告 |

改变轨迹或飞行时间必须重新计算采集事件、在线状态和逐时隙到达；只改变数据量时可用 `recompute_data_amounts` 保留轨迹和采集时刻，仅重算数据量派生文件。

## 9. 结果文件

统一结果位于 `paper2_baselines/direct_return/results/direct_scheduler_comparison_seed2026_data2027/`。根目录包含 `comparison_summary.json`、`scheduler_comparison.csv`、共同完成集合对照和三张比较图；`rate_greedy/`、`round_robin/`、`oldest_first/`具有相同结构，只是调度规则和结果不同。

| 策略目录内文件 | 内容 |
|---|---|
| `summary.json` | 配置、交付率、bit比例、时延、积压、缓存、容量上界和守恒汇总 |
| `experiment_config.json` | 实际信道、仿真、时域、输入和种子 |
| `data_units.csv` | 每单元原始/已收/剩余量、首次发送、完成时刻、时延和正传输时隙数；未完成时完成字段留空 |
| `uav_buffers.csv` | 每机生成/交付、完整率、条件时延、峰值/最终缓存、服务机会和退出残留 |
| `slot_metrics.csv` | 系统逐时隙生成、发送、基站累计接收、未交付量和总缓存 |
| `slot_uav_metrics.csv` | 每机逐时隙缓存、在线有积压、是否调度、实际发送量和可用直达速率 |
| `residual_locations.csv` | 最终残留数据所在无人机、数据ID和剩余bit |
| `top50_completed_delays.csv` | 已完成数据中时延最大的50项及分解 |
| PNG图 | 累计生成/接收、未交付量、逐机交付率和缓存等 |

**完整单元交付率**以完整收到的单元数除以8000；**bit交付比例**以基站实际收到bit除以总生成bit。部分接收只提高bit比例，不计为完整单元交付。存在未完成单元时，全体平均完成时延为未定义；已完成数据的条件平均时延必须连同交付率和样本数报告。

`paper2_baselines/direct_return/results/history_metadata/`只保留固定103671 byte、固定2 MB及固定1 MB等已淘汰实验的摘要和配置，不是当前运行输入或推荐结果。

## 10. 实际运行命令

以下命令均在 `D:\信息回传\code` 下运行。

### 使用现有输入重跑速率贪心基准和正式输入校验

```powershell
.\.runtime\python\python.exe -m paper2_baselines.direct_return.verification.simulator_validation
```

只读当前输入和冻结比较值；结果写入 `paper2_baselines/direct_return/verification/simulator_validation_seed2026_data2027/`，不会覆盖历史三策略结果。

### 运行三策略比较

```powershell
.\.runtime\python\python.exe -m paper2_baselines.direct_return.run_comparison
```

只读当前输入和冻结基准，重建并覆盖当前统一三策略结果目录。三种策略共享同一个 `ChannelModel`缓存快照。

### 容量诊断和信道演示

```powershell
.\.runtime\python\python.exe -m paper2_model.capacity_diagnostic
.\.runtime\python\python.exe -m paper2_model.demo_channel
```

均只读当前输入并输出到控制台。容量上界忽略数据到达、积压位置、干扰和中继开销，是必要条件诊断，不是实际吞吐量。

### 运行全部测试与项目内独立审计

```powershell
.\.runtime\python\python.exe -m unittest discover -v
.\.runtime\python\python.exe paper2_baselines\direct_return\verification\simulator_audit.py
```

测试会生成可再生字节码缓存；审计写入 `paper2_baselines/direct_return/verification/audit_results.json`，不依赖项目外ChatGPT临时目录。

### 更新冻结记录

```powershell
.\.runtime\python\python.exe -m paper2_baselines.direct_return.verification.freeze_baseline
```

该命令只应在测试、审计、正式输入校验和基准回归全部通过后运行；写入 `paper2_baselines/direct_return/verification/baseline_frozen.json` 和 `source_sha256.json`。

### 必要时重新生成场景、轨迹和当前输入

```powershell
.\.runtime\python\python.exe -m paper1.run_pipeline
.\.runtime\python\python.exe -m paper2_baselines.direct_return.prepare_input
```

`paper1/run_pipeline.py`读取代码内 `CONFIG`，运行Paper 1并写入 `paper1/outputs/paper2_input_10mps/`，会改变源输入和图；不要把它当作只读命令。`prepare_input`在当前实验输入不存在时复制该源输入，按种子2027抽样一次并保存逐单元数据量；输入已存在时会验证并复用已保存大小，不重新抽样。它还会在 `paper2_baselines/direct_return/results/` 创建独立速率贪心输出。重新生成后必须重新运行全部验证并正式冻结。

## 11. 当前三策略结果

| 策略 | 完整单元 | 完整率 | bit交付比例 | 已完成均值 / P95 / 最大时延 (s) | 最终未交付 (bit) | 缓存超限 |
|---|---:|---:|---:|---:|---:|---:|
| 速率贪心 | 8000 | 100% | 100% | 63.584761 / 188.251284 / 298.671086 | 0 | 否 |
| 轮询 | 6066 | 75.825% | 75.891354% | 210.432363 / 399.320625 / 435.898433 | 14,473,477,884.116 | 否 |
| 最老数据优先 | 5288 | 66.100% | 66.102200% | 212.576629 / 368.379319 / 373.230838 | 20,350,336,767.837 | 否 |

轮询和最老数据优先的时延只针对各自已完成样本，不能与速率贪心的8000个完整样本脱离交付率直接比较。三者共同完成集合为5271项；共同集合结果只作补充，因为它存在成功样本选择偏差。当前结果表明在这组固定输入与信道样本中，速率贪心优于两个简单公平性基线；它不证明中继一定能改善性能。

## 12. 验证范围与限制

- 当前40项测试和12项审计覆盖几何遮挡、衰落统计、干扰、快照复现、输入校验、时隙因果、单跳/多跳、分片汇合、缓存违规、离线残留和数据守恒。
- 输入校验要求显式采集时间位于对应半开时隙及源机在线区间。仅合成兼容案例允许精确时间为 `None`，此时以时隙末为代理。
- 信道采用时隙中点位置、块衰落和理想互易；尚无时间相关阴影、多普勒、协议开销、重传、控制信令或误包模型。
- `b log2(1+SINR)`是理想信息论速率，不等同于实际链路吞吐。
- 一次有效首次读取立即产生一个完整数据单元，没有建模拍摄持续时间或多业务类型。
- 容量上界只能证明“超过上界必然不可完成”；需求低于上界不能证明调度可行。
- 缓存超限不丢包，但运行被标为不可行；未完成数据的完成时刻和时延保持为空，不能填0或仿真结束时间。
- 下一阶段应在冻结输入、同一信道样本和现有队列引擎上设计协同中继调度，并与三种直接基线公平比较；当前不承诺中继必然带来改善。
