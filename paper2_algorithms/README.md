# Paper 2算法实验目录

此目录只保存下一阶段新增算法。`paper2_model/`是所有算法共享的信道、DataUnit、队列、缓存、守恒和结果写出实现；`paper2_baselines/direct_return/`是冻结的直接回传对照。不得把新算法脚本加入这两个公共目录，也不得复制公共模型代码。

每个算法建立独立子目录：

```text
paper2_algorithms/
└─ algorithm_name/
   ├─ README.md
   ├─ run.py
   ├─ config.py       # 只有专属参数时创建
   └─ results/
```

算法的 `run.py` 应读取同一冻结输入 `paper2_baselines/direct_return/input/input_ref_5to10mbit_v10/`，导入 `paper2_model.channel` 与 `paper2_model.return_simulation`，并把输出写入自身 `results/`。不同算法不得重新生成轨迹、采集事件或逐目标数据量。
