# ProMoAI 实验脚本

这些文件是本项目基于 Plant DNA LLMs 预训练模型编写的 DSS3 专用实验代码，不是上游仓库的完整副本。模型来源和许可证见 `../upstream.md`。

## 推荐顺序

1. `build_dss3_chromosome_holdout_v4_gc_matched.py`：从获准使用的 DSS3 FASTA/GFF 构建 GC 匹配、染色体留出的数据集。
2. `train_dss3_bpe_1000bp_head_only.py`：冻结 BPE 骨干，仅训练分类头。
3. `evaluate_dss3_bpe_1000bp.py`：在固定测试染色体上评估 BPE 模型。
4. `train_dss3_6mer_1000bp_head_only_v4_gc.py`：冻结 6-mer 骨干，仅训练分类头。
5. `evaluate_dss3_6mer_1000bp_v4_gc.py`：评估 6-mer 模型。
6. `analyze_dss3_v4_threshold_sensitivity.py`：对固定预测分数做阈值敏感性分析。

先执行各脚本的 `--help` 并显式填写项目路径。数据、预训练模型、微调权重和逐样本预测均不随 Git 仓库分发；小型配置和汇总指标位于 `../results/`。

## 可复现性要求

- 保持 seed、数据划分和测试集不变。
- 只用开发集选择检查点和阈值，测试集只用于最终报告。
- 记录预训练模型版本、输入数据 SHA256、依赖版本及 GPU/CUDA 环境。
- 不要把本机绝对路径、密钥、模型权重或原始受限数据提交到 GitHub。
