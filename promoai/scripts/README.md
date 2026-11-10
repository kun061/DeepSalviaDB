# ProMoAI 实验脚本

这些文件是本项目基于 Plant DNA LLMs 预训练模型编写的鼠尾草专用实验代码，不是上游仓库的完整副本。模型来源和许可证见 `../upstream.md`。

## 推荐顺序

1. `build_dss3_chromosome_holdout_v4_gc_matched.py`：从获准使用的FASTA/GFF 构建 GC 匹配、染色体留出的数据集。
2. `train_dss3_bpe_1000bp_head_only.py`：冻结 BPE 骨干，仅训练分类头。
3. `evaluate_dss3_bpe_1000bp.py`：在固定测试染色体上评估 BPE 模型。
4. `train_dss3_6mer_1000bp_head_only_v4_gc.py`：冻结 6-mer 骨干，仅训练分类头。
5. `evaluate_dss3_6mer_1000bp_v4_gc.py`：评估 6-mer 模型。
6. `analyze_dss3_v4_threshold_sensitivity.py`：对固定预测分数做阈值敏感性分析。



