# Validation results

本目录只保存适合进入 Git 的小型验证汇总：

- `experiment_metadata.json`：验证设计、时期、样本配对限制、模型和 DEG 数量。
- `summary_by_model.csv`：四种模型在所有代谢物—时期组合上的汇总指标。

逐组合指标、逐折预测和逐基因稳定性文件可由 `analysis/cross_validate.py` 重新生成，但默认不进入发布仓库。

现有结果显示四种模型的中位 R² 均为负。该结果必须与六个时期均值、样本未配对及非嵌套 DEG 筛选等限制一起解释。
