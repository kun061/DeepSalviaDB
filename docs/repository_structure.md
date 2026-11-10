# 仓库目录与文件说明

## 根目录

| 路径 | 作用 |
|---|---|
| `README.md` | 项目总览、服务入口、启动方法和发布注意事项。 |
| `.env.example` | 环境变量模板，只保留占位值，不包含真实密钥。 |
| `.gitignore` | 排除权重、数据、数据库、日志、缓存和本地配置。 |
| `requirements.txt` | 统一 API 网关和常用功能的 Python 依赖。 |
| `THIRD_PARTY_NOTICES.md` | Plant DNA LLMs 等第三方代码、模型和许可证说明。 |
| `scripts/check_release.py` | 检查敏感文件、超大文件及不应发布的模型/数据。 |

## `docs/`

| 文件 | 作用 |
|---|---|
| `api.md` | REST API 路由、请求参数和响应形式。 |
| `model_assets.md` | 仓库外模型、数据和数据库的放置位置及发布要求。 |
| `release_checklist.md` | 区分已完成项目和正式公开前仍需人工确认的项目。 |
| `repository_structure.md` | 本清单及统一命名规则。 |

## `deepppi/`

| 路径 | 作用 |
|---|---|
| `README.md` | DeepPPI-D 的模型方法、训练、评估和推理说明。 |
| `app.py` | DeepPPI 独立 FastAPI worker，加载三个种子模型并提供 PPI 推理。 |
| `data_scope.json` | 训练/验证/测试数据及外部资源范围清单。 |
| `environment/environment.yml` | 原始 Conda 环境定义，供完整复现参考。 |
| `environment/requirements.lock.txt` | 已冻结的 pip 包版本。 |
| `scripts/run.sh` | DeepPPI 训练、评估等常用命令入口。 |
| `scripts/reproduce.py` | 按配置运行复现实验。 |
| `scripts/verify_package.py` | 外部资产就位后核对数据、权重和清单。 |
| `training/train.py` | DeepPPI-D 训练和模型保存逻辑。 |
| `training/metrics.py` | 汇总并计算复现实验评价指标。 |
| `weights/README.md` | 三个种子权重的预期结构；实际权重不进入 Git。 |

## `promoai/`

| 路径 | 作用 |
|---|---|
| `README.md` | ProMoAI 数据构建、模型训练、评估与上游关系说明。 |
| `upstream.md` | Plant DNA LLMs 的来源、使用边界和引用要求。 |
| `scripts/README.md` | 实验脚本的推荐执行顺序与输入输出。 |
| `scripts/build_dss3_chromosome_holdout_v4_gc_matched.py` | 构建按染色体留出的 GC 匹配 DSS3 数据集。 |
| `scripts/train_dss3_bpe_1000bp_head_only.py` | 训练 BPE 版本的 1 kb 启动子分类头。 |
| `scripts/evaluate_dss3_bpe_1000bp.py` | 评估 BPE 模型。 |
| `scripts/train_dss3_6mer_1000bp_head_only_v4_gc.py` | 训练 6-mer 版本的 1 kb 启动子分类头。 |
| `scripts/evaluate_dss3_6mer_1000bp_v4_gc.py` | 评估 6-mer 模型。 |
| `scripts/analyze_dss3_v4_threshold_sensitivity.py` | 分析分类阈值对开发集和测试集指标的影响。 |
| `results/bpe_v4_seed42/` | BPE 配置、加载审计、验证曲线和测试汇总；不含权重与逐条预测。 |
| `results/6mer_v4_seed42/` | 6-mer 配置与汇总指标；不含权重与逐条预测。 |
| `results/threshold_sensitivity/` | 阈值敏感性汇总和报告。 |
| `results/baselines/` | RF、SVM、CNN 基线配置及开发集/测试集汇总。 |
| `results/comparison.csv` | 各模型最终指标对比表。 |

`promoai/` 不复制 Plant DNA LLMs 的完整上游仓库。用户应依据 `upstream.md` 获取有权使用的预训练模型。

### ProMoAI 结果文件

| 文件名 | 作用 |
|---|---|
| `best_dev.json` | 最佳开发集检查点的汇总指标。 |
| `training_config.json`、`model_config.json`、`tokenizer_config.json` | 训练、模型和分词器配置快照。 |
| `model_loading_report.json`、`input_length_audit.json`、`data_manifest.json` | 模型加载、输入长度和数据范围审计。 |
| `validation_metrics.csv`、`training_history.csv` | 逐轮验证指标和训练历史。 |
| `test_metrics.json`、`confusion_matrix.csv` | 固定测试集的汇总指标和混淆矩阵。 |
| `baselines/*/config.json` | RF、SVM、CNN 各基线配置。 |
| `baselines/*/dev_metrics.json`、`test_metrics.json` | 各基线开发集和测试集汇总。 |
| `threshold_metrics_dev.csv`、`threshold_metrics_test.csv` | 不同分类阈值下的指标。 |
| `threshold_sensitivity_summary.json`、`threshold_sensitivity_report.md`、`audit.json` | 阈值实验汇总、可读报告和输入审计。 |

## `metabolome_service/`

| 路径 | 作用 |
|---|---|
| `README.md` | 模型性质、外部数据、启动、API、验证结果与限制。 |
| `app.py` | 独立 FastAPI worker；按请求执行 DESeq2 和四类机器学习排序。 |
| `requirements.txt` | 独立 worker 的最小依赖。 |
| `data/README.md` | 四个外部组学/注释文件的字段与许可要求。 |
| `analysis/cross_validate.py` | 留一时期交叉验证和候选基因稳定性分析。 |
| `results/experiment_metadata.json` | 验证设计、样本配对限制与 DEG 数量。 |
| `results/summary_by_model.csv` | 四种算法的汇总泛化指标。 |
| `results/README.md` | 结果文件范围和解释限制。 |

该服务没有固定预训练权重；树结构和回归系数在请求期间拟合。真实组学数据及大型派生结果不进入 Git。

## `salviachat/`

| 路径 | 作用 |
|---|---|
| `app.py` | 统一 FastAPI 网关，注册所有功能路由并提供健康检查。 |
| `data/README.md` | 运行时数据目录说明；真实数据不进入 Git。 |
| `routers/chat.py` | SalviaChat 多智能体问答、流式响应和绘图接口。 |
| `routers/promoter.py` | ProMoAI 启动子预测和顺式元件标注接口。 |
| `routers/metabolome.py` | 将请求转发给内部 Metabolome worker。 |
| `routers/deepppi.py` | 将统一网关请求转发给内部 DeepPPI worker。 |
| `routers/structure.py` | 蛋白结构预测接口。 |
| `routers/primer.py` | 引物设计接口。 |
| `src/core/settings.py` | 配置和环境变量读取。 |
| `src/core/state.py` | 多智能体会话状态定义。 |
| `src/core/router.py` | 问题分类和智能体路由。 |
| `src/core/rag.py` | RAG 检索与问答编排。 |
| `src/core/generator.py` | 回答生成与图表相关输出。 |
| `src/core/evaluator.py` | 回答质量评估。 |
| `src/tools/entity_linker.py` | 生物实体识别、规范化与链接。 |
| `src/agents/knowledge_agent/component.py` | 知识检索智能体的节点实现。 |
| `src/agents/knowledge_agent/graph.py` | 知识检索智能体的 LangGraph 流程。 |
| `src/agents/salvia_agent/component.py` | 丹参数据库查询智能体的节点实现。 |
| `src/agents/salvia_agent/graph.py` | 丹参数据库查询智能体的 LangGraph 流程。 |
| `src/agents/tool_agent/component.py` | 生物信息工具智能体的节点实现。 |
| `src/agents/tool_agent/graph.py` | 生物信息工具智能体的 LangGraph 流程。 |
| 各级 `__init__.py` | 将对应目录声明为可导入的 Python 包。 |

## 不进入 GitHub 的内容

- `.env`、API Key、数据库密码和真实服务器地址。
- PyTorch、Safetensors、Joblib 等模型权重与检查点。
- 原始 FASTA、CSV、数据库转储、向量库和 BLAST/Pfam 数据库。
- 用户请求、聊天记录、临时上传、日志、PID、缓存和 `__pycache__`。
- 可重新生成的逐样本预测和大型图表；只保留必要配置与汇总指标。
