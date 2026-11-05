> GitHub 发布说明：本目录不随仓库分发模型权重、训练数据和历史结果；请先阅读根目录 docs/model_assets.md。
# DeepPPI 一串红 D方案复现包

本目录提供 DeepPPI-D 的训练、评估和 REST 推理源码。冻结数据、三个种子的模型权重及历史结果属于外部资产，不进入普通 Git；下载和放置方式见根目录 `docs/model_assets.md`。

## 目录

| 目录 | 内容 |
|---|---|
| `data/D/` | 外部资产：蛋白序列、训练/验证/测试样本表、蛋白分区和数据 manifest |
| `training/` | 实际Siamese CNN、编码、训练、指标和bootstrap代码；入口仅支持protein-disjoint |
| `scripts/` | 文件与数据检查、GPU启动、D模型复测脚本 |
| `weights/D/seed_42,43,44/` | 外部资产：三个种子的最佳权重和训练元数据 |
| `results/D_test/` | 可选外部资产：历史测试预测、指标和曲线 |
| `results/training_D/` | 可选外部资产：训练历史和选模曲线 |
| `environment/` | Conda配置、60项pip运行依赖锁定及环境记录 |
| `verification/` | 运行验证后生成，不提交 |
| `MANIFEST.json` | 完整资产包提供的校验清单；源码仓库不含该文件 |

## 环境与运行

在Linux x86_64 NVIDIA GPU服务器上解压并进入本包根目录，在终端运行：

```bash
conda env create -f environment/environment.yml
conda activate salvia-deepppi-d
# 下载 data、weights 和 MANIFEST.json 后执行完整校验
python scripts/verify_package.py
bash scripts/run.sh smoke
bash scripts/run.sh evaluate
```

环境固定Python 3.10.8、pip 26.2.1、TensorFlow 2.15.1、Keras 2.15.0、NumPy 1.26.4、h5py 3.10.0及matplotlib 3.8.4。CUDA运行库和cuDNN来自已锁定的NVIDIA pip包，由启动脚本自动配置当前环境的库路径。宿主机需有可用NVIDIA驱动；参考服务器为RTX 3090 24GB、驱动560.35.03。环境安装需要网络。

历史训练来自独立venv，这里的Conda配置按其实际运行依赖重建。60项运行依赖均精确锁定；setuptools和wheel由Conda提供，原构建工具版本仅作溯源记录，不要求安装镜像中不可用的旧版本。

`verify_package.py`仅用Python标准库，可在安装依赖前运行。`smoke`执行GPU合成输入训练检查；`evaluate`用随包三个权重计算测试预测并与各自历史参考逐对比较，概率及指标误差容许值为1e-6，输出到新目录`runs/evaluation_D`。该结果反映固定权重复测，不等同重新训练。

## 重新训练

```bash
bash scripts/run.sh train
```

使用随包D数据，从头训练种子42、43、44；Adam学习率0.001、batch 128、最多30轮、早停patience 5、按验证AUC选择最佳检查点，分类阈值固定0.5。结果保存到`runs/retrain_D`。已存在的输出目录不会被覆盖。

重新训练后的模型可这样评估：

```bash
bash scripts/run.sh evaluate \
  --weights-dir runs/retrain_D/protein_disjoint \
  --output-dir runs/evaluation_retrained_D
```

新权重评估不强制等于原参考结果。固定随机种子也不保证跨GPU、驱动或软件构建的完整训练结果逐位相同。

## 数据与方法

| 集合 | 阳性对 | 负例对 | 合计 |
|---|---:|---:|---:|
| 训练 | 14,326 | 11,884 | 26,210 |
| 验证 | 1,570 | 1,269 | 2,839 |
| 测试 | 1,724 | 1,247 | 2,971 |

研究物种为一串红Salvia splendens，taxid 180675。正例为作者900版CSV候选中经STRING v12.0核查combined_score ≥900并清洗去重的关联；负例是指定关联文件未记录的抽样对，不使用已有低分关联作负例。

完整序列在`data/D/proteins.csv.gz`中，通过`protein_id`与各样本表的`protein_a`和`protein_b`连接。一行是无方向蛋白质对。氨基酸`ACDEFGHIKLMNPQRSTVWY`编码为1至20，其他字符为0，取N端前512位，不足末端补0。`pair_id`为排序后两端ID用竖线连接的SHA256，`model_pair_id`用同样方法连接两端编码输入哈希。

完全相同的512位输入绑定成组，以固定种子42按约60%/20%/20%的输入组数分配训练/验证/测试，跨分区蛋白质对舍弃。上述比例不是样本对比例。三个集合的蛋白ID及完全相同输入均无交集；未按一般同源相似度聚类。

D根据每集合阳性端点出现频次与负/正样本数之比设置整数配额，再用合法边替换修复。候选负例仅在对应蛋白分区内构建，输入组成员之间任一ID对在指定STRING基础文件有记录就排除。训练负例的归一化端点频次TVD为0.0182583；训练整数配额L1残差为0。验证/测试TVD分别为0.0432443/0.0476447。

本包直接复现冻结D数据上的训练与测试，不执行从原始STRING重新构造负例。`proteins.csv.gz`保留候选蛋白范围；实际训练/评估仅使用各集合样本表引用的蛋白。数据表与序列表字节保持原样；manifest仅保留当前协议的字段，并记录原D manifest的SHA256。最佳权重元数据中的历史manifest哈希是来源标记，不是运行时外部路径依赖。

## D方案结果

| 指标 | 三个种子均值±样本标准差 |
|---|---:|
| ROC-AUC | 0.7817 ± 0.0170 |
| PR-AUC 梯形积分 | 0.8232 ± 0.0080 |
| AP | 0.8233 ± 0.0080 |
| Precision | 0.8003 ± 0.0455 |
| Recall | 0.5787 ± 0.1394 |
| F1 | 0.6617 ± 0.0869 |

指标先按种子独立计算再求均值和样本标准差。PR-AUC与AP定义不同；标准差不代表95%置信区间。STRING高置信功能关联不等同实验验证的直接物理PPI；未记录对仍可能包含未知关联。当前未提供外部实验金标准或跨物种泛化验证。
