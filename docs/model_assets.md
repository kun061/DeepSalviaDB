# Model and data assets

大型资源不进入 Git。建议将可公开资源上传到 Hugging Face 或 Zenodo，并在此文档中补充下载链接、版本、许可证与 SHA256。


## SHA256
80cfe1364ca1bd0ab0821055ed91cfe4ee2fe26bdf21888a2ca7ea8e381a9044  ProMoAI/finetune/dss3_bpe_1000bp_head_only_v4_gc_seed42/best_model/model.safetensors
1bb3bfe696235085ffdfc7aee6b0c33c8ab945517dde668e8966e32e3e5e405d  ProMoAI/finetune/dss3_6mer_1000bp_head_only_v4_gc_seed42/best_model/model.safetensors

4da43123f027afb6f8a3927c0b5b08a7ed42110ed7de1aacd3f73d2f2f8a715c  DeepPPI/weights/D/seed_42/best.weights.h5
ed4e6a18e38547dd107afe5bb8223d688f0df03462ea3185844fdcf3d18ffab3  DeepPPI/weights/D/seed_43/best.weights.h5
e310729f4f5d0c94679abccc26fc76b6184a966d2202286fa966430c0e872c3f  DeepPPI/weights/D/seed_44/best.weights.h5
## DeepPPI

API 默认读取：

```text
deepppi/weights/D/
├── seed_42/best.weights.h5
├── seed_43/best.weights.h5
└── seed_44/best.weights.h5
```

训练/复现还需要 `deepppi/data/D/`。如果只提供在线推理，可以只发布三个权重并说明训练数据申请或下载方式。

## ProMoAI

设置 `PROMOAI_MODEL_PATH` 指向 Hugging Face 格式的本地分类模型目录，至少应包含模型配置、分词器文件和 `model.safetensors`。不要把当前约 367 MB 的权重直接提交到普通 Git 历史。

## Metabolome Service

独立 worker 默认需要：

```text
metabolome_service/data/pred_gene/
├── count_updated.txt
├── result_fpkm_updated.txt
└── pathway_metabolites_intensity.csv
metabolome_service/data/dss_final_annotation.csv
```

也可以通过环境变量使用仓库外数据：

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
METABOLOME_ANNOTATION_PATH=/absolute/path/to/dss_final_annotation.csv
```

该工具按请求运行 DESeq2 并拟合传统机器学习模型，当前没有固定预训练权重。可公开资源更适合发布为数据集或可复现分析资产，而不是伪装成神经网络检查点。

如果数据不能再分发，应提供字段说明、合成小样例、SHA256 和数据申请流程。服务启动时缺少这些文件会在 `/health` 中报告 `data_ready=false`。

## SalviaChat

核心问答依赖 MySQL 数据库和 Ollama 模型。数据库凭据通过 `.env` 配置，数据库本身不进入 Git。

可选工具还需要：

- `Homology.csv`
- BLAST nucleotide/protein database prefixes
- `Pfam-A.hmm`
- Chroma 向量数据库
- `BAAI/bge-large-zh-v1.5` embedding model

相应路径通过 `.env.example` 中的环境变量设置。

## Released models

| Model | Version | Download | License | SHA256 |
|---|---|---|---|---|
| DeepPPI-D | v1.0.0 | https://huggingface.co/wangkun0601/deepppi-d | |4da43123f027afb6f8a3927c0b5b08a7ed42110ed7de1aacd3f73d2f2f8a715c  seed_42 |
| ProMoAI BPE | v1.0.0 | https://huggingface.co/wangkun0601/promoai-dss3-bpe | |80cfe1364ca1bd0ab0821055ed91cfe4ee2fe26bdf21888a2ca7ea8e381a9044 |
| ProMoAI 6-mer | v1.0.0 | https://huggingface.co/wangkun0601/promoai-dss3-6mer | |1bb3bfe696235085ffdfc7aee6b0c33c8ab945517dde668e8966e32e3e5e405d |
