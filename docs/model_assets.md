# Model and data assets


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
模型权重可以通过：https://huggingface.co/wangkun0601/deepppi-d进行下载

## ProMoAI

BPE模型：https://huggingface.co/wangkun0601/promoai-dss3-bpe
6-mer模型：https://huggingface.co/wangkun0601/promoai-dss3-6mer

## Metabolome Service

独立 worker 默认需要：

```text
metabolome_service/data/pred_gene/
├── count_updated.txt
├── result_fpkm_updated.txt
└── pathway_metabolites_intensity.csv
```

也可以通过环境变量使用仓库外数据：

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
```


## SalviaChat

核心问答依赖 MySQL 数据库和 Ollama 模型。数据库凭据通过 `.env` 配置
