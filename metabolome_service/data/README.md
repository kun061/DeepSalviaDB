# Metabolome Service data

运行服务需要：

| 文件 | 必需字段/作用 |
|---|---|
| `pred_gene/count_updated.txt` | 行索引 `Geneid`，包含各时期的原始转录本计数，用于 DESeq2。 |
| `pred_gene/result_fpkm_updated.txt` | `Geneid` 及 DAG30、DAG60、DAG90、DAG240、DAG270、DAG390 的表达重复。 |
| `pred_gene/pathway_metabolites_intensity.csv` | `Name`、`Pathway` 及六个时期的代谢物强度重复。 |

推荐通过环境变量使用仓库外数据：

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
```
