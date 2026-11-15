# TMC-Miner data

The worker requires these three data files:

| File | Required fields and purpose |
|---|---|
| `pred_gene/count_updated.txt` | `Geneid` row index and raw transcript counts by stage; used by DESeq2. |
| `pred_gene/result_fpkm_updated.txt` | `Geneid` and expression replicates for DAG30, DAG60, DAG90, DAG240, DAG270, and DAG390. |
| `pred_gene/pathway_metabolites_intensity.csv` | `Name`, `Pathway`, and metabolite-intensity replicates for six stages. |

A separate annotation CSV is also needed. Prefer external, access-controlled paths:

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
METABOLOME_ANNOTATION_PATH=/absolute/path/to/dss_final_annotation.csv
```

Check the original provider's terms before redistributing these files.
