# TMC-Miner data

These three files are bundled with the repository and are read by default:

| File | Required fields and purpose |
|---|---|
| `pred_gene/count_updated.txt` | `Geneid` row index and raw transcript counts by stage; used by DESeq2. |
| `pred_gene/result_fpkm_updated.txt` | `Geneid` and expression replicates for DAG30, DAG60, DAG90, DAG240, DAG270, and DAG390. |
| `pred_gene/pathway_metabolites_intensity.csv` | `Name`, `Pathway`, and metabolite-intensity replicates for six stages. |

Set an external data directory only if you want to use different tables:

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
```

The SHA256 checksums are recorded in `../../docs/model_assets.md`. Follow the original providers' terms when using or redistributing the data.
