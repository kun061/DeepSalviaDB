# Model and data assets

See [LICENSE](../LICENSE) for the provenance and licensing boundaries of models and data.

## Published datasets

The frozen DeepPPI D dataset is included at `deepppi/data/D/`. File counts, label counts, and SHA256 checksums are recorded in `deepppi/data/D/manifest.json`. The dataset includes material from STRING v12.0; credit STRING and follow its CC BY 4.0 terms for that material. Verify the original terms for content from other providers before redistributing it.

ProMoAI's frozen 1 kb train, development, and test splits are included at `promoai/DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/`. They contain 37,516, 5,448, and 3,704 sequences, respectively. The source genome, GFF, and per-sample chromosome metadata are not included.

TMC-Miner includes the three count, FPKM, and metabolite-intensity tables required by its worker under `tmc_miner/data/pred_gene/`. SalviaChat includes `salviachat/data/SalviaBench-Final-150.csv`, a 150-question benchmark export with gold answers.

### Dataset SHA256 checksums

```text
9b9a761f13e17e0bc552b12e2498e32ffa09ec8e78efb08efa302e244d5695a9  promoai/DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/train.csv
46a83e88b3b623e34f83e6b5dc034f0e3528b8233b202304065deba86748e144  promoai/DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/dev.csv
3aeffb6be07061e5ed925915c048dcc6bac77cabd0cf3542785fee7f71408a1a  promoai/DSS3/promoter_1kb/chromosome_holdout_v4_gc_matched/test.csv
02a2118397b2cce22462e04debfd0580285c4139ca06ca0484dc2b7958f4d438  tmc_miner/data/pred_gene/count_updated.txt
990d1dbeb2700f446c36fbc48835a7cf45e89ed660e54b8072a7331c2e9df220  tmc_miner/data/pred_gene/result_fpkm_updated.txt
4b460fb86b40c199bf76563c3fd07884b4b2f409ca104b78fbacdd8ae176659b  tmc_miner/data/pred_gene/pathway_metabolites_intensity.csv
b6992d5aebb0ff315175d5f0e3ff50c8a364ccf5e87c5097a96e2394ac9f1db0  salviachat/data/SalviaBench-Final-150.csv
```

## Published model repositories

| Model | Hugging Face repository | Verified repository revision |
|---|---|---|
| DeepPPI-D | [wangkun0601/deepppi-d](https://huggingface.co/wangkun0601/deepppi-d) | `0216e75f3301cd2df6ec2da1b16f57f6add0fa56` |
| ProMoAI BPE | [wangkun0601/promoai-dss3-bpe](https://huggingface.co/wangkun0601/promoai-dss3-bpe) | `f967ef55e489412a0dbc8b911f9378fe4c5d2c6d` |
| ProMoAI 6-mer | [wangkun0601/promoai-dss3-6mer](https://huggingface.co/wangkun0601/promoai-dss3-6mer) | `9a0ffa16e0326b5bc73f9307c2e921ec6df1322d` |

These repository revisions and the weight SHA256 values below were checked against the public Hugging Face repository metadata on 2026-11-15. The repositories currently have no model-card license metadata. The ProMoAI checkpoints are based on upstream Plant DNAGPT models labeled CC BY-NC-SA 4.0; confirm the applicable terms before reusing the weights. DeepPPI weight licensing should be stated explicitly by the publisher.

## Model weight SHA256 checksums

```text
80cfe1364ca1bd0ab0821055ed91cfe4ee2fe26bdf21888a2ca7ea8e381a9044  promoai-dss3-bpe/model.safetensors
1bb3bfe696235085ffdfc7aee6b0c33c8ab945517dde668e8966e32e3e5e405d  promoai-dss3-6mer/model.safetensors
4da43123f027afb6f8a3927c0b5b08a7ed42110ed7de1aacd3f73d2f2f8a715c  deepppi-d/D/seed_42/best.weights.h5
ed4e6a18e38547dd107afe5bb8223d688f0df03462ea3185844fdcf3d18ffab3  deepppi-d/D/seed_43/best.weights.h5
e310729f4f5d0c94679abccc26fc76b6184a966d2202286fa966430c0e872c3f  deepppi-d/D/seed_44/best.weights.h5
```

The checksums above match the published Hugging Face LFS metadata. After downloading, run `sha256sum` on each local file to verify its contents.

## DeepPPI

The API expects the downloaded repository's `D/` directory beneath `deepppi/weights/`:

```text
deepppi/weights/D/
├── seed_42/best.weights.h5
├── seed_43/best.weights.h5
└── seed_44/best.weights.h5
```

## ProMoAI

The standalone worker loads the BPE model from the absolute directory specified by `PROMOAI_MODEL_PATH`. The 6-mer checkpoint supports the published experiments but is not loaded by that worker. Training and evaluation scripts use the bundled promoter splits by default.

## TMC-Miner

The worker reads these three bundled files under `tmc_miner/data/pred_gene/`:

```text
tmc_miner/data/pred_gene/
├── count_updated.txt
├── result_fpkm_updated.txt
└── pathway_metabolites_intensity.csv
```

To use another dataset, configure an external directory:

```dotenv
METABOLOME_DATA_DIR=/absolute/path/to/pred_gene
```

## SalviaChat

The bundled SalviaBench CSV is evaluation data, not a runtime requirement. Full question answering depends on MySQL, Ollama models, and other scientific data resources. Configure local database credentials and paths in `.env`.
