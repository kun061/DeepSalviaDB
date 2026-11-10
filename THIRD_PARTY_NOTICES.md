# Third-party notices

## Plant DNA LLMs / PDLLMs

ProMoAI uses the Plant DNAGPT-BPE and Plant DNAGPT-6mer pretrained models developed by Zhang Tao Lab.

- Source code: https://github.com/zhangtaolab/Plant_DNA_LLMs
- BPE model: https://huggingface.co/zhangtaolab/plant-dnagpt-BPE
- 6-mer model: https://huggingface.co/zhangtaolab/plant-dnagpt-6mer
- License text: https://github.com/zhangtaolab/Plant_DNA_LLMs/blob/main/LICENSE
- Copyright remains with the upstream authors.

Citation:

> Liu GQ, Chen L, Wu YC, Han YS, Bao Y, Zhang T. PDLLMs: A group of tailored DNA large language models for analyzing plant genomes. Molecular Plant. 2025;18(2):175-178. https://doi.org/10.1016/j.molp.2024.12.006

The upstream repository is not vendored into this project. The files under `promoai/scripts/` implement this project's DSS3 dataset construction, head-only fine-tuning, evaluation, and threshold analysis. Users must obtain the original pretrained checkpoints from the upstream Hugging Face repositories.

