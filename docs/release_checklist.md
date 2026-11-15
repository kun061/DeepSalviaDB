# GitHub release checklist

## Current status

This repository is a release candidate with an organized structure and consistent naming. The code and documentation can be published, but the remaining checks below should be completed before labeling a release as final.

## Completed

- [x] Organized the source code for four AI services and the FastAPI gateway.
- [x] Used lowercase product directories: `deepppi/`, `promoai/`, `tmc_miner/`, and `salviachat/`.
- [x] Used `snake_case` Python filenames and `app.py` FastAPI entry points.
- [x] Grouped SalviaChat endpoints under `salviachat/routers/`.
- [x] Grouped DeepPPI training code under `deepppi/training/`.
- [x] Excluded model weights, raw data, databases, caches, credentials, and large runtime results.
- [x] Included an environment-variable template, dependency files, provenance notice, and model asset instructions.
- [x] Kept small ProMoAI configurations and summary metrics for reviewers, without weights or per-sample predictions.
- [x] Verified that the three model repository pages are publicly accessible and recorded their revisions.
- [x] Confirmed with the repository owner that the frozen DeepPPI D dataset may be redistributed.

## Before labeling a final release

- [ ] Decide whether and how to license the project's original code. The current `LICENSE` is a third-party resource notice, not a grant of rights for original code.
- [x] Verify published weight SHA256 values against Hugging Face LFS metadata.
- [ ] Add model cards and explicit license metadata to all published Hugging Face model repositories.
- [ ] Add complete citations for the DeepSalviaDB paper, DeepPPI data sources, and Plant DNA LLMs.
- [ ] Confirm that every published small result and sample dataset may be redistributed.
- [ ] Start DeepPPI, TMC-Miner, ProMoAI, and the gateway in a clean environment; request `/health` and the main prediction endpoints.
- [ ] Review tracked files for real credentials, tokens, private server addresses, personal paths, and restricted sample information.

## Recommended

- [ ] Use fixed model revisions in release instructions and create a DOI if the publication requires one.
- [ ] Tag a release (for example, `v1.0.0`) and record the model versions in the GitHub Release notes.
- [ ] If complete experimental reproduction is required, publish a frozen split manifest, seeds, and summary results separately. Keep large files out of regular Git history.

## Release decision

Label this candidate as final only after the required items are complete, automated checks pass, and a clean-environment service smoke test succeeds.
