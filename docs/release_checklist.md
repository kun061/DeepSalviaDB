# GitHub 发布清单

## 当前结论

这个目录是结构和命名已经整理好的 **release candidate**，不是可以不经确认直接公开的最终版本。代码、文档和忽略规则可以进入 GitHub；下列“发布前必须完成”项目需要仓库所有者确认后，才能标记正式版本。

## 已完成

- [x] 四项服务源码和统一 FastAPI 网关已整理。
- [x] 产品目录使用小写名称：`deepppi/`、`promoai/`、`salviachat/`。
- [x] Python 文件使用 `snake_case`，FastAPI 入口统一为 `app.py`。
- [x] SalviaChat 接口按功能放入 `salviachat/routers/`。
- [x] DeepPPI 训练代码集中在 `deepppi/training/`。
- [x] 权重、原始数据、数据库、缓存、密钥和大体积运行结果已排除。
- [x] 提供环境变量模板、依赖文件、第三方来源说明和模型资源说明。
- [x] 保留 ProMoAI 的小型配置与汇总指标，便于审稿人核对实验；不包含模型权重和逐条预测。
- [x] 提供发布前自动检查脚本 `scripts/check_release.py`。

## 发布前必须完成

- [ ] 选择并添加仓库根目录 `LICENSE`；不能在未确认上游和数据许可时默认套用 MIT。
- [ ] 在 `docs/model_assets.md` 填写模型/数据的正式下载地址、版本、许可证和 SHA256。
- [ ] 在根目录 `README.md` 补充本项目论文、DeepPPI 数据集以及 Plant DNA LLMs 的完整引用。
- [ ] 确认所有拟公开的小型实验结果和示例数据均允许再分发。
- [ ] 用干净环境按 README 启动两个服务，并实际请求 `/health` 和核心预测接口。
- [ ] 搜索并确认仓库中不存在真实密码、Token、服务器地址、个人路径或受限样本信息。

## 建议完成

- [ ] 将权重上传至 Zenodo 或 Hugging Face，使用固定版本链接，并为正式发布创建 DOI。
- [ ] 在 GitHub Actions 中运行语法检查和 `scripts/check_release.py`。
- [ ] 使用版本标签（例如 `v1.0.0`）并在 GitHub Release 中记录模型资源版本。
- [ ] 若论文需要完整复现，另行发布冻结的数据划分清单、随机种子和汇总结果；大文件不要直接放普通 Git。

## 最终发布判定

“发布前必须完成”全部勾选、自动检查通过且服务完成一次干净环境冒烟测试后，才建议把本 release candidate 标为最终版。
