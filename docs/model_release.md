# Generated Model and Artifact Policy

## English

This source repository is complete for standalone use. It contains source code, requirements, tests, packaging scripts, and submission templates needed to install, train, evaluate, and package GRAY-LBPH, RGB-LBPH, and CA-ME-LBPH locally.

Generated artifacts are excluded from source control by design:

- private face images and identity folders;
- real identity mappings and `label_mapping.json` files created from private labels;
- trained `.xml` LBPH models;
- RGB or multi-evidence `.npz` indexes;
- reports, experiment folders, and final `Algorithm.tar.gz` archives.

GRAY-LBPH produces a single OpenCV LBPH model plus label and preprocessing metadata. RGB-LBPH adds a color index and runtime rerank configuration. CA-ME-LBPH adds an auxiliary gray model, a multi-evidence index, preprocessing config, rerank config, and a training report. These are runtime artifacts, not public source files.

Any separate generated model or archive delivery must document:

- the authorized dataset used for training;
- the exact training command and runtime parameters;
- whether face detection was enabled or disabled;
- the train/test split or benchmark harness;
- the observed accuracy and error cases;
- the known generalization boundary;
- whether rerank logic helped or harmed any samples.

Private images and real identity mappings are restricted data and are not part of this source distribution. A generated model or archive is not required for the repository to be a complete GitHub source project; users generate those runtime files locally from their own authorized data.

## 中文

本源码仓库可独立使用，包含在本地安装、训练、评估和打包 GRAY-LBPH、RGB-LBPH、CA-ME-LBPH 所需的源码、依赖、测试、打包脚本和提交模板。

以下生成产物按设计不纳入版本控制：

- 私有人脸图片和身份目录；
- 由真实身份生成的 `label_mapping.json` 等映射文件；
- 训练后的 `.xml` LBPH 模型；
- RGB 或多证据 `.npz` 索引；
- 报告、实验目录和最终 `Algorithm.tar.gz` 压缩包。

GRAY-LBPH 生成单个 OpenCV LBPH 模型及标签、预处理元数据。RGB-LBPH 额外生成颜色索引和 runtime 重排配置。CA-ME-LBPH 额外生成辅助灰度模型、多证据索引、预处理配置、重排配置和训练报告。这些都属于运行产物，不属于公开源码文件。

任何单独交付的模型或压缩包都必须说明：

- 训练所用授权数据集；
- 精确训练命令和运行参数；
- 是否启用人脸检测；
- 训练/测试划分或 benchmark harness；
- 准确率和错误样本；
- 已知泛化边界；
- 重排逻辑是否产生帮助或伤害。

私有人脸图片和真实身份映射属于受限数据，不属于本源码分发内容。生成模型或提交压缩包不是本仓库作为完整 GitHub 源码项目的前置条件；使用者应基于自己具备授权的数据在本地生成运行文件。
