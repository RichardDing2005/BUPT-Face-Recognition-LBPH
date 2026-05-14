# Generated Model and Artifact Policy

## English

This source repository is complete for standalone use. It contains the source code, requirements, configuration examples, tests, packaging scripts, and submission templates needed to install, train, evaluate, and package GRAY-LBPH and RGB-LBPH locally.

Trained models, private face images, real identity mappings, and final `Algorithm.tar.gz` archives are generated artifacts. They are excluded from source control by design and are produced locally through the documented training and packaging commands. GRAY-LBPH produces `face_recognizer_model.xml`; RGB-LBPH produces a GRAY-LBPH model plus `color_index.npz` and runtime rerank configuration files.

Any separate generated model or archive delivery must document:

- The model was trained on classroom-collected face images.
- The dataset contains approximately 90 identities and approximately 7,500 images.
- The reported test-set accuracy is more than 98.5% under the same collection setting.
- Face detection is disabled in the reported runtime profile.
- Training and testing images share similar clothing, ambient lighting, camera framing, and background conditions.
- The result is not an open-world generalization benchmark.
- RGB-LBPH is the course project's SOTA model, but it is more sensitive to light intensity and light color than GRAY-LBPH.
- The exact training command, runtime parameters, and benchmark boundary used to produce the artifact.

Private images and real identity mappings are restricted data and are not part of this source distribution. A generated model or archive is not required for the repository to be a complete GitHub source project; users generate those runtime files locally from their own authorized data.

## 中文

本源码仓库作为 GitHub 项目可独立使用，包含在本地安装、训练、评估和打包 GRAY-LBPH 与 RGB-LBPH 所需的源码、依赖、配置示例、测试、打包脚本和提交模板。

训练模型、私有人脸图片、真实身份映射和最终 `Algorithm.tar.gz` 属于生成产物。它们按设计不纳入源码版本控制，而是通过文档中的训练和打包命令在本地生成。GRAY-LBPH 会生成 `face_recognizer_model.xml`；RGB-LBPH 会生成 GRAY-LBPH 模型、`color_index.npz` 和运行时重排序配置文件。

任何单独交付的生成模型或压缩包都必须说明：

- 模型基于课堂实际采集的人脸图片训练。
- 数据集约 90 人、约 7500 张图片。
- 在同一采集条件下的测试集上，准确率达到 98.5% 以上。
- 报告运行配置中 face detection disabled。
- 训练和测试图片具有相似服饰、环境光、取景方式和背景条件。
- 该结果不是开放场景泛化 benchmark。
- RGB-LBPH 是本次课程项目的 SOTA 模型，但比 GRAY-LBPH 更依赖光强和光颜色。
- 产物对应的准确训练命令、运行参数和 benchmark 边界。

私有人脸图片和真实身份映射属于受限数据，不属于本源码分发内容。已生成模型或压缩包不是本仓库作为完整 GitHub 源码项目的前置条件；使用者应基于自己具备使用授权的数据在本地生成这些运行文件。
