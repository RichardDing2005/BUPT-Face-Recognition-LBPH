# Benchmark Summary

## English

This repository records aggregate benchmark information only. It does not include raw images, private identity folders, trained models, or real identity mappings.

The current public result summary is based on a classroom face dataset collected for the BUPT course project:

- Data source: face images collected in class.
- Dataset size: approximately 90 identities and approximately 7,500 images.
- Evaluation setting: held-out test split from the same classroom collection process.
- Reported result: more than 98.5% test-set accuracy.
- Runtime profile: fixed preprocessing with face detection disabled.

GRAY-LBPH is the stable baseline. RGB-LBPH keeps GRAY-LBPH as the primary recognizer and adds an RGB color-rerank criterion for especially similar images. In the classroom benchmark, RGB-LBPH improves over plain GRAY-LBPH and is treated as the course project's SOTA model.

These numbers are not open-world generalization claims. Training and testing images share similar clothing, ambient lighting, camera framing, and background conditions. Any citation of the reported result must include this generalization risk.

## 中文

本仓库只记录聚合 benchmark 信息，不包含原始图片、私有身份目录、训练模型或真实身份映射。

当前公开结果摘要基于 BUPT 课程项目中课堂实际采集的人脸数据：

- 数据来源：课堂采集人脸图片。
- 数据规模：约 90 人、约 7500 张图片。
- 评估设置：来自同一课堂采集流程的测试集划分。
- 结果口径：测试集准确率 98.5% 以上。
- 运行配置：固定预处理，face detection disabled。

GRAY-LBPH 是稳定基线。RGB-LBPH 保留 GRAY-LBPH 作为主识别器，并为特别相似的图片加入 RGB 颜色重排序判据。在课堂 benchmark 中，RGB-LBPH 相比单纯 GRAY-LBPH 有提升，因此作为本次课程项目的 SOTA 模型。

这些数字不是开放场景泛化结论。训练和测试图片具有相近服饰、环境光、取景方式和背景条件。引用结果时必须同时说明这一 generalization risk。
