# Benchmark Summary

## English

This repository records aggregate benchmark information only. It does not include raw images, private identity folders, trained models, real identity mappings, `.npz` evidence indexes, or final submission archives.

The public result summary is based on classroom-collected face images for the BUPT course project:

- Data source: face images collected in class.
- Evaluation setting: held-out test split from the same classroom collection process or a score2026-style local benchmark.
- Runtime profile: fixed preprocessing with face detection disabled.
- Published boundary: controlled classroom benchmark, not open-world deployment.

GRAY-LBPH is the stable baseline. RGB-LBPH keeps GRAY-LBPH as the primary recognizer and adds RGB color-rerank evidence for especially similar images. CA-ME-LBPH keeps the same top-k safety boundary but uses multiple evidence sources: auxiliary gray LBPH, Lab/HSV/chromaticity color histograms, local texture statistics, and quality factors.

In local score2026-style validation, CA-ME-LBPH reached full local correctness on the available benchmark data after excluding known dataset-label issues from the cloud benchmark discussion. These numbers should be cited with the acquisition boundary: training and testing images share similar clothing, ambient lighting, camera framing, and background conditions.

## 中文

本仓库只记录聚合 benchmark 信息，不包含原始图片、私有身份目录、训练模型、真实身份映射、`.npz` evidence index 或最终提交压缩包。

公开结果摘要基于 BUPT 课程项目中的课堂采集人脸图像：

- 数据来源：课堂采集人脸图片。
- 评估设置：来自同一课堂采集流程的 holdout 测试集，或 score2026 风格本地 benchmark。
- 运行配置：固定预处理，关闭人脸检测。
- 结果边界：受控课堂 benchmark，不是开放场景部署结论。

GRAY-LBPH 是稳定基线。RGB-LBPH 保留 GRAY-LBPH 作为主识别器，并为相近图像加入 RGB 颜色重排证据。CA-ME-LBPH 延续 top-k 安全边界，但使用更多证据：辅助灰度 LBPH、Lab/HSV/chromaticity 颜色直方图、局部纹理统计和质量因子。

在本地 score2026 风格验证中，CA-ME-LBPH 在可用 benchmark 数据上达到本地全正确；云端 benchmark 中已知标签问题需要单独说明。引用这些数字时必须同时说明采集边界：训练和测试图片具有相近服装、环境光、取景方式和背景条件。
