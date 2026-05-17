# Extensibility

## English

The current project is optimized for classroom benchmark data: fixed preprocessing, identity-organized image folders, and face detection disabled during the reported runtime. Deployment-level generalization requires additional validation beyond this course-project setup.

Recommended extension directions:

- Re-enable and evaluate face detection when backgrounds, framing, or head position vary.
- Collect additional images under different clothing, lighting intensity, light color, camera positions, and backgrounds.
- Add an independent external test set that is not collected in the same classroom session.
- Report GRAY-LBPH, RGB-LBPH, and CA-ME-LBPH separately because their error modes differ.
- For RGB-LBPH, evaluate color normalization before relying on RGB evidence in mixed lighting.
- For CA-ME-LBPH, tune top-k size, evidence weights, confidence gates, and switch margin with rerank help/harm reporting.
- Use augmentation helpers only as controlled experiments; never mix augmented copies into a test split.
- Use confusion-cluster diagnostics when a small label group repeatedly appears in top-k results.

The CLI exposes the main preprocessing and LBPH parameters:

- `--resize WIDTHxHEIGHT`
- `--no-detect-face`
- `--equalization`
- `--radius`
- `--neighbors`
- `--grid-x`
- `--grid-y`

For rerank models, also evaluate:

- `--candidate-top-k` or `--top-k`
- `--confidence-gate`
- `--gray-margin-gate`
- `--switch-margin`
- evidence-specific bins and grid sizes

## 中文

当前项目针对课堂 benchmark 优化：固定预处理、按身份目录组织图片，并在报告运行时关闭人脸检测。如果要声称部署级泛化能力，需要在课程项目设置之外继续验证。

推荐扩展方向：

- 当背景、取景或头部位置变化明显时，重新启用并评估人脸检测。
- 增加不同服装、光强、光色、摄像头位置和背景下采集的图片。
- 引入独立外部测试集，避免只在同一课堂采集批次内评估。
- 分别报告 GRAY-LBPH、RGB-LBPH 和 CA-ME-LBPH，因为三者错误模式不同。
- 对 RGB-LBPH，在混合光照下使用前应评估颜色归一化或光照校正。
- 对 CA-ME-LBPH，应结合 `rerank_help / rerank_harm / net_gain` 调整 top-k、证据权重、置信门限和切换 margin。
- 轻度增强工具只作为受控实验使用，不应把增强副本混入测试集。
- 当少数标签持续出现在 top-k 混淆中时，优先做混淆簇诊断。

当前 CLI 暴露了主要预处理和 LBPH 参数：

- `--resize WIDTHxHEIGHT`
- `--no-detect-face`
- `--equalization`
- `--radius`
- `--neighbors`
- `--grid-x`
- `--grid-y`

对重排模型，还应评估：

- `--candidate-top-k` 或 `--top-k`
- `--confidence-gate`
- `--gray-margin-gate`
- `--switch-margin`
- 各类证据的 bins 和 grid 设置
