# Extensibility

## English

The current project is optimized for the classroom benchmark: approximately 90 identities, approximately 7,500 classroom-collected images, fixed preprocessing, and face detection disabled during the reported test run. Deployment-level generalization requires additional validation beyond this course-project setup.

Recommended extension directions:

- Re-enable and evaluate face detection when backgrounds, framing, or head position vary.
- Collect additional images under different clothing, lighting intensity, light color, camera positions, and backgrounds.
- Add an independent external test set that is not collected in the same classroom session.
- Evaluate color normalization or illumination correction before relying on RGB-LBPH in mixed lighting.
- Report GRAY-LBPH and RGB-LBPH separately, because RGB-LBPH has stronger sensitivity to light intensity and light color.

The CLI already exposes the main preprocessing and LBPH parameters:

- `--resize WIDTHxHEIGHT`
- `--no-detect-face`
- `--equalization`
- `--radius`
- `--neighbors`
- `--grid-x`
- `--grid-y`
- `--threshold`

For RGB-LBPH, also evaluate:

- `--color-bins`
- `--top-k`
- `--confidence-gate`
- `--margin-ratio`

## 中文

当前项目针对课堂 benchmark 优化：约 90 人、约 7500 张课堂采集图片、固定预处理，并且报告测试时 face detection disabled。这是一个较强的课程项目设置，但如果要声称部署级泛化能力，还需要扩展验证。

推荐扩展方向：

- 当背景、取景或头部位置变化更明显时，重新启用并评估人脸检测。
- 增加不同服饰、光强、光颜色、摄像头位置和背景下采集的图片。
- 引入独立外部测试集，避免只在同一课堂采集批次内评估。
- 在混合光照下使用 RGB-LBPH 前，评估颜色归一化或光照校正。
- 分别报告 GRAY-LBPH 和 RGB-LBPH，因为 RGB-LBPH 对光强和光颜色更敏感。

当前 CLI 已暴露主要预处理和 LBPH 参数：

- `--resize WIDTHxHEIGHT`
- `--no-detect-face`
- `--equalization`
- `--radius`
- `--neighbors`
- `--grid-x`
- `--grid-y`
- `--threshold`

对 RGB-LBPH 还应评估：

- `--color-bins`
- `--top-k`
- `--confidence-gate`
- `--margin-ratio`
