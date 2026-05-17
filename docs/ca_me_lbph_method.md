# CA-ME-LBPH Method Notes

## English

CA-ME-LBPH means Confusion-Aware Multi-Evidence LBPH. It is built as a conservative rerank layer on top of GRAY-LBPH rather than as an independent classifier.

### Data Flow

1. The training set is organized as one folder per identity.
2. The primary image path follows the score2026-style input adapter and fixed preprocessing profile.
3. The primary GRAY-LBPH recognizer is trained on `400x450` grayscale images.
4. An auxiliary coarse GRAY-LBPH recognizer is trained on `200x200` grayscale images.
5. For each training image, the trainer stores multi-evidence features by identity:
   - Lab a/b, HSV H/S, and chromaticity histograms;
   - local grayscale texture histograms and gradient statistics;
   - brightness, contrast, Laplacian blur, and color reliability factors.
6. At runtime, GRAY-LBPH produces the candidate list. CA-ME-LBPH reranks only within the configured top-k candidates.

### Runtime Decision

The runtime first checks whether the primary GRAY-LBPH result needs rerank consideration. If the first-stage confidence and gray margin are already sufficiently safe, it keeps the original label. Otherwise, it computes normalized candidate scores from primary gray distance, auxiliary gray distance, color evidence, texture evidence, and quality-aware reliability. The final label changes only when the best secondary candidate beats the original label by the configured switch margin.

### Artifacts

Local training creates:

```text
gray_model.xml
gray_aux_model.xml
evidence_index.npz
label_mapping.json
preprocess_config.json
rerank_config.json
training_report.json
```

The submission packager maps `gray_model.xml` to `face_recognizer_model.xml`, copies the auxiliary model and evidence index, writes runtime config files, and builds `Algorithm.tar.gz`. These generated artifacts are excluded from Git.

### Risk Boundary

CA-ME-LBPH reduces some close-candidate confusion in controlled classroom data, but it is still sensitive to the same acquisition assumptions as the underlying LBPH pipeline. Color and quality evidence can shift under different lighting, camera white balance, background color, compression, or blur. For every parameter search or new dataset, report both rerank help and rerank harm.

## 中文

CA-ME-LBPH 是 Confusion-Aware Multi-Evidence LBPH。它不是独立分类器，而是在 GRAY-LBPH 之上增加的保守二级重排层。

### 数据流

1. 训练集按身份目录组织。
2. 输入读取使用 score2026 风格 adapter 和固定预处理配置。
3. 主 GRAY-LBPH 识别器使用 `400x450` 灰度图训练。
4. 辅助粗粒度 GRAY-LBPH 识别器使用 `200x200` 灰度图训练。
5. 训练阶段为每张图片保存多证据特征：
   - Lab a/b、HSV H/S 和 chromaticity 颜色直方图；
   - 局部灰度纹理直方图和梯度统计；
   - 亮度、对比度、Laplacian 模糊度和颜色可靠性因子。
6. 运行时先由 GRAY-LBPH 生成候选列表，CA-ME-LBPH 只在配置的 top-k 候选内重排。

### 运行时判断

runtime 会先判断第一阶段 GRAY-LBPH 是否需要二级判断。如果第一阶段置信距离和灰度 margin 已经足够安全，则保持原标签。否则，系统会综合主灰度距离、辅助灰度距离、颜色证据、纹理证据和质量可靠性，计算候选分数。只有当最佳二级候选相对原标签超过配置的 switch margin 时，最终标签才会切换。

### 训练产物

本地训练会生成：

```text
gray_model.xml
gray_aux_model.xml
evidence_index.npz
label_mapping.json
preprocess_config.json
rerank_config.json
training_report.json
```

打包器会将 `gray_model.xml` 映射为 `face_recognizer_model.xml`，复制辅助模型和 evidence index，写入 runtime 配置文件，并生成 `Algorithm.tar.gz`。这些生成产物不会提交到 Git。

### 风险边界

CA-ME-LBPH 可以减少受控课堂数据中的一部分相近候选混淆，但它仍然依赖 LBPH 管线的采集假设。光照、相机白平衡、背景颜色、压缩和模糊都会影响颜色与质量证据。每次参数搜索或迁移到新数据集时，都应同时报告 `rerank_help` 和 `rerank_harm`。
