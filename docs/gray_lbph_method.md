# GRAY-LBPH Method

## English

GRAY-LBPH uses the standard Local Binary Patterns Histograms recognizer from OpenCV. The pipeline converts each input image to grayscale, resizes it to the configured benchmark shape, applies the configured contrast-normalization method, and trains or predicts with LBPH histograms.

The current classroom benchmark profile is:

- Color handling: convert RGB/BGR input to grayscale.
- Region policy: full-image recognition.
- Face detection: disabled.
- Resize: `400x450`.
- Equalization: CLAHE.
- LBPH parameters: `radius=2`, `neighbors=8`, `grid_x=10`, `grid_y=11`, `threshold=None`.

The model was trained and tested on classroom-collected face data with approximately 90 identities and approximately 7,500 images. It reaches more than 98.5% accuracy on the test split from the same collection setting.

The main limitation is the acquisition boundary. Because face detection is disabled and the images share similar clothing, background, framing, and ambient lighting, the measured score is a classroom benchmark result rather than an open-world face-recognition guarantee.

## 中文

GRAY-LBPH 使用 OpenCV 标准 Local Binary Patterns Histograms 识别器。流程会将输入图像转为灰度图，resize 到配置的 benchmark 尺寸，执行配置的对比度归一化方法，然后使用 LBPH 直方图进行训练或预测。

当前课堂 benchmark 配置为：

- 颜色处理：将 RGB/BGR 输入转为灰度图。
- 区域策略：整图识别。
- 人脸检测：关闭。
- Resize：`400x450`。
- 均衡化：CLAHE。
- LBPH 参数：`radius=2`、`neighbors=8`、`grid_x=10`、`grid_y=11`、`threshold=None`。

模型基于课堂采集人脸数据训练和测试，数据规模约 90 人、约 7500 张图片。在同一采集条件下的测试集上，准确率达到 98.5% 以上。

主要限制来自采集边界。由于 face detection disabled，且图片具有相似服饰、背景、取景方式和环境光条件，该分数应理解为课堂 benchmark 结果，而不是开放场景人脸识别保证。
