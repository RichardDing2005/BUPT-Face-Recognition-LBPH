# RGB-LBPH Face Recognition

## English

RGB-LBPH is the course project's SOTA model. This subproject is complete for standalone use and includes its own source code, requirements, tests, configuration, packaging script, and score2026-style submission template.

The model was trained and tested on the same classroom face dataset used by GRAY-LBPH, with approximately 90 identities and approximately 7,500 images. On the held-out test split from this same collection setting, the project reaches more than 98.5% accuracy, and RGB-LBPH improves over plain GRAY-LBPH on the most similar samples.

### Algorithm Notes

- GRAY-LBPH produces the primary label and confidence.
- RGB-LBPH then uses RGB-channel color histograms as an additional criterion inspired by the LBPH idea of comparing local distributions.
- The RGB evidence is restricted to the configured GRAY-LBPH top-k candidate labels; it is not a global color-only recognizer.
- The prediction changes only when the RGB nearest-neighbor margin is strong enough.
- During the reported classroom evaluation, face detection is disabled and the same fixed image profile is used as the GRAY-LBPH baseline.

### Risk Boundary

RGB-LBPH is more sensitive to lighting than GRAY-LBPH. Because the additional criterion directly depends on R, G, and B channel values, changes in light intensity, light color, camera white balance, clothing reflection, or background color can affect the rerank decision. The reported improvement is meaningful for the classroom benchmark; it is not an open-world generalization claim without external validation under varied lighting and acquisition conditions.

### Default Profile

- Image size: `400x450`
- Face detection: disabled
- Equalization: CLAHE
- LBPH: `radius=2`, `neighbors=8`, `grid_x=10`, `grid_y=11`
- Color feature: per-channel RGB histograms with `8` bins per channel
- Rerank search: GRAY-LBPH top-k candidate labels

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Data Layout

Use one folder per identity:

```text
datasets/
  score2026_holdout/
    Faces_train/
      <identity>/
        *.jpg
    Faces_test/
      <identity>/
        *.jpg
```

The repository intentionally ignores `datasets/` and image payloads.

### Parameter Search

```powershell
python .\search_color_rerank_params.py `
  --raw-dir "datasets\score2026\Faces_raw" `
  --output-dir "experiments\color_rerank_param_search_5fold" `
  --folds 5 `
  --seed 42 `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11 `
  --color-bins 8
```

### Train

```powershell
python .\train_color_rerank_lbph.py `
  --train-dir "datasets\score2026_holdout\Faces_train" `
  --output-dir "experiments\GRAY_LBPH_color_rerank_60_40" `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11 `
  --color-bins 8
```

Training creates:

```text
gray_model.xml
color_index.npz
label_mapping.json
rerank_config.json
training_report.json
```

### Evaluate

```powershell
python .\evaluate_color_rerank_lbph.py `
  --test-dir "datasets\score2026_holdout\Faces_test" `
  --model-dir "experiments\GRAY_LBPH_color_rerank_60_40" `
  --reports-dir "experiments\GRAY_LBPH_color_rerank_60_40\reports" `
  --top-k 2 `
  --confidence-gate 60 `
  --margin-ratio 0.10
```

### Build A Submission Runtime

`build_submission_package.py` converts training outputs into the score2026-style runtime names expected by `submission_template/Algorithm`.

```powershell
python .\build_submission_package.py `
  --model-dir "experiments\GRAY_LBPH_color_rerank_60_40" `
  --output-dir "submission_build\color_rerank" `
  --candidate-top-k 2 `
  --confidence-gate 70 `
  --rerank-margin-ratio 0 `
  --tar-path "submission_build\Algorithm.tar.gz"
```

Generated runtime contents:

```text
Algorithm/
  AlgorithmImplement.py
  Interface/
  face_recognizer_model.xml
  color_index.npz
  label_mapping.json
  preprocess_config.json
  rerank_runtime_config.json
  requirements.txt
  training_report.json
```

### Test

```powershell
python -m compileall -q .
python -m pytest tests -q --basetemp .tmp_pytest_RGB_LBPH
Remove-Item -LiteralPath .\.tmp_pytest_RGB_LBPH -Recurse -Force
```

## 中文

RGB-LBPH 是本次课程项目的 SOTA 模型。该子项目可独立使用，包含源码、依赖、测试、配置、打包脚本和 score2026 风格提交模板。

该模型与 GRAY-LBPH 使用同一课堂人脸数据集进行训练和测试，数据规模约 90 人、约 7500 张图片。在同一采集条件下划分出的测试集上，本项目模型准确率达到 98.5% 以上；其中 RGB-LBPH 在最相似样本上相较单纯 GRAY-LBPH 有进一步提升。

### 算法说明

- GRAY-LBPH 首先给出主预测标签和 confidence。
- RGB-LBPH 随后将 LBPH 比较局部分布的思想引入 R/G/B 三个通道的颜色直方图，作为附加判据。
- RGB 证据只在配置的 GRAY-LBPH top-k 候选标签内部进行比较，不是全局颜色最近邻识别器。
- 只有当 RGB 最近邻距离优势足够明显时，最终预测才会从 GRAY-LBPH 标签切换到颜色判据支持的标签。
- 本次课堂测试中 face detection disabled，并且使用与 GRAY-LBPH 基线一致的固定图像预处理配置。

### 风险边界

RGB-LBPH 比 GRAY-LBPH 更依赖光照环境。由于附加判据直接依赖 R、G、B 通道数值，光强、光颜色、相机白平衡、服饰反光和背景颜色变化都可能影响重排序判断。该提升对课堂 benchmark 有意义；在没有经过多光照、多采集环境和外部测试集验证前，它不是开放场景泛化能力声明。

### 默认配置

- 图像尺寸：`400x450`
- 人脸检测：关闭
- 均衡化：CLAHE
- LBPH：`radius=2`、`neighbors=8`、`grid_x=10`、`grid_y=11`
- 颜色特征：每个 RGB 通道使用 `8` bins 直方图
- 重排序范围：GRAY-LBPH top-k 候选标签

### 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 数据结构

每个身份一个文件夹：

```text
datasets/
  score2026_holdout/
    Faces_train/
      <identity>/
        *.jpg
    Faces_test/
      <identity>/
        *.jpg
```

仓库会刻意忽略 `datasets/` 和图片 payload。

### 参数搜索

```powershell
python .\search_color_rerank_params.py `
  --raw-dir "datasets\score2026\Faces_raw" `
  --output-dir "experiments\color_rerank_param_search_5fold" `
  --folds 5 `
  --seed 42 `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11 `
  --color-bins 8
```

### 训练

```powershell
python .\train_color_rerank_lbph.py `
  --train-dir "datasets\score2026_holdout\Faces_train" `
  --output-dir "experiments\GRAY_LBPH_color_rerank_60_40" `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11 `
  --color-bins 8
```

训练会生成：

```text
gray_model.xml
color_index.npz
label_mapping.json
rerank_config.json
training_report.json
```

### 评估

```powershell
python .\evaluate_color_rerank_lbph.py `
  --test-dir "datasets\score2026_holdout\Faces_test" `
  --model-dir "experiments\GRAY_LBPH_color_rerank_60_40" `
  --reports-dir "experiments\GRAY_LBPH_color_rerank_60_40\reports" `
  --top-k 2 `
  --confidence-gate 60 `
  --margin-ratio 0.10
```

### 构建提交运行时

`build_submission_package.py` 会把训练输出转换成 `submission_template/Algorithm` 运行时需要的 score2026 风格文件名。

```powershell
python .\build_submission_package.py `
  --model-dir "experiments\GRAY_LBPH_color_rerank_60_40" `
  --output-dir "submission_build\color_rerank" `
  --candidate-top-k 2 `
  --confidence-gate 70 `
  --rerank-margin-ratio 0 `
  --tar-path "submission_build\Algorithm.tar.gz"
```

生成的运行时目录：

```text
Algorithm/
  AlgorithmImplement.py
  Interface/
  face_recognizer_model.xml
  color_index.npz
  label_mapping.json
  preprocess_config.json
  rerank_runtime_config.json
  requirements.txt
  training_report.json
```

### 测试

```powershell
python -m compileall -q .
python -m pytest tests -q --basetemp .tmp_pytest_RGB_LBPH
Remove-Item -LiteralPath .\.tmp_pytest_RGB_LBPH -Recurse -Force
```

## License

RGB-LBPH is released under the MIT License.
