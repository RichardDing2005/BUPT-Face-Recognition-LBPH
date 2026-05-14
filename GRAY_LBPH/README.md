# GRAY-LBPH Face Recognition

## English

GRAY-LBPH is the stable baseline model for the BUPT classroom face-recognition project. It uses OpenCV's Local Binary Patterns Histograms recognizer on grayscale images, with fixed preprocessing and a score2026-style submission template. This subproject is complete for standalone use and includes its own source code, requirements, tests, and configuration examples.

The model was trained and tested on face images collected in class. The classroom dataset contains approximately 90 identities and approximately 7,500 images. On the held-out test split from this same classroom collection, GRAY-LBPH reaches more than 98.5% accuracy and provides the baseline that RGB-LBPH improves on.

### Algorithm Notes

- Input images are converted to grayscale before LBPH training or prediction.
- The benchmark profile resizes images to `400x450`, applies CLAHE, and uses `radius=2`, `neighbors=8`, `grid_x=10`, `grid_y=11`.
- During the reported classroom evaluation, face detection is disabled. The model uses full-image recognition or a fixed preprocessing profile instead of dynamically detecting and cropping a face.
- This setup is intentionally aligned with the classroom data format and score2026-style runtime.

### Risk Boundary

The reported accuracy is tied to a controlled classroom collection process. Training and testing images share similar clothing, background, camera framing, and ambient lighting. The model may not generalize well to changed clothing, different illumination, different camera positions, or open-world face-recognition scenarios. Any use of trained models or benchmark numbers must document this generalization boundary.

### What Is Included

- `src/`: reusable training, preprocessing, prediction, evaluation, packaging, and dataset utilities.
- `train_LBPH.py`: compatibility training entrypoint.
- `test_face.py`: evaluation entrypoint.
- `split_dataset.py`: stratified dataset split helper.
- `staged_train.py`: staged training and resume flow.
- `prepare_public_dataset.py`: AT&T/ORL public-data preparation helper.
- `configs/`: benchmark-oriented profile examples.
- `tests/`: unit and workflow tests.
- `submission_template/Algorithm/`: runtime template for benchmark submission packaging.
- `LBPH_training_testing_evaluation.ipynb`: optional notebook workflow.

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`jupyter`, `ipykernel`, `pandas`, and `matplotlib` are included for the optional notebook and analysis workflow. The core CLI path uses OpenCV, NumPy, and Pillow.

### Data Layout

Use one folder per identity:

```text
datasets/
  score2026/
    Faces_raw/
      <identity>/
        *.jpg
```

The repository intentionally ignores `datasets/` and image payloads.

### Train

```powershell
python .\train_LBPH.py `
  --workspace . `
  --train-dir datasets/score2026/Faces_raw `
  --algorithm-dir Algorithm_GRAY_LBPH `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --input-adapter score2026_framework `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11
```

The generated model directory contains:

```text
face_recognizer_model.xml
label_mapping.json
preprocess_config.json
training_report.json
```

### Evaluate

```powershell
python .\test_face.py `
  --workspace . `
  --test-dir datasets/score2026/Faces_test `
  --algorithm-dir Algorithm_GRAY_LBPH `
  --reports-dir reports/GRAY_LBPH_eval
```

### Build A Submission Runtime

The template at `submission_template/Algorithm` already contains runtime code and interface stubs. After training, copy the generated files into that folder or another package directory:

```text
face_recognizer_model.xml
label_mapping.json
preprocess_config.json
```

### Test

```powershell
python -m compileall -q .
python -m pytest tests -q --basetemp .tmp_pytest_GRAY_LBPH
Remove-Item -LiteralPath .\.tmp_pytest_GRAY_LBPH -Recurse -Force
```

## 中文

GRAY-LBPH 是 BUPT 课堂人脸识别项目中的稳定基线模型。该子项目可独立使用，包含源码、依赖、测试、配置示例和 score2026 风格提交模板。

该模型基于课堂上实际采集的人脸数据进行训练和测试。本地课堂数据集约 90 人、约 7500 张图片。在同一课堂采集条件下划分出的测试集上，GRAY-LBPH 准确率达到 98.5% 以上，并作为 RGB-LBPH 继续提升的基线。

### 算法说明

- 输入图像在训练或预测前会转为灰度图。
- benchmark 配置会将图像 resize 到 `400x450`，使用 CLAHE，并采用 `radius=2`、`neighbors=8`、`grid_x=10`、`grid_y=11`。
- 本次课堂测试中 face detection disabled，即关闭人脸检测。模型使用整图识别或固定预处理配置，而不是动态检测并裁剪人脸区域。
- 这种设置与课堂数据格式和 score2026 风格运行时保持一致。

### 风险边界

测试准确率依赖受控课堂采集流程。训练集和测试集具有相似的服饰、背景、取景方式和环境光条件。模型在服饰变化、光照变化、摄像头位置变化或开放场景人脸识别中可能无法保持同等表现。任何使用训练模型或 benchmark 数字的说明都必须同步标注这一 generalization risk。

### 包含内容

- `src/`：训练、预处理、预测、评估、打包和数据集工具。
- `train_LBPH.py`：兼容式训练入口。
- `test_face.py`：评估入口。
- `split_dataset.py`：分层数据切分工具。
- `staged_train.py`：阶段训练和恢复流程。
- `prepare_public_dataset.py`：AT&T/ORL 公开数据准备工具。
- `configs/`：面向 benchmark 的配置示例。
- `tests/`：单元测试和流程测试。
- `submission_template/Algorithm/`：benchmark 提交运行时模板。
- `LBPH_training_testing_evaluation.ipynb`：可选 notebook 工作流。

### 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`jupyter`、`ipykernel`、`pandas` 和 `matplotlib` 用于可选 notebook 和分析流程。核心命令行流程主要依赖 OpenCV、NumPy 和 Pillow。

### 数据结构

每个身份一个文件夹：

```text
datasets/
  score2026/
    Faces_raw/
      <identity>/
        *.jpg
```

仓库会刻意忽略 `datasets/` 和图片 payload。

### 训练

```powershell
python .\train_LBPH.py `
  --workspace . `
  --train-dir datasets/score2026/Faces_raw `
  --algorithm-dir Algorithm_GRAY_LBPH `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --input-adapter score2026_framework `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11
```

生成的模型目录包含：

```text
face_recognizer_model.xml
label_mapping.json
preprocess_config.json
training_report.json
```

### 评估

```powershell
python .\test_face.py `
  --workspace . `
  --test-dir datasets/score2026/Faces_test `
  --algorithm-dir Algorithm_GRAY_LBPH `
  --reports-dir reports/GRAY_LBPH_eval
```

### 构建提交运行时

`submission_template/Algorithm` 已包含运行时代码和接口桩。训练后，将生成文件复制到该目录或其他打包目录：

```text
face_recognizer_model.xml
label_mapping.json
preprocess_config.json
```

### 测试

```powershell
python -m compileall -q .
python -m pytest tests -q --basetemp .tmp_pytest_GRAY_LBPH
Remove-Item -LiteralPath .\.tmp_pytest_GRAY_LBPH -Recurse -Force
```

## License

GRAY-LBPH is released under the MIT License.
