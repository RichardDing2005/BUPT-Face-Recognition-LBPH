# CA-ME-LBPH Face Recognition

## English

CA-ME-LBPH is the confusion-aware multi-evidence LBPH model for the BUPT classroom face-recognition project. It keeps GRAY-LBPH as the first-stage recognizer and adds a second-stage rerank step for close GRAY-LBPH candidates. This subproject is organized for standalone source release and includes training, evaluation, parameter search, dataset augmentation helpers, tests, and a score2026-style submission runtime template.

The model was designed for a controlled classroom benchmark. The local development data used identity-organized classroom face images, with fixed image framing and face detection disabled during the reported score2026-style runtime. The repository does not include private images, identity mappings, trained XML models, NPZ evidence indexes, or packaged `Algorithm.tar.gz` archives.

### Algorithm Notes

- GRAY-LBPH produces the primary top-k candidate list.
- CA-ME-LBPH only reranks labels inside the GRAY-LBPH top-k set.
- The secondary evidence combines an auxiliary coarse GRAY-LBPH model, Lab/HSV/chromaticity color histograms, local grayscale texture statistics, and image quality factors.
- A switch is made only when the secondary score has a configured margin over the first-stage label.
- The default score2026 profile uses `400x450`, CLAHE, face detection disabled, `radius=2`, `neighbors=8`, and `grid_x=10`, `grid_y=11`.

### Risk Boundary

CA-ME-LBPH is still a classical LBPH-based benchmark model. Its reported behavior is tied to the acquisition setting: similar camera framing, background, clothing, and lighting. The multi-evidence rerank can improve difficult classroom confusions, but it is not an open-world face-recognition claim. Color and quality evidence can be affected by illumination, white balance, blur, compression, and background color.

### Included Source

- `src/confusion_rerank.py`: CA-ME model configuration, training, prediction, secondary evidence, and top-k rerank logic.
- `src/confusion_param_search.py`: 5-fold style parameter search and aggressive runtime parameter selection.
- `src/confusion_submission_package.py`: conversion from training artifacts to score2026 runtime package layout.
- `src/augmentation_preview.py` and `src/dataset_augmentation.py`: reproducible light-noise/light-blur preview and dataset augmentation helpers.
- `submission_template/Algorithm/`: benchmark runtime implementation and interface stubs.
- `tests/`: unit and runtime-template consistency tests for CA-ME rerank, packaging, and augmentation helpers.

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Data Layout

Use one folder per identity. Image payloads are intentionally ignored by Git.

```text
datasets/
  score2026/
    Faces_raw/
      <identity>/
        *.jpg
```

For holdout evaluation:

```text
datasets/
  score2026_v4/
    Faces_train/<identity>/*.jpg
    Faces_test/<identity>/*.jpg
```

### Train

```powershell
python .\train_confusion_rerank_lbph.py `
  --train-dir .\datasets\score2026\Faces_raw `
  --output-dir .\Algorithm_score2026_confusion_rerank_full `
  --resize 400x450 `
  --equalization clahe `
  --no-detect-face `
  --input-adapter score2026_framework `
  --radius 2 `
  --neighbors 8 `
  --grid-x 10 `
  --grid-y 11 `
  --aux-resize 200x200 `
  --aux-grid-x 7 `
  --aux-grid-y 7
```

Training creates local artifacts that are not committed:

```text
gray_model.xml
gray_aux_model.xml
evidence_index.npz
label_mapping.json
preprocess_config.json
rerank_config.json
training_report.json
```

### Evaluate

```powershell
python .\evaluate_confusion_rerank_lbph.py `
  --model-dir .\Algorithm_score2026_confusion_rerank_full `
  --test-dir .\datasets\score2026_v4\Faces_test `
  --reports-dir .\reports\confusion_rerank_eval `
  --candidate-top-k 4 `
  --confidence-gate 60 `
  --gray-margin-gate 65 `
  --switch-margin 0.05
```

The evaluation report records overall accuracy, error cases, `rerank_help`, `rerank_harm`, and `net_gain`.

### Parameter Search

```powershell
python .\search_confusion_rerank_params.py `
  --raw-dir .\datasets\score2026\Faces_raw `
  --output-dir .\experiments\confusion_rerank_param_search_5fold `
  --folds 5 `
  --seed 42
```

### Optional Augmentation Preview

The augmentation helpers are included for reproducibility experiments only. Generated augmented datasets are local artifacts and are not part of this source release.

```powershell
python .\preview_light_augmentations.py `
  --source-image .\datasets\score2026\Faces_raw\<identity>\1.jpg `
  --output-dir C:\tmp\lbph_augmentation_preview

python .\augment_score2026_dataset.py `
  --source-dir .\datasets\score2026\Faces_raw `
  --output-dir .\datasets\score2026_aug_light\Faces_raw `
  --manifest-path .\datasets\score2026_aug_light\augmentation_manifest.csv `
  --report-path .\datasets\score2026_aug_light\dataset_report.json `
  --overwrite
```

### Build A Submission Runtime

```powershell
python .\build_confusion_submission_package.py `
  --model-dir .\Algorithm_score2026_confusion_rerank_full `
  --output-dir .\submission_build\confusion_rerank `
  --candidate-top-k 4 `
  --confidence-gate 60 `
  --gray-margin-gate 65 `
  --switch-margin 0.05
```

The generated package contains runtime files under `Algorithm/` and an `Algorithm.tar.gz` archive. Those files are generated locally and excluded from Git.

### Test

```powershell
python -m compileall -q .
python -m pytest tests -q --basetemp .tmp_pytest_CA_ME_LBPH
Remove-Item -LiteralPath .\.tmp_pytest_CA_ME_LBPH -Recurse -Force
```

## 中文

CA-ME-LBPH 是面向 BUPT 课堂人脸识别任务的 Confusion-Aware Multi-Evidence LBPH 模型。它保留 GRAY-LBPH 作为第一阶段识别器，只在 GRAY-LBPH top-k 候选内部使用多证据二级重排。该子项目按独立源码发布整理，包含训练、评估、参数搜索、轻度增强辅助程序、测试和 score2026 风格提交运行时模板。

该模型服务于受控课堂 benchmark。开发数据采用按身份目录组织的课堂采集人脸图像，报告运行时关闭人脸检测并使用固定图像预处理。仓库不包含私有图片、真实身份映射、训练后的 XML 模型、NPZ evidence index 或最终 `Algorithm.tar.gz` 提交包。

### 算法说明

- 第一阶段由 GRAY-LBPH 生成候选标签和置信距离。
- CA-ME-LBPH 的二级判断只在 GRAY-LBPH top-k 候选内运行。
- 二级证据包括辅助粗粒度 GRAY-LBPH、Lab/HSV/chromaticity 颜色直方图、局部灰度纹理统计和图像质量因子。
- 只有当二级分数相对第一阶段标签具有明确 margin 时，最终标签才会切换。
- 默认 score2026 配置为 `400x450`、CLAHE、关闭人脸检测、`radius=2`、`neighbors=8`、`grid_x=10`、`grid_y=11`。

### 风险边界

CA-ME-LBPH 仍然是经典 LBPH 路线下的 benchmark 模型。其表现依赖课堂采集条件：相近的取景、背景、服装和光照。多证据重排可以改善课堂数据中的困难混淆，但不能直接视为开放场景泛化能力。颜色与质量证据会受到光照、白平衡、模糊、压缩和背景颜色影响。

### 本项目包含

- `src/confusion_rerank.py`：CA-ME 配置、训练、预测、二级证据和 top-k 重排逻辑。
- `src/confusion_param_search.py`：参数搜索与 aggressive runtime 参数选择。
- `src/confusion_submission_package.py`：将训练产物转换为 score2026 runtime 文件结构。
- `src/augmentation_preview.py` 与 `src/dataset_augmentation.py`：轻度噪声/模糊预览和批量增强辅助工具。
- `submission_template/Algorithm/`：benchmark 运行时实现和接口 stub。
- `tests/`：CA-ME 重排、打包、runtime 预处理一致性和增强辅助逻辑测试。

### 数据与产物策略

训练数据按身份目录放置，但 `datasets/`、图片、模型权重、`evidence_index.npz` 和提交压缩包都不会进入 Git。使用者需要在本地用具备授权的数据生成这些运行文件。

### 训练、评估、打包

训练、评估和打包命令与英文部分一致。推荐先使用 holdout 划分验证 `rerank_help / rerank_harm / net_gain`，确认二级重排没有伤害已有正确样本后，再使用全量授权数据训练最终模型。
