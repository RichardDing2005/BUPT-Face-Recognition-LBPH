# Face Recognition LBPH BUPT

## English

This repository is a complete standalone source repository for a BUPT classroom face-recognition project. It is built around two LBPH subprojects that can be pulled, installed, tested, and run independently:

- `GRAY_LBPH/`: the stable GRAY-LBPH baseline based on OpenCV grayscale LBPH features.
- `RGB_LBPH/`: the RGB-LBPH color-rerank model, built on top of GRAY-LBPH and used as the course project's SOTA model.

The algorithms were trained and tested on face images collected in class. The local classroom dataset contains approximately 90 identities and approximately 7,500 images. On the held-out test split from this same collection setting, the models reach more than 98.5% accuracy, and RGB-LBPH improves over plain GRAY-LBPH on especially similar or easily confused images.

This accuracy is bounded by its experimental setting. The data was collected under similar clothing, camera framing, background, and ambient lighting conditions. During testing, face detection is disabled, so the runtime uses the full image or fixed preprocessing profile rather than detecting and cropping a face region dynamically. This makes the result strong for the classroom benchmark, but it also creates a clear generalization risk for clothing changes, lighting shifts, different backgrounds, and open-world deployment.

### LBPH Algorithm Overview

Local Binary Patterns Histograms (LBPH) is a classical face-recognition method that describes local texture around each pixel by comparing it with neighboring pixels, then aggregates those binary pattern codes into histograms over image grid cells. During recognition, the histogram of a test image is compared with the stored histograms of training identities. LBPH is lightweight, interpretable, and effective for controlled classroom benchmarks, especially when image size, alignment, illumination processing, and input format are consistent.

In this repository, GRAY-LBPH applies the standard OpenCV LBPH recognizer to grayscale facial texture. RGB-LBPH keeps that GRAY-LBPH prediction as the primary decision and adds a color-channel rerank step for highly similar samples, using R/G/B histogram evidence as an additional criterion.

### Model Design

GRAY-LBPH follows a compact classical pipeline: it reads identity-organized image folders, applies a fixed preprocessing profile, trains an OpenCV LBPH recognizer, saves the recognizer with the label mapping and preprocessing configuration, and uses the same preprocessing path during evaluation or submission runtime prediction.

RGB-LBPH keeps GRAY-LBPH as the primary recognizer and adds a second-stage color rerank module. The training program builds the GRAY-LBPH model first, then indexes RGB histogram features by identity. At prediction time, the runtime evaluates only the configured GRAY-LBPH top-k candidates and changes the final label only when the RGB evidence gives a clear margin. This makes the RGB stage an auxiliary rerank criterion rather than an independent color-only classifier.

For command-level usage, parameter choices, runtime entrypoints, and risk boundaries, see `GRAY_LBPH/README.md` and `RGB_LBPH/README.md`.

### Project Layout

```text
Face_Recognition_LBPH_BUPT/
  GRAY_LBPH/
    README.md
    requirements.txt
    train_LBPH.py
    test_face.py
    split_dataset.py
    staged_train.py
    prepare_public_dataset.py
    src/
    tests/
    submission_template/Algorithm/
  RGB_LBPH/
    README.md
    requirements.txt
    train_color_rerank_lbph.py
    evaluate_color_rerank_lbph.py
    search_color_rerank_params.py
    build_submission_package.py
    src/
    tests/
    submission_template/Algorithm/
  docs/
  requirements.txt
```

### Standalone Checkout

Each subproject can be pulled with Git sparse checkout:

```powershell
git clone --filter=blob:none --sparse <repo-url> Face_Recognition_LBPH_BUPT
cd .\Face_Recognition_LBPH_BUPT
git sparse-checkout set GRAY_LBPH
```

For RGB-LBPH:

```powershell
git sparse-checkout set RGB_LBPH
```

You can also download the repository ZIP from GitHub and keep only the subproject folder you need. Each subproject includes its own README, requirements, license, ignore rules, tests, and submission-template notes.

### Install For Full Repository Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For independent subproject usage, enter the subproject directory and install that folder's `requirements.txt`.

### Verification

```powershell
python -m compileall -q GRAY_LBPH RGB_LBPH
python -m pytest GRAY_LBPH/tests RGB_LBPH/tests -q --basetemp .tmp_pytest_lbph_publish
Remove-Item -LiteralPath .\.tmp_pytest_lbph_publish -Recurse -Force
```

The repo-local `--basetemp` keeps pytest temporary files inside this workspace so they can be removed after verification.

## 中文

本源码仓库作为 GitHub 项目可独立使用，面向 BUPT 课堂人脸识别任务，围绕两个可以独立拉取、安装、测试和运行的 LBPH 子项目组织：

- `GRAY_LBPH/`：稳定的 GRAY-LBPH 基线，基于 OpenCV 灰度 LBPH 特征。
- `RGB_LBPH/`：RGB-LBPH 颜色重排序模型，建立在 GRAY-LBPH 之上，是本次课程项目的 SOTA 模型。

两个算法基于课堂上实际采集的人脸数据进行训练和测试。本地课堂数据集约 90 人、约 7500 张图片。在同一采集条件下划分出的测试集上，模型准确率达到 98.5% 以上；其中 RGB-LBPH 针对特别相似或容易混淆的图片，相比单纯 GRAY-LBPH 有进一步提升。

这个准确率必须放在实验边界内理解。采集数据来自相同或相近的服饰、拍摄背景、取景方式和环境光条件。测试时 face detection disabled，也就是关闭人脸检测，运行时使用整图或固定预处理配置，而不是动态检测并裁剪人脸区域。因此，该结果可以说明模型适合本课程课堂 benchmark，但不能直接解释为开放场景泛化能力；跨服饰、跨光照、跨背景和真实部署仍存在明确 generalization risk。

### LBPH 算法简介

Local Binary Patterns Histograms（LBPH）是一种经典人脸识别方法。它通过比较像素与邻域像素的灰度关系来描述局部纹理，再把这些二值模式编码按图像网格统计成直方图。识别时，测试图像的直方图会与训练身份的直方图进行比较。LBPH 计算量小、可解释性强，在图像尺寸、对齐方式、光照处理和输入格式较一致的受控课堂 benchmark 中表现稳定。

本仓库中，GRAY-LBPH 使用 OpenCV 标准 LBPH 识别器建模灰度人脸纹理；RGB-LBPH 保留 GRAY-LBPH 作为主判据，并针对高度相似样本增加颜色通道重排序步骤，把 R/G/B 直方图证据作为附加判据。

### 模型设计

GRAY-LBPH 采用紧凑的经典识别流程：程序读取按身份组织的图片目录，执行固定预处理配置，训练 OpenCV LBPH 识别器，保存识别器、标签映射和预处理配置，并在评估或提交运行时预测时复用同一套预处理路径。

RGB-LBPH 保留 GRAY-LBPH 作为主识别器，并增加第二阶段颜色重排序模块。训练程序先构建 GRAY-LBPH 模型，再按身份索引 RGB 直方图特征；预测时，运行时只在配置的 GRAY-LBPH top-k 候选标签内评估颜色证据，并且只有在 RGB 判据具有明确距离优势时才切换最终标签。因此，RGB 阶段是辅助重排序判据，而不是独立的纯颜色分类器。

具体命令、参数选择、运行入口和风险边界请参考 `GRAY_LBPH/README.md` 与 `RGB_LBPH/README.md`。

### 项目结构

```text
Face_Recognition_LBPH_BUPT/
  GRAY_LBPH/
    README.md
    requirements.txt
    train_LBPH.py
    test_face.py
    split_dataset.py
    staged_train.py
    prepare_public_dataset.py
    src/
    tests/
    submission_template/Algorithm/
  RGB_LBPH/
    README.md
    requirements.txt
    train_color_rerank_lbph.py
    evaluate_color_rerank_lbph.py
    search_color_rerank_params.py
    build_submission_package.py
    src/
    tests/
    submission_template/Algorithm/
  docs/
  requirements.txt
```

### 独立拉取子项目

每个子项目都可以通过 Git sparse checkout 单独拉取：

```powershell
git clone --filter=blob:none --sparse <repo-url> Face_Recognition_LBPH_BUPT
cd .\Face_Recognition_LBPH_BUPT
git sparse-checkout set GRAY_LBPH
```

RGB-LBPH：

```powershell
git sparse-checkout set RGB_LBPH
```

也可以下载 GitHub ZIP 后只保留需要的子目录。两个子项目都包含独立 README、依赖、许可、忽略规则、测试和提交模板说明。

### 完整仓库开发安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

独立使用某个子项目时，进入对应子目录并安装该目录下的 `requirements.txt`。

### 验证

```powershell
python -m compileall -q GRAY_LBPH RGB_LBPH
python -m pytest GRAY_LBPH/tests RGB_LBPH/tests -q --basetemp .tmp_pytest_lbph_publish
Remove-Item -LiteralPath .\.tmp_pytest_lbph_publish -Recurse -Force
```

使用仓库内 `--basetemp` 可以把 pytest 临时文件限制在当前工作区，方便测试后清理。

## License

This source repository is released under the MIT License.
