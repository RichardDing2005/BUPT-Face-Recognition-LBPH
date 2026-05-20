# Face Recognition LBPH BUPT

## English

This repository is a standalone source release for a BUPT classroom face-recognition project. It is organized as three peer LBPH subprojects that can be installed, tested, and used independently:

- `GRAY_LBPH/`: the stable grayscale OpenCV LBPH baseline.
- `RGB_LBPH/`: a color-rerank model that keeps GRAY-LBPH as the first stage and reranks close candidates with RGB histogram evidence.
- `CA_ME_LBPH/`: a confusion-aware multi-evidence rerank model that uses auxiliary gray LBPH, color histograms, texture statistics, and quality factors inside the GRAY-LBPH top-k candidate set.

The algorithms were developed for classroom-collected face images. The reported benchmark setting uses fixed preprocessing and disables face detection at runtime. These results are meaningful for the controlled classroom benchmark, but they are not open-world generalization claims. Lighting changes, clothing changes, camera framing shifts, background changes, blur, and compression can all affect the behavior of these classical LBPH-based models.

### LBPH Algorithm Overview

Local Binary Patterns Histograms (LBPH) is a classical face-recognition method. It describes local texture by comparing each pixel with neighboring pixels, encodes the local binary pattern, and aggregates these codes into histograms over image grid cells. During recognition, the test histogram is compared with stored identity histograms. LBPH is lightweight, interpretable, and effective when image size, alignment, illumination processing, and input format are consistent.

GRAY-LBPH applies the standard OpenCV LBPH recognizer to grayscale texture. RGB-LBPH adds a color-channel rerank step for highly similar samples. CA-ME-LBPH generalizes the rerank idea: it still trusts GRAY-LBPH for the candidate set, but combines auxiliary gray evidence, Lab/HSV/chromaticity color histograms, local texture statistics, and image quality features before changing the final label.

One practical lesson from this project is that the default input size is part of the benchmark contract. The score2026 cloud benchmark images used by this project are aligned to `400x450`, so the default preprocessing profile also uses `400x450`. In real collection or deployment settings, face detection and face-region cropping should usually be enabled; in that case, the image actually processed by LBPH will be smaller than the original camera frame. Captured images may be JPEG-compressed, but their aspect ratio should not be changed. LBPH is sensitive to edge and local texture distributions, and geometric stretching can alter those edge textures enough to affect recognition.

### Project Layout

```text
Face_Recognition_LBPH_BUPT/
  GRAY_LBPH/
    README.md
    requirements.txt
    src/
    tests/
    submission_template/Algorithm/
  RGB_LBPH/
    README.md
    requirements.txt
    src/
    tests/
    submission_template/Algorithm/
  CA_ME_LBPH/
    README.md
    requirements.txt
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
git sparse-checkout set CA_ME_LBPH
```

For the other variants:

```powershell
git sparse-checkout set GRAY_LBPH
git sparse-checkout set RGB_LBPH
```

You can also download the repository ZIP from GitHub and keep only the subproject folder you need.

### Install For Full Repository Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For independent subproject usage, enter the subproject directory and install that folder's `requirements.txt`.

### Data And Artifact Policy

This repository intentionally excludes:

- raw face images and identity folders;
- real label mappings that identify people;
- trained `.xml` models;
- `.npz` evidence indexes;
- generated reports, experiments, and `Algorithm.tar.gz` submission archives.

Users should train and package runtime artifacts locally from their own authorized data.

### Verification

```powershell
python -m compileall -q GRAY_LBPH RGB_LBPH CA_ME_LBPH
python -m pytest GRAY_LBPH/tests RGB_LBPH/tests CA_ME_LBPH/tests -q --basetemp .tmp_pytest_lbph_publish
Remove-Item -LiteralPath .\.tmp_pytest_lbph_publish -Recurse -Force
```

The repo-local `--basetemp` keeps pytest temporary files inside this workspace so they can be removed after verification.

## 中文

本仓库是北京邮电大学（BUPT）电子工程学院的初级项目课人脸识别项目的独立源码发布版本。项目由三个平行 LBPH 子项目组成，每个子项目都可以单独安装、测试和使用：

- `GRAY_LBPH/`：稳定的 OpenCV 灰度 LBPH 基线。
- `RGB_LBPH/`：保留 GRAY-LBPH 作为第一阶段，并使用 RGB 直方图证据对相近候选进行颜色重排。
- `CA_ME_LBPH/`：Confusion-Aware Multi-Evidence LBPH，在 GRAY-LBPH top-k 候选内结合辅助灰度 LBPH、颜色直方图、纹理统计和质量因子进行二级判断。

这些算法面向课堂采集人脸图像开发。报告中的 benchmark 使用固定预处理，并在运行时关闭人脸检测。结果适用于受控课堂 benchmark，不应直接解释为开放场景泛化能力。光照、服装、取景、背景、模糊和压缩变化都可能影响经典 LBPH 路线模型的表现。

### LBPH 算法简介

Local Binary Patterns Histograms (LBPH) 是一种经典人脸识别方法。它通过比较像素与邻域像素的灰度关系描述局部纹理，再把局部二值模式编码按图像网格统计成直方图。识别时，测试图像直方图会与训练身份的直方图进行比较。LBPH 计算量小、可解释性强，在图像尺寸、对齐方式、光照处理和输入格式一致的受控 benchmark 中表现稳定。

本仓库中，GRAY-LBPH 使用标准 OpenCV LBPH 建模灰度纹理；RGB-LBPH 在相近样本上增加颜色通道重排；CA-ME-LBPH 进一步把重排扩展为多证据二级判断，但仍只在 GRAY-LBPH top-k 候选内部工作。

本项目得到的一个实践经验是：默认输入尺寸本身也是 benchmark 契约的一部分。本次 score2026 云端 benchmark 使用的图片规格与项目输入对齐为 `400x450`，因此当前默认预处理也采用 `400x450`。在真实采集或部署场景中，通常建议开启人脸检测和人脸区域裁切；此时 LBPH 实际处理的图像区域会小于原始相机画面。采集图片可以进行 JPEG 压缩，但不建议改变图片宽高比例。LBPH 对边缘纹理和局部分布较敏感，拉伸或压缩比例变化会改变边缘纹理，从而影响识别结果。

### 项目结构

```text
Face_Recognition_LBPH_BUPT/
  GRAY_LBPH/
  RGB_LBPH/
  CA_ME_LBPH/
  docs/
  requirements.txt
```

### 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果只使用某一个子项目，进入对应子目录并安装该目录下的 `requirements.txt` 即可。

### 数据与产物边界

仓库不会提交原始人脸图片、真实身份目录、真实标签映射、训练后的 `.xml` 模型、`.npz` evidence index、实验报告目录或 `Algorithm.tar.gz` 提交包。使用者需要基于自己具备授权的数据在本地训练并生成运行产物。

### 验证

```powershell
python -m compileall -q GRAY_LBPH RGB_LBPH CA_ME_LBPH
python -m pytest GRAY_LBPH/tests RGB_LBPH/tests CA_ME_LBPH/tests -q --basetemp .tmp_pytest_lbph_publish
Remove-Item -LiteralPath .\.tmp_pytest_lbph_publish -Recurse -Force
```

使用仓库内 `--basetemp` 可以把 pytest 临时文件限制在当前工作区，方便测试后清理。

## License

This source repository is released under the MIT License.
