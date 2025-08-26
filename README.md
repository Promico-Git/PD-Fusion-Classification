# Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning

## Project Overview

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder that affects millions worldwide, often leading to both motor and non-motor impairments. Early diagnosis is essential but challenging due to reliance on clinical assessment, which can result in misdiagnosis.

This project investigates the use of **feature-level fusion of MRI and speech data** for PD classification using **deep learning models**. Unlike unimodal approaches, multimodal feature fusion leverages complementary information, resulting in more reliable predictions.

The models were implemented in **PyTorch** and evaluated on publicly available datasets, demonstrating the promise of multimodal deep learning for PD detection.

------------------------------------------------------------------------

## Datasets

Two publicly available datasets were used:

-   **MRI Data**: OpenNeuro
    -   Format: `.nii` (NIfTI) volumetric brain scans\
    -   Samples: 108 healthy controls, 118 PD patients
-   **Speech Data**: Figshare
    -   Format: `.wav` audio recordings\
    -   Samples: 41 healthy controls, 40 PD patients

------------------------------------------------------------------------

## Methodology

### Preprocessing

-   MRI scans resized to **64×64×64 voxels**, skull stripping, normalization, and augmentation (random rotations).\
-   Audio recordings resampled to **22,050 Hz**, converted into **128×128 Mel-spectrograms**, with augmentations (pitch shifting, time stretching, noise injection).

### Model Design

-   **MRI Branch**: CNN and CNN-LSTM architectures for volumetric image feature extraction.\
-   **Audio Branch**: CNN and CNN-RNN architectures for spectrogram-based feature extraction.\
-   **Fusion Model**: Concatenates features from both branches at the **feature level** before classification.

### Training

-   Unimodal pretraining (MRI-only, Audio-only).\
-   Transfer of pretrained weights into the multimodal fusion classifier.\
-   Optimizer: **Adam**\
-   Loss function: **Cross-Entropy**

### Evaluation Metrics

-   Accuracy, Precision, Recall, F1-score, ROC AUC

------------------------------------------------------------------------

## Results

-   **MRI-only Model (CNN-LSTM):** 77.9% Accuracy, 0.805 AUC\
-   **Audio-only Model (CNN-RNN):** 96.2% Accuracy, 0.989 AUC\
-   **Fusion Model:**
    -   Accuracy: **95.2%**\
    -   Precision: **99.2%**\
    -   Recall: **95.2%**\
    -   F1-score: **97.1%**\
    -   AUC: **0.991**

Fusion outperformed unimodal models, especially in balancing precision and recall, proving the strength of multimodal learning.

------------------------------------------------------------------------

## Installation & Setup

### Prerequisites

-   Python 3.8+\
-   GPU recommended (CUDA-compatible)

### Dependencies

Install required packages:

``` bash
pip install -r requirements.txt
```

Typical libraries used: - `torch`, `torchvision`\
- `numpy`, `scipy`, `pandas`\
- `nibabel` (MRI handling)\
- `librosa` (audio processing)\
- `matplotlib`, `seaborn`

------------------------------------------------------------------------

## Usage

### Clone Repository

``` bash
git clone https://github.com/Promico-Git/PD-Fusion-Classification.git
cd PD-Fusion-Classification
```

### Run Preprocessing

-   Place MRI `.nii` files under `data/mri/{healthy,patient}/`\
-   Place audio `.wav` files under `data/audio/{healthy,patient}/`\
-   Adjust paths in `config.json` if needed.

### Train Models

``` bash
python train_mri.py
python train_audio.py
python train_fusion.py
```

### Evaluate Models

``` bash
python evaluate.py
```

------------------------------------------------------------------------

## Visualization

-   Training & validation loss curves are logged.\
-   Confusion matrices, ROC curves, and accuracy plots are generated for performance comparison.

------------------------------------------------------------------------

## Future Work

-   Explore **attention-based fusion** to improve feature integration.\
-   Test **transformers** for speech and multimodal data.\
-   Incorporate additional modalities (clinical/genetic data).\
-   Expand dataset for greater generalizability.

------------------------------------------------------------------------

## Citation

If you use this project in your research, please cite:

**Promise Ndiagwalu (2025).**\
*Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning.*\
Master’s Research Project, Toronto Metropolitan University.

------------------------------------------------------------------------

## Author

**Promise Ndiagwalu**\
MSc Data Science and Analytics\
Toronto Metropolitan University (2025)

Contact: promise.ndiagwalu\@torontomu.ca\
GitHub: [Promico-Git](https://github.com/Promico-Git)
