# Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning

## Introduction

This project investigates the effectiveness of a **feature-level fusion** of **speech** and **neuroimaging (MRI) biomarkers** for the classification of Parkinson's Disease (PD) using deep learning. The primary goal is to develop a more accurate and reliable AI-based diagnostic tool for the early detection of PD, addressing the limitations of unimodal data and subjective clinical assessments.

## Key Features

* **Multimodal Data Fusion**: Integrates audio (speech) and 3D MRI (neuroimaging) data for a more comprehensive analysis.
* **Deep Learning Models**: Utilize various neural network architectures:
    * **3D Convolutional Neural Networks (3D-CNN)** for MRI feature extraction.
    * **2D Convolutional Neural Networks (2D-CNN)** for audio spectrogram analysis.
    * **Recurrent Neural Networks (RNN/LSTM)** for sequential data processing.
* **Feature-Level Fusion**: Combines features extracted from both modalities before the final classification, allowing the model to learn complex inter-modal relationships.
* **End-to-End Pipeline**: Includes data loading, preprocessing, augmentation, model training, and evaluation.

## Technologies Used

* **Python 3.x**
* **PyTorch**: For building and training deep learning models.
* **librosa**: For audio processing and feature extraction.
* **nibabel**: For reading and processing MRI data in NIfTI format.
* **scikit-learn**: For evaluation metrics and data splitting.
* **NumPy**: For numerical operations.
* **Matplotlib**: For plotting and visualization.
* **OpenCV (cv2)**: For image processing tasks.

## Dataset

This project utilizes two publicly available datasets:

1.  **MRI Dataset**: Sourced from **OpenNeuro**, containing structural MRI scans for healthy controls and PD patients in NIfTI (`.nii`) format.
2.  **Audio Dataset**: Sourced from **Figshare**, consisting of voice recordings from healthy individuals and PD patients in WAV (`.wav`) format.

To handle class and modality imbalances, the datasets are resampled to ensure a balanced input for model training.

## Methodology

The project follows a structured methodology:

1.  **Data Preprocessing**:
    * MRI scans are normalized, resized to `64x64x64`, and converted to a single channel.
    * Audio files are resampled, and their Mel spectrograms are generated and resized to `128x128`.
    * Data augmentation techniques (e.g., random rotation for MRI, time stretching/pitch shifting for audio) are applied to the training set to improve model generalization.
2.  **Unimodal Model Pre-training**:
    * An **MRI 3D-CNN** model is trained on the MRI dataset.
    * An **Audio CNN** model is trained on the audio spectrograms.
3.  **Feature-Level Fusion**:
    * The pre-trained unimodal models are used as feature extractors.
    * The extracted feature vectors from both modalities are concatenated.
    * A final classifier is trained on the fused feature vector to make the final prediction.
4.  **Evaluation**:
    * The performance of both the unimodal and the final fusion model is evaluated using metrics such as **Accuracy, Precision, Recall, F1-score, and AUC-ROC**.

## Getting Started

### Prerequisites

* Python 3.8 or higher
* `pip` for package management

### Installation

1.  **Clone the repository**:
    ```
    git clone [https://github.com/Promico-Git/PD-Fusion-Classification](https://github.com/Promico-Git/PD-Fusion-Classification)
    cd PD-Fusion-Classification
    ```
2.  **Create a virtual environment (recommended)**:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages**:
    ```
    pip install torch torchvision torchaudio
    pip install librosa nibabel scikit-learn numpy matplotlib opencv-python
    ```

### Usage

1.  **Organize your dataset** into a `Datasets` directory with the following structure:
    ```
    Datasets/
    ├── Audio/
    │   ├── healthy/
    │   │   └── ... (audio files)
    │   └── patient/
    │       └── ... (audio files)
    └── MRI/
        ├── healthy/
        │   └── ... (MRI files)
        └── patient/
            └── ... (MRI files)
    ```
2.  **Run the scripts**:
    * To train and evaluate the unimodal models:
        ```
        python "Audio Classification Models.py"
        python "MRI Classification Models.py"
        ```
    * To train and evaluate the feature-fusion model:
        ```
        python "Project Code_Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning.py"
        ```

## File Descriptions

* `Audio Classification Model.py`: Contains the code for training and evaluating the unimodal **CNN and RNN models** on the audio dataset.
* `MRI Classification Model.py`: Contains the code for training and evaluating the unimodal **3D-CNN and CNN-LSTM models** on the MRI dataset.
* `Project Code_Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning.py`: The main script implements the **feature-level fusion** by pre-training the individual models and then training a final classifier on the combined features.

## Results

The fusion model demonstrated superior performance compared to the individual unimodal models, highlighting the benefit of integrating complementary information from both speech and neuroimaging data.

| Model          | Accuracy | Precision | Recall | F1 Score | AUC   |
| -------------- | -------- | --------- | ------ | -------- | ----- |
| **MRI (CNN)** | 0.709    | 0.732     | 0.682  | 0.706    | 0.790 |
| **Audio (CNN)**| 0.952    | 1.000     | 0.944  | 0.971    | 0.968 |
| **Fusion (CNN)**| **0.952**| **0.992** | **0.952**| **0.971**| **0.991**|

The fusion model achieves an excellent balance of precision and recall, with a near-perfect AUC, indicating its strong ability to distinguish between classes.

## Future Work

* Explore alternative fusion strategies, such as attention-based mechanisms.
* Investigate other deep learning architectures like Transformers.
* Incorporate additional data modalities (e.g., clinical, genetic data).
* Validate the model on larger, more diverse datasets.

## Citation

If you use this project in your research, please cite:

**Promise Ndiagwalu (2025).**
*Feature-Level Fusion for Parkinson's Disease Classification Using Deep Learning.*
Master’s Research Project, Toronto Metropolitan University.

## Author

**Promise Ndiagwalu**
MSc Data Science and Analytics
Toronto Metropolitan University (2025)

Contact: promise.ndiagwalu@torontomu.ca
GitHub: [Promico-Git](https://github.com/Promico-Git)
