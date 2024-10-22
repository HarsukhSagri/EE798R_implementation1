# EE798R - Speech Emotion Recognition with Co-Attention Based Multi-level Acoustic Information

This repository contains the implementation of Speech Emotion Recognition (SER) using multi-level acoustic features, including MFCC, spectrogram, and wav2vec2 (W2E) embeddings, with a co-attention mechanism. The project is part of the EE798R coursework, building on the research presented in the paper [Speech Emotion Recognition with Co-Attention Based Multi-Level Acoustic Information](https://arxiv.org/abs/2203.15326).

## 1. File system
- `models/`
  - Contains various model-related files, including the co-attention module.
- `results/`
  - Includes visualizations such as t-SNE plots.
- `extracted_features.pkl`
  - Pickled file containing pre-extracted features.
- `crossval_SER.py`
  - Script for performing cross-validation experiments.
- `train_ser.py`
  - Training script for SER.
- `data_utils.py`
  - Utility functions for data loading and processing.
- `main_code/`
  - Contains the .ipynb which needs to be run to train the model and then display the result for cross-validation strategies.
- `description_pdf.pdf`
  - Contains the description of how the implementation is done and is a comprehensive report.
- `requirements.txt`
  - Dependencies for running the project.

## 2. Environment
- **Python version**:  3.10
- **PyTorch**
- **CUDA**
- **cudnn**
- **GPU**

## 3. How to Use
1. Download the processed data (dataset) from:
   - [Google Drive](https://drive.google.com/drive/folders/1SsxMib2NoKVvoTspe-XDF-2NC7PNEpMw?usp=sharing)
2. Install dependencies by running:
   ```bash
   pip install -r requirements.txt
3. Run the files in [main_code](https://github.com/HarsukhSagri/EE798R_implementation1/tree/main/main_code) folder for training on 5-fold and 10-fold cross-validation strategies. Clone the github repo as given in [main_code](https://github.com/HarsukhSagri/EE798R_implementation1/tree/main/main_code) first and then make required changes in ser_model.py using this [file](https://github.com/HarsukhSagri/EE798R_implementation1/blob/main/models/ser_model.py) and train_ser.py using this [file](https://github.com/HarsukhSagri/EE798R_implementation1/blob/main/train_ser.py).
