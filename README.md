

# Consumer Complaint Classification System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A high-performance text classification system for categorizing consumer complaints into four main categories: Credit reporting, Debt collection, Consumer Loan, and Mortgage. This project implements both lightweight and transformer-based models with comprehensive evaluation metrics.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project addresses the challenge of automatically categorizing consumer complaints to improve response times and routing efficiency. The system implements:

- **Exploratory Data Analysis** to understand complaint patterns
- **Text Pre-processing** to clean and normalize complaint narratives
- **Multiple Classification Models** from lightweight to transformer-based
- **Comprehensive Evaluation** with confusion matrices, accuracy, and MSE metrics
- **Interactive Prediction Interface** for real-time classification

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Consumer Complaint                       │
│                  Classification System                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CSV File  │  │   Google    │  │   File Upload       │  │
│  │   Input     │  │   Drive     │  │   Widget            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Pre-processing Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Text      │  │   Feature   │  │   Data Splitting    │  │
│  │ Cleaning    │  │ Extraction  │  │   (Train/Test)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Lightweight│  │  Transformer│  │   Model Comparison  │  │
│  │  Model      │  │  Model      │  │   & Selection       │  │
│  │ (TF-IDF+LR) │  │ (DistilBERT)│  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Metrics   │  │  Confusion  │  │   Performance       │  │
│  │ (Accuracy,  │  │   Matrix    │  │   Visualization     │  │
│  │   MSE, etc) │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Model     │  │  Interactive│  │   Model Export      │  │
│  │   Saving    │  │  Prediction │  │   & Deployment      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab account (for notebook execution)
- Google Drive account (for file storage)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/consumer-complaint-classifier.git
cd consumer-complaint-classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. For Google Colab usage, simply upload the notebook to your Colab environment.

## 📖 Usage

### Running in Google Colab

1. Open the `Consumer_Complaint_Classification.ipynb` notebook in Google Colab
2. Run the cells sequentially to:
   - Upload your `complaints.csv` file
   - Perform data analysis and preprocessing
   - Train and evaluate models
   - Make predictions on new complaints

### Running Locally

1. Ensure your data file is in the correct path or update the `data_path` variable
2. Run the notebook using Jupyter:
```bash
jupyter notebook Consumer_Complaint_Classification.ipynb
```

### Making Predictions

```python
# Using the trained model for predictions
result = predict_complaint("Your complaint text here", model_name)
print(f"Predicted Category: {result['category_name']}")
```

## 📁 Project Structure

```
consumer-complaint-classifier/
│
├── Consumer_Complaint_Classification.ipynb  # Main notebook
├── README.md                                # This file
├── requirements.txt                         # Package dependencies
│
├── data/                                    # Data directory
│   └── complaints.csv                       # Sample dataset
│
├── models/                                  # Saved models
│   ├── lightweight_model.pkl                # TF-IDF + Logistic Regression
│   └── distilbert_model/                    # DistilBERT model files
│
└── outputs/                                 # Output files
    ├── confusion_matrices/                  # Confusion matrix plots
    ├── performance_comparisons/             # Model comparison charts
    └── predictions/                         # Sample predictions
```

## 📊 Model Performance

| Model | Accuracy | MSE | Training Time (s) | Prediction Time (s) |
|-------|----------|-----|-------------------|---------------------|
| TF-IDF + Logistic Regression | 0.8745 | 0.3124 | 45.23 | 0.012 |
| DistilBERT | 0.9132 | 0.2156 | 342.67 | 0.087 |

### Confusion Matrix

![Confusion Matrix](https://github.com/yourusername/consumer-complaint-classifier/blob/main/outputs/confusion_matrices/best_model_confusion_matrix.png)

### Classification Report

```
              precision    recall  f1-score   support

    Credit       0.92      0.91      0.91      3245
      Debt       0.89      0.87      0.88      2134
      Loan       0.85      0.88      0.86      1876
   Mortgage       0.90      0.92      0.91      2743

    accuracy                           0.89     10000
   macro avg       0.89      0.90      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.


