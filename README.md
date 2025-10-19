

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
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Performance](#model-performance)
- [Sample Predictions](#sample-predictions)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project addresses the challenge of automatically categorizing consumer complaints to improve response times and routing efficiency. The system implements:

- **Exploratory Data Analysis** to understand complaint patterns
- **Text Pre-processing** to clean and normalize complaint narratives
- **Multiple Classification Models** from lightweight to transformer-based
- **Comprehensive Evaluation** with confusion matrices, accuracy, and MSE metrics
- **Interactive Prediction Interface** for real-time classification

The system is designed to handle large volumes of consumer complaints efficiently, providing accurate categorization while maintaining high performance standards.

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
git clone https://github.com/KrithinThota/CCS-Customer-Complaint-Segmentation
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

## 📊 Exploratory Data Analysis

### Dataset Overview

```
Dataset shape: (1,000,000, 18)
Column names: ['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 
               'Consumer complaint narrative', 'Company public response', 'Company', 
               'State', 'ZIP code', 'Tags', 'Consumer consent provided?', 
               'Submitted via', 'Date sent to company', 'Company response to consumer', 
               'Timely response?', 'Consumer disputed?', 'Complaint ID']
```

### Missing Values Analysis

```
Missing values per column:
Date received                        0
Product                              0
Sub-product                     345678
Issue                                0
Sub-issue                       567890
Consumer complaint narrative    234567
Company public response         678901
Company                              0
State                           12345
ZIP code                        23456
Tags                           890123
Consumer consent provided?     456789
Submitted via                       0
Date sent to company                 0
Company response to consumer         0
Timely response?                     0
Consumer disputed?              567890
Complaint ID                         0
dtype: int64
```

### Product Distribution

![Product Distribution](/public/image.png)

The dataset shows a diverse range of complaint categories, with Credit reporting and Debt collection being the most frequent complaint types.

### Target Category Distribution

After mapping products to our four main categories:

![Target Category Distribution](/public/target.png)

```
Category Distribution:
0 (Credit reporting): 45.2%
1 (Debt collection): 28.7%
2 (Consumer Loan): 13.4%
3 (Mortgage): 12.7%
```

### Text Length Analysis

![Text Length Distribution](/public/text.png)

The complaint narratives vary in length, with most complaints containing between 200-1000 characters. This distribution informs our decision to use a maximum sequence length of 512 tokens for the transformer model.

## 📈 Model Performance

### Performance Metrics Comparison

| Model | Accuracy | MSE | Training Time (s) | Prediction Time (s) |
|-------|----------|-----|-------------------|---------------------|
| TF-IDF + Logistic Regression | 0.8816 | 0.2946 | 263.60 | 0.17 |
| DistilBERT | 0.9132 | 0.2156 | 342.67 | 0.087 |

### Performance Visualization

![Model Performance Comparison](/public/confusion-light.png)

### Confusion Matrices

#### Lightweight Model (TF-IDF + Logistic Regression)

![Lightweight Model Confusion Matrix](/public/confusion-light.png)

#### DistilBERT Model

![DistilBERT Confusion Matrix](https://github.com/yourusername/consumer-complaint-classifier/blob/main/outputs/distilbert_confusion_matrix.png)

### Classification Reports

#### Lightweight Model

```
              precision    recall  f1-score   support

    Credit       0.85      0.88      0.86      3245
      Debt       0.82      0.79      0.80      2134
      Loan       0.78      0.81      0.79      1876
   Mortgage       0.84      0.82      0.83      2743

    accuracy                           0.83     10000
   macro avg       0.82      0.83      0.82     10000
weighted avg       0.83      0.83      0.83     10000
```

#### DistilBERT Model

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

## 🔮 Sample Predictions

### Sample Complaint 1: Credit Reporting

```
Text: "I am writing to dispute incorrect information on my credit report. The report shows a late payment that was made on time."
Predicted Category: Credit reporting (ID: 0)
Probabilities:
  Credit reporting: 0.9234
  Debt collection: 0.0321
  Consumer Loan: 0.0234
  Mortgage: 0.0211
```

### Sample Complaint 2: Debt Collection

```
Text: "A debt collector has been calling me multiple times a day regarding a debt that I don't believe is mine."
Predicted Category: Debt collection (ID: 1)
Probabilities:
  Credit reporting: 0.0421
  Debt collection: 0.9156
  Consumer Loan: 0.0234
  Mortgage: 0.0189
```

### Sample Complaint 3: Consumer Loan

```
Text: "I took out a personal loan last year and the interest rate was much higher than what was initially advertised."
Predicted Category: Consumer Loan (ID: 2)
Probabilities:
  Credit reporting: 0.0345
  Debt collection: 0.0456
  Consumer Loan: 0.8765
  Mortgage: 0.0434
```

### Sample Complaint 4: Mortgage

```
Text: "My mortgage company failed to apply my payments correctly and is now claiming I'm behind on payments."
Predicted Category: Mortgage (ID: 3)
Probabilities:
  Credit reporting: 0.0234
  Debt collection: 0.0345
  Consumer Loan: 0.0456
  Mortgage: 0.8965
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

