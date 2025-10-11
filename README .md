# Breast Cancer Diagnosis Prediction
**By Rolddy Surpris and Naël Yssa Iben Ahmed Robert**

## Overview
This project aims to build a machine learning model capable of distinguishing malignant from benign breast tumors using diagnostic features derived from digitized cell nuclei images.

Inspired by Pink October, the international month for breast cancer awareness, this project combines data science and healthcare research to explore how AI can support early detection and improve diagnostic accuracy.

## Business Understanding
Breast cancer remains one of the leading causes of death among women worldwide. Early detection is key to improving survival rates. This project leverages data-driven modeling to assist pathologists and healthcare providers by offering a decision-support system that can complement human diagnosis.

**Target audience:**  
- Medical data analysts  
- Diagnostic technology researchers  
- Healthcare policymakers  

If deployed, this predictive tool could contribute to faster, more reliable, and cost-effective diagnostic workflows.

## Data Understanding
The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset, available through:  
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)  
- Scikit-learn’s built-in dataset loader

**Key facts:**  
- 569 samples, 30 numeric features  
- Target variable: Diagnosis (Malignant = M, Benign = B)  
- Features include: mean radius, texture, smoothness, compactness, symmetry, and more

## Data Preparation
Data preprocessing steps include:  
1. Feature scaling using StandardScaler  
2. Train-test split to ensure robust validation  
3. Label encoding (M = 1, B = 0)  
4. Building pipelines to streamline preprocessing and model training

Visualizations such as heatmaps and feature importance plots are used to explore relationships between variables and diagnosis.

## Modeling
This is a supervised binary classification problem.  
Initial and advanced models include:  
- Logistic Regression (baseline model)  
- Random Forest Classifier  
- XGBoost Classifier  
- K-Nearest Neighbors (KNN)

Model tuning:  
- Performed using GridSearchCV for optimal hyperparameters  
- Evaluation based on train/test splits and cross-validation

## Evaluation
Models will be evaluated using the following metrics:  
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC score and ROC curve

Visualizations will include confusion matrices and ROC curves to compare model performance.

## Deployment
Results will be presented through:  
- A Jupyter Notebook documenting the full workflow  
- A GitHub repository containing code, data references, and documentation  
- Optionally, a Streamlit web app for interactive prediction, allowing users to input tumor measurements and receive a diagnostic prediction (Benign or Malignant)

## Tools and Libraries
**Languages:** Python  
**Environment:** Jupyter Notebook

**Main libraries:**  
- pandas, numpy for data handling  
- matplotlib, seaborn for visualization  
- scikit-learn for modeling, pipelines, and tuning  
- xgboost for gradient boosting

## Project Structure
```
breast-cancer-diagnosis/
│
├── data/
│   └── breast_cancer_data.csv
│
├── notebooks/
│   └── breast_cancer_diagnosis.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── README.md
└── requirements.txt
```

## Summary
This project unites social impact and technical innovation by applying advanced machine learning to a vital healthcare issue.  
The objective is not only to reach high predictive accuracy but also to ensure transparency and interpretability, key factors for building trust in AI-assisted medical diagnosis.
