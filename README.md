![Header Image](header.png)
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
Data preprocessing steps included:  
1. Dropping irrelevant columns (`id`, `Unnamed: 32`)  
2. Encoding the target variable (`Malignant = 1`, `Benign = 0`)  
3. Handling missing values and cleaning feature names  
4. Standardizing all numerical features using `StandardScaler`  
5. Saving the cleaned data as `clean_data.csv`  

Feature exploration involved heatmaps, correlation analysis, and summary statistics to understand feature distributions and relationships.

## Modeling
This is a **supervised binary classification** problem.  
Several models were tested and compared to identify the most interpretable and effective solution:

- Logistic Regression (baseline and final model)  
- Random Forest Classifier  
- Multilayer Perceptron (MLP)

After model comparison, **Logistic Regression** was selected as the final model for its balance between high recall, accuracy, and clinical interpretability.

## Model Tuning
Hyperparameter tuning was performed using **GridSearchCV** to maximize recall for malignant cases.  
The best configuration achieved:
- **C:** 10  
- **Penalty:** L2  
- **Solver:** saga  
- **Class weight:** balanced  

This configuration provided a recall score of **0.965**, ensuring that most malignant tumors were correctly identified.

## Evaluation
The tuned model was tested on unseen data, achieving:  
- **Accuracy:** 97.4%  
- **Recall (Malignant):** 0.95  
- **Precision (Malignant):** 0.98  

The confusion matrix showed only two malignant cases misclassified as benign and one benign case as malignant.  
These results highlight strong reliability while acknowledging the natural unpredictability of biological data.

**Evaluation metrics used:**  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

**Visualizations included:**  
- Confusion matrix  
- ROC curve  
- Precision-Recall vs. Threshold curve  

## Deployment
The trained logistic regression model was exported using **Joblib** and deployed via a **Streamlit web app**.  
The app allows users to input tumor characteristics and instantly receive a diagnostic prediction (Benign or Malignant) along with a probability score.

**Deployment workflow:**  
1. The best model was saved as `best_logistic_model.pkl`.  
2. A `Deployment.py` script was created to load the model and build an interactive web interface.  
3. The app can be run locally using Streamlit.

**To run the app locally (on your computer):**  
```bash
pip install streamlit joblib pandas scikit-learn
streamlit run Deployment.py
```
**To use the app on your browser**
Link : https://predicting-breast-cancer-diagnosis.streamlit.app/

The application demonstrates how data science solutions can transition from research to real-world clinical decision support.

## Tools and Libraries

Language: Python
Environment: Jupyter Notebook

## Main libraries:

pandas, numpy — Data manipulation

matplotlib, seaborn — Data visualization

scikit-learn — Modeling, tuning, and evaluation

joblib — Model persistence

streamlit — App deployment

## Project Structure

```
breast-cancer-diagnosis/

│
├── data.csv
├── clean_data.csv
│
├── EDA.ipynb
├── data_cleaning.ipynb
├── modeling.ipynb
├── optimisation.ipynb
│
├── deployment.py
├── best_logistic_model.pkl
├── requirements.txt
│
├── Rapport final.pdf
├── Non-technical Presentation.pdf
├── Notebooks.pdf
│
├── LICENSE
├── .gitignore
├── README.md
```
