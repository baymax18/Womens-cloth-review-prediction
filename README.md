# Womens-cloth-review-prediction

# 🧠 Women's Clothing Review Rating Prediction (NLP + ML)

This project builds a machine learning model that predicts the **rating (1 to 5)** for women's clothing products based on customer **text reviews**. It uses the **Women’s E-Commerce Clothing Reviews** dataset, which contains over 23,000 anonymized commercial reviews.

---

## 📁 Dataset Overview

- **Total Rows**: 23,486  
- **Total Features**: 10  
- **Target Variable**: `Rating` (1 to 5)
- **Primary Input**: `Review Text`

The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).

---

## 🧰 Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy**
- **scikit-learn**: Logistic Regression, TF-IDF Vectorization
- **Matplotlib**, **Seaborn**: Visualization
- **Jupyter Notebook** for development

---

## 📊 Problem Statement

> Given a customer’s written review of a clothing product, predict the numeric **rating (1-5)** they would give.

This is a **multi-class text classification** problem using supervised learning.

---

## 🛠️ Project Pipeline

1. **Data Preprocessing**  
   - Remove nulls  
   - Clean text (lowercasing, removing punctuation, etc.)

2. **Text Vectorization**  
   - Convert text reviews to TF-IDF feature vectors.

3. **Model Training**  
   - Logistic Regression for multi-class classification.

4. **Evaluation**  
   - Accuracy  
   - Precision, Recall, F1-Score  
   - Confusion Matrix
