# Autism Spectrum Disorder (ASD) Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![pandas](https://img.shields.io/badge/pandas-latest-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-latest-blue.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-red.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-latest-purple.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-yellow.svg)

## 📋 Overview

This project uses machine learning techniques to predict Autism Spectrum Disorder (ASD) based on behavioral traits and other relevant features. By analyzing patterns in patient data, the model helps identify potential ASD cases with high accuracy, potentially assisting in earlier diagnosis and intervention.

## 🧩 What is Autism Spectrum Disorder?

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that affects communication, social interaction, and behavior in various ways. Early detection is crucial for effective intervention and support. This project aims to leverage machine learning to assist in the detection process by analyzing behavioral patterns and other relevant data points.

## 📊 Dataset

The dataset used in this project contains information about individuals with various behavioral and personal attributes, including:

- 10 behavioral features (A1-A10 Scores)
- Age, gender, and ethnicity
- Family history (jaundice, autism)
- Country of residence
- Screening app usage
- Screening score (result)
- Other descriptive features

The project uses the [Autism Screening Dataset](https://www.kaggle.com/datasets/shivamshinde123/autismprediction) from Kaggle.

## 🛠️ Tech Stack

- **Python**: Core programming language
- **pandas & NumPy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning models and evaluation metrics
- **Matplotlib & Seaborn**: Data visualization
- **XGBoost**: Gradient boosting implementation
- **imbalanced-learn (SMOTE)**: Handling class imbalance

## 📝 Project Structure

- `Autism Prediction.ipynb`: Jupyter notebook containing the complete analysis and model building process
- `train.csv`: Training dataset
- `best_model.pkl`: Saved best-performing model
- `encoders.pkl`: Saved label encoders for categorical variables

## 🔍 Features & Methodology

### Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling/normalization
- Class imbalance handling with SMOTE

### Feature Engineering
- Feature selection based on importance
- Correlation analysis
- Dimensionality reduction techniques

### Models Explored
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### Model Evaluation
- Cross-validation
- Hyperparameter tuning with RandomizedSearchCV
- Performance metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC curve and AUC

## 🧪 Results

The final model achieves high accuracy in detecting ASD based on behavioral patterns and other features. The project demonstrates the potential of machine learning in assisting healthcare professionals with early detection of developmental disorders.

## 💻 How to Use

1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open and run the Jupyter notebook: `jupyter notebook "Autism Prediction.ipynb"`
4. To use the pre-trained model:
   ```python
   import pickle
   # Load the model
   with open('best_model.pkl', 'rb') as file:
       model = pickle.load(file)
   
   # Load encoders
   with open('encoders.pkl', 'rb') as file:
       encoders = pickle.load(file)
   
   # Prepare your data (X_new) using the same preprocessing steps
   # Make predictions
   predictions = model.predict(X_new)
   ```

## ⚠️ Disclaimer

This tool is designed to assist healthcare professionals and is not intended as a replacement for professional medical diagnosis. Always consult healthcare professionals for proper diagnosis of ASD.

## 🔗 Credits

- Original dataset from [Kaggle](https://www.kaggle.com/datasets/shivamshinde123/autismprediction)
- Project inspired by [KNOWLEDGE DOCTOR](https://www.youtube.com/watch?v=HvMokkugzVM)

## 📝 License

This project is open source and available under the MIT License.
