# Student Lifestyle & Stress Prediction

A production-quality machine learning project that predicts student stress levels from lifestyle, academic, social and financial indicators. The project includes synthetic data generation, exploratory analysis, training, explainability and an interactive Streamlit dashboard.

## Project Overview

This repository builds a realistic synthetic dataset for 10,000 students and trains multiple classification models to predict whether a student is likely to experience low, moderate or high stress. The data generation process uses student personas and a hidden weighted stress score so the task is realistic while avoiding target leakage.

## Features

- Synthetic dataset with 10,000 students and 50+ meaningful features
- Persona-driven generation for Topper, Average, Athlete, Struggling and Working Student profiles
- No target leakage in the generated data
- Modular preprocessing pipeline with encoding, scaling and train/test split
- Model comparison across Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVM, KNN, Naive Bayes and optional boosting libraries
- Explainability with SHAP and model feature importance
- Interactive Streamlit dashboard for real-time stress predictions

## Folder Structure

```text
Student-Stress-Prediction/
├── data/
│   ├── student_dataset.csv
│   └── data_dictionary.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Training.ipynb
├── src/
│   ├── generate_dataset.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── models/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Dataset Description

The generated dataset contains both numeric and categorical features such as sleep, study hours, exercise, social activity, commute time, caffeine intake, support scores, enrollment characteristics and lifestyle habits. The target label is Stress_Level with Low, Moderate and High classes.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

1. Generate the dataset
   ```bash
   python src/generate_dataset.py
   ```

2. Train the models
   ```bash
   python src/train_model.py
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

## Screenshots

- Placeholder: EDA notebook summary
- Placeholder: Streamlit dashboard preview
- Placeholder: model comparison dashboard

## Future Scope

- Add more realistic behavior patterns and seasonal effects
- Deploy the app to cloud services
- Integrate real student data with privacy safeguards
- Add fairness and bias monitoring for educational contexts

## License

MIT License
