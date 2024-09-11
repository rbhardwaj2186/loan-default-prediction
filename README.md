# Loan Default Prediction Dashboard

This project is a web-based loan default prediction tool built using Streamlit. The tool predicts whether a loan applicant will default or not based on user input features. It uses a pre-trained model (RandomForestClassifier) and allows interactive predictions via the web interface.

![Loan Default Prediction Logo](D:/Work/Gre/UTD/Courses/Winter/Projects/Data Analytic Projects/Credit_Default/df.jpg)

## Project Structure

```bash
loan-default-prediction/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # Project overview
├── data/                 # Directory for dataset(s)
│   └── loan_data.csv     # Dataset file
├── models/               # Pre-trained models
│   └── loan_model.pkl    # Pre-trained loan prediction model
├── src/                  # Source folder for your project
│   ├── __init__.py       # Module initialization
│   ├── data_preprocessing.py  # Data preprocessing script
│   ├── model_training.py      # Model training script
│   ├── prediction.py          # Prediction script
│   └── utils.py               # Utility functions
└── .gitignore            # Files to ignore in version control

# Loan Default Prediction Dashboard

![Loan Default Prediction Logo](D:/Work/Gre/UTD/Courses/Winter/Projects/Data Analytic Projects/Credit_Default/df.jpg)

This project provides a **Loan Default Prediction Dashboard** using **Streamlit**. It predicts whether a loan applicant will default based on their input features such as the number of employees, loan amount, and other financial details. The dashboard includes visualizations of key features, allows users to enter their data for prediction, and incorporates explainability of feature importance in model predictions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Visualizations](#visualizations)
- [Setup](#setup)
- [How to Run the Project](#how-to-run-the-project)
- [Model Details](#model-details)
- [Explanation of Features](#explanation-of-features)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Loan Default Prediction Dashboard** is an interactive web app that predicts whether a loan applicant will default or not. The prediction is based on several features, including the number of employees, the gross approval amount, and the presence of a revolving line of credit. The dashboard allows users to visualize important data insights and predict default risk.

## Project Structure

The project follows a structured, object-oriented approach, and is built using **Streamlit** for the frontend and **scikit-learn** for machine learning.
