# 🚀 Loan Default Prediction Dashboard

https://github.com/user-attachments/assets/dd97498e-904f-4107-b64e-e986aa147b17

This project provides a **Loan Default Prediction Dashboard** using **Streamlit**. It predicts whether a loan applicant will default based on input features such as the number of employees, loan amount, and other financial details. The dashboard includes visualizations, allows users to enter their data for prediction, and explains the importance of key features using model explainability techniques.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-Framework-orange)](https://streamlit.io/)

---

## 🌟 Features

- 🖥️ **Interactive Dashboard**: Built with Streamlit for easy user interaction.
- 🔍 **Loan Default Prediction**: Uses a machine learning model to predict loan default.
- 📊 **Data Visualization**: Includes histograms, bar charts, correlation matrices, and trend lines.
- 🔑 **Feature Importance**: Displays the significance of different features in predictions.
- ❄️ **Snowfall Animation**: Aesthetic snowfall effect for a visual enhancement.

---

## 📁 Project Structure

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

![df](https://github.com/user-attachments/assets/cb79fbe7-f928-44af-8443-8902dea92518)


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

## Features

- **User-friendly Dashboard**: Built with Streamlit for easy user interaction and data entry.
- **Loan Default Prediction**: Uses a machine learning model (RandomForestClassifier) to predict loan default based on key features.
- **Data Visualization**: Includes visualizations such as histograms, bar charts, correlation matrices, and trend lines.
- **Feature Importance**: Displays feature importance to explain which features impact the predictions the most.
- **Snowfall Animation**: Added visual enhancement for a winter theme with falling snowflakes.

## Visualizations

The dashboard includes the following visualizations:
- **Histogram**: Shows the distribution of numerical features.
- **Bar Chart**: Displays counts of categories.
- **Correlation Matrix**: Helps identify relationships between numerical features.
- **Trend Line**: Shows trends between two numerical features.

## Setup

### Requirements

- Python 3.7 or higher
- Packages listed in `requirements.txt`

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install Dependencies:**:
   ```bash
   pip install -r requirements.txt
4 **Place Dataset**:
   ```bash
   pip install -r requirements.txt

### Running the Project
``` bash

streamlit run app.py

Model Details

The model used is a RandomForestClassifier from scikit-learn, trained on key features from the dataset. The model is trained using the following pipeline:

    Data Preprocessing: Missing values are handled, categorical features are encoded using OrdinalEncoder.
    Model Training: The RandomForest model is trained using class balancing (class_weight='balanced') to handle imbalanced data.
    Saving Model: The model is saved to models/loan_model.pkl.

Retraining the Model

If the model is missing or needs to be updated, it can be retrained directly through the app:

    When the model file is not found, the app retrains the model from scratch and saves it as loan_model.pkl.

Explanation of Features

Below are key features in the dataset used for prediction:

    NoEmp: Number of employees associated with the loan applicant.
    CreateJob: Number of jobs created by the loan applicant.
    RetainedJob: Number of jobs retained by the loan applicant.
    GrAppv: Gross approval amount of the loan.
    SBA_Appv: SBA-approved portion of the loan.
    DisbursementGross: Gross disbursement of the loan.
    RevLineCr: Whether there is a revolving line of credit (Y for Yes, N for No).
    default: Target column indicating whether the applicant defaulted on the loan.

Contributing

If you’d like to contribute to this project:

    Fork the repository.
    Create a new branch for your feature (git checkout -b feature/new-feature).
    Commit your changes (git commit -am 'Add new feature').
    Push to the branch (git push origin feature/new-feature).
    Create a new Pull Request.

