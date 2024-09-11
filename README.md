# Loan Default Prediction Dashboard

This project is a web-based loan default prediction tool built using Streamlit. The tool predicts whether a loan applicant will default or not based on user input features. It uses a pre-trained model (RandomForestClassifier) and allows interactive predictions via the web interface.

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