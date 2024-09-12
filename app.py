import os
import streamlit as st
import pandas as pd
import numpy as np
from src.utils import validate_inputs, load_model
from src.model_training import ModelTrainer
from src.prediction import Predictor
import category_encoders as ce  # Category encoders for categorical data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.graphs import plot_histograms, plot_bar_charts, plot_correlation_matrix, plot_trend_lines
import joblib  # For saving and loading the model

# Display the logo
st.image('images/df.jpg', use_column_width=True)

# Define model path
model_path = 'models/loan_model.pkl'

# Snowfall effect using CSS
snowfall_css = """
<style>
/* Snowflake style */
.snowflake {
  color: white;
  font-size: 1.5em;
  position: absolute;
  top: -50px;
  z-index: 9999;
  user-select: none;
  pointer-events: none;
  animation: fall linear infinite;
}

/* Snowfall animation */
@keyframes fall {
  to {
    transform: translateY(100vh);
  }
}
.snowflake:nth-of-type(1) {left: 10%; animation-duration: 10s; animation-delay: 0s;}
.snowflake:nth-of-type(2) {left: 20%; animation-duration: 12s; animation-delay: 2s;}
.snowflake:nth-of-type(3) {left: 30%; animation-duration: 8s; animation-delay: 4s;}
.snowflake:nth-of-type(4) {left: 40%; animation-duration: 14s; animation-delay: 6s;}
.snowflake:nth-of-type(5) {left: 50%; animation-duration: 10s; animation-delay: 8s;}
.snowflake:nth-of-type(6) {left: 60%; animation-duration: 12s; animation-delay: 10s;}
.snowflake:nth-of-type(7) {left: 70%; animation-duration: 8s; animation-delay: 12s;}
.snowflake:nth-of-type(8) {left: 80%; animation-duration: 14s; animation-delay: 14s;}
.snowflake:nth-of-type(9) {left: 90%; animation-duration: 10s; animation-delay: 16s;}
</style>

<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
<div class="snowflake">❄️</div>
"""

# Inject the CSS into the Streamlit app
st.markdown(snowfall_css, unsafe_allow_html=True)

# Load description from an external file
def load_description(file_path):
    with open(file_path, 'r') as file:
        return file.read()
# Load description from an external file (e.g., 'data/description.txt')
description_text = load_description('data/description.txt')  # Adjust the path based on your directory structure

# Description Box for Dataset Information
with st.expander("📊 About the Dataset and Features", expanded=True):
    st.markdown(description_text)

# Load dataset
@st._cache_data
def load_data():
    return pd.read_csv('data/load_data.csv')

# Display logo and title
st.title('Loan Default Prediction Dashboard')

# Sidebar for visualizations
st.sidebar.header('Visualization Options')

# Load data
df = load_data()

# Select columns for visualization (select 4 numerical features)
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
selected_columns = st.sidebar.multiselect('Select 4 numerical features to visualize', numerical_columns, default=numerical_columns[:4])

if selected_columns:
    # Select the type of visualization
    plot_type = st.sidebar.selectbox('Select Visualization Type', ['Histogram', 'Bar Chart', 'Correlation Matrix', 'Trend Line', 'SHAP Summary'])

    # Plot the selected graph
    if plot_type == 'Histogram':
        plot_histograms(df, selected_columns)
    elif plot_type == 'Bar Chart':
        plot_bar_charts(df, selected_columns)
    elif plot_type == 'Correlation Matrix':
        plot_correlation_matrix(df, selected_columns)
    elif plot_type == 'Trend Line':
        # For trend lines, we select two numerical columns
        x_col = st.sidebar.selectbox('Select X axis', selected_columns)
        y_col = st.sidebar.selectbox('Select Y axis', [col for col in selected_columns if col != x_col])
        plot_trend_lines(df, x_col, y_col)


# Function to retrain and save the model
def train_and_save_model(X_train, y_train, model_path):
    # Use a balanced class weight for training to handle class imbalance
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_path)
    st.success(f"Model trained and saved to {model_path}.")

    return model


# Load or retrain the model
def load_or_train_model(model_path):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("Model loaded successfully.")

        # Retrieve the training features from the model
        try:
            training_features = model.feature_names_in_  # Check for feature names in the model
        except AttributeError:
            st.warning("The model does not contain 'feature_names_in_' attribute. Proceeding with default columns.")
            training_features = None
    else:
        st.warning("Model not found. Retraining the model from scratch.")
        model = None
        training_features = None

    return model, training_features


# Check if the model exists and load or train it
model, training_features = load_or_train_model(model_path)

# Sidebar for user input
st.sidebar.header('Enter Loan Applicant Information')

# User input for the original features (adjust these to match your dataset)
no_emp = st.sidebar.number_input('Number of Employees', min_value=1, max_value=1000, value=10,
                                 help='Enter the number of employees.')
create_job = st.sidebar.number_input('Number of Jobs Created', min_value=0, max_value=1000, value=5,
                                     help='Enter the number of jobs created.')
retained_job = st.sidebar.number_input('Number of Jobs Retained', min_value=0, max_value=1000, value=5,
                                       help='Enter the number of jobs retained.')
gr_appv = st.sidebar.number_input('Gross Approval Amount ($)', min_value=1000, max_value=1000000, value=50000,
                                  help='Enter the gross approval amount.')
sba_appv = st.sidebar.number_input('SBA Approval Amount ($)', min_value=1000, max_value=1000000, value=50000,
                                   help='Enter the SBA approval amount.')
disbursement_gross = st.sidebar.number_input('Disbursement Gross ($)', min_value=1000, max_value=1000000, value=50000,
                                             help='Enter the disbursement gross.')
rev_line_cr = st.sidebar.selectbox('Revolving Line of Credit (Y/N)', ['Y', 'N'],
                                   help='Select whether there is a revolving line of credit.')

# Collect user inputs into a DataFrame
user_input = pd.DataFrame({
    'NoEmp': [no_emp],
    'CreateJob': [create_job],
    'RetainedJob': [retained_job],
    'GrAppv': [gr_appv],
    'SBA_Appv': [sba_appv],
    'DisbursementGross': [disbursement_gross],
    'RevLineCr': [rev_line_cr]  # Categorical input
})

st.subheader('User Input:')
st.write(user_input)


# Dummy function to simulate loading and splitting the training data
# Replace this with actual logic for loading your dataset and splitting it
def load_training_data():
    # Sample dataset generation - replace with your actual dataset loading code
    data = pd.DataFrame({
        'NoEmp': np.random.randint(1, 500, 100),
        'CreateJob': np.random.randint(1, 500, 100),
        'RetainedJob': np.random.randint(1, 500, 100),
        'GrAppv': np.random.uniform(1000, 100000, 100),
        'SBA_Appv': np.random.uniform(1000, 100000, 100),
        'DisbursementGross': np.random.uniform(1000, 100000, 100),
        'RevLineCr': np.random.choice(['Y', 'N'], 100),  # Categorical column
        'default': np.random.choice([0, 1], 100)  # Target column
    })

    X = data.drop(columns=['default'])
    y = data['default']

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Ensure that training features are available
if model is None:
    X_train, X_test, y_train, y_test = load_training_data()

    # **Encode categorical features** before training the model
    encoder = ce.OrdinalEncoder(cols=['RevLineCr'])  # Assuming OrdinalEncoder for categorical features
    X_train_encoded = encoder.fit_transform(X_train)

    model = train_and_save_model(X_train_encoded, y_train, model_path)
    training_features = X_train_encoded.columns  # Update training features

# If the model is trained, proceed with predictions
if model is not None and training_features is not None:
    # Apply encoding for categorical features (like 'RevLineCr')
    encoder = ce.OrdinalEncoder(cols=['RevLineCr'])  # Assuming OrdinalEncoder for categorical features
    user_input_encoded = encoder.fit_transform(user_input)

    # Add missing columns to the user input with default values
    for feature in training_features:
        if feature not in user_input_encoded.columns:
            user_input_encoded[feature] = 0  # Fill with 0 for missing features

    # Reorder the columns to match the training features
    user_input_encoded = user_input_encoded[training_features]

    # Validate input data BEFORE modifying the DataFrame
    try:
        validate_inputs(user_input_encoded)  # Validate the original inputs
    except ValueError as e:
        st.error(f"Input validation error: {e}")

    # Threshold slider for default probability
    threshold = st.sidebar.slider('Default Risk Threshold', min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                  help="Adjust the decision threshold for predicting defaults.")

    # Check if the model exists and make prediction
    if st.button('Predict Default Risk'):
        predictor = Predictor(model)
        default_prob = model.predict_proba(user_input_encoded)[0][1]  # Probability of default

        st.write(f"Probability of default: {default_prob:.2f}")

        if default_prob > threshold:
            st.error(f'Prediction: The applicant is likely to default on the loan. (Probability: {default_prob:.2f})')
        else:
            st.success(
                f'Prediction: The applicant is unlikely to default on the loan. (Probability: {default_prob:.2f})')

    # Display feature importance to understand model behavior
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        'Feature': training_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.subheader("Feature Importances")
    st.write(feature_importances_df)

else:
    st.error("Cannot proceed without training features.")