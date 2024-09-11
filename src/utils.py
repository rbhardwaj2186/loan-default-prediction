import pandas as pd
import joblib


def validate_inputs(df: pd.DataFrame) -> bool:
    """
    Validates user input data for any missing or invalid values.

    Args:
    - df: A pandas DataFrame containing the input features.

    Returns:
    - bool: True if all inputs are valid, otherwise raises a ValueError.
    """
    if df.isnull().values.any():
        raise ValueError("Input data contains missing values.")

    # Validate specific columns based on the new input structure
    if not 1 <= df['NoEmp'].values[0] <= 1000:
        raise ValueError("Number of Employees must be between 1 and 1000.")
    if not 0 <= df['CreateJob'].values[0] <= 1000:
        raise ValueError("Number of Jobs Created must be between 0 and 1000.")
    if not 0 <= df['RetainedJob'].values[0] <= 1000:
        raise ValueError("Number of Jobs Retained must be between 0 and 1000.")
    if not 1000 <= df['GrAppv'].values[0] <= 1000000:
        raise ValueError("Gross Approval Amount must be between $1,000 and $1,000,000.")
    if not 1000 <= df['SBA_Appv'].values[0] <= 1000000:
        raise ValueError("SBA Approval Amount must be between $1,000 and $1,000,000.")
    if not 1000 <= df['DisbursementGross'].values[0] <= 1000000:
        raise ValueError("Disbursement Gross must be between $1,000 and $1,000,000.")

    return True


def load_model(model_path: str):
    """
    Loads a pre-trained model from the given file path.

    Args:
    - model_path: The path to the pre-trained model file.

    Returns:
    - model: The loaded machine learning model.
    """
    return joblib.load(model_path)