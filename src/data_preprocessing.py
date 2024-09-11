import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.train_encoded = None
        self.test_encoded = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df.drop(columns="index", inplace=True)
        return self.df

    def preprocess(self):
        # Handle invalid values in 'RevLineCr', 'LowDoc', 'NewExist'
        self.df['RevLineCr'] = self.df['RevLineCr'].apply(lambda x: 'N' if x not in ['Y', 'N'] else x)
        self.df['LowDoc'] = self.df['LowDoc'].apply(lambda x: 'N' if x not in ['Y', 'N'] else x)
        self.df['NewExist'] = self.df['NewExist'].apply(lambda x: None if x not in [1, 2] else x)

        # Fill missing categorical values with mode
        category_cols = ['City', 'State', 'Bank', 'BankState', 'RevLineCr', 'LowDoc', 'NewExist']
        for column in category_cols:
            self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

        # Split into train and test sets
        X_train, X_test = train_test_split(self.df, test_size=0.3, random_state=123)

        # Apply target encoding to categorical columns
        categorical_columns = ['City', 'State', 'Bank', 'BankState', 'RevLineCr', 'LowDoc', 'NewExist', 'UrbanRural']
        encoder = ce.TargetEncoder(cols=categorical_columns)
        encoder.fit(X_train, X_train['MIS_Status'])
        self.train_encoded = encoder.transform(X_train)
        self.test_encoded = encoder.transform(X_test)

        # Standard scaling for numerical columns
        numerical_columns = ['NoEmp', 'CreateJob', 'RetainedJob', 'GrAppv', 'SBA_Appv', 'DisbursementGross',
                             'BalanceGross']
        scaler = StandardScaler()
        self.train_encoded[numerical_columns] = scaler.fit_transform(self.train_encoded[numerical_columns])
        self.test_encoded[numerical_columns] = scaler.transform(self.test_encoded[numerical_columns])

        # Feature engineering
        self.train_encoded['Log_DisbursementGross'] = np.log1p(self.train_encoded['DisbursementGross'])
        self.train_encoded['Log_GrAppv'] = np.log1p(self.train_encoded['GrAppv'])
        self.train_encoded['Log_SBA_Appv'] = np.log1p(self.train_encoded['SBA_Appv'])
        self.train_encoded['Log_BalanceGross'] = np.log1p(self.train_encoded['BalanceGross'])
        self.train_encoded['TotalJobs'] = self.train_encoded['CreateJob'] + self.train_encoded['RetainedJob']
        self.train_encoded['IncomeToLoanRatio'] = self.train_encoded['DisbursementGross'] / self.train_encoded[
            'SBA_Appv']
        self.train_encoded['EmployeesToLoanRatio'] = self.train_encoded['NoEmp'] / self.train_encoded['SBA_Appv']
        self.train_encoded['JobPerLoan'] = self.train_encoded['TotalJobs'] / self.train_encoded['SBA_Appv']
        self.train_encoded['Gauren_SBA_Appv'] = self.train_encoded['GrAppv'] / self.train_encoded['SBA_Appv']

        return self.train_encoded, self.test_encoded

    def get_train_test_data(self):
        return self.train_encoded, self.test_encoded