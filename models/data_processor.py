import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Handle categorical variables
        categorical_columns = ['gender', 'parental_education', 'family_income']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Separate features and target
        X = df.drop('pass', axis=1)
        y = df['pass']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def transform_input(self, input_data):
        """Transform user input for prediction"""
        df = pd.DataFrame([input_data])
        
        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns
        df = df[self.feature_columns]
        
        # Scale the data
        df_scaled = self.scaler.transform(df)
        
        return df_scaled