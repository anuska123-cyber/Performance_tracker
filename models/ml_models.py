import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class StudentPredictor:
    def __init__(self):
        self.lr_model = LogisticRegression(random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models = {
            'logistic_regression': self.lr_model,
            'random_forest': self.rf_model
        }
        self.is_trained = False
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train both models"""
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name.title()} Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        return results
    
    def predict_probability(self, input_data, model_type='random_forest'):
        """Predict pass probability for given input"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        model = self.models[model_type]
        
        # Get probability of passing (class 1)
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        return {
            'probability': probability,
            'prediction': prediction,
            'model_used': model_type
        }
    
    def get_feature_importance(self, model_type='random_forest'):
        """Get feature importance for Random Forest"""
        if model_type == 'random_forest' and self.is_trained:
            return self.rf_model.feature_importances_
        return None
    
    def save_models(self, path='trained_models/'):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'{path}{name}_model.pkl')
        
        print("Models saved successfully!")
    
    def load_models(self, path='trained_models/'):
        """Load trained models"""
        try:
            for name in self.models.keys():
                self.models[name] = joblib.load(f'{path}{name}_model.pkl')
            self.is_trained = True
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("No saved models found. Please train the models first.")