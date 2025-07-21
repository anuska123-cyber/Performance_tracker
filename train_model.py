# Create this file to train models separately if needed
import pandas as pd
from models.ml_models import StudentPredictor
from models.data_processor import DataProcessor

def main():
    # Generate sample data (if not exists)
    # ... (use the data generation code from Step 2.1)
    
    # Initialize
    predictor = StudentPredictor()
    processor = DataProcessor()
    
    # Load and prepare data
    df = pd.read_csv('data/student_data.csv')
    X_train, X_test, y_train, y_test = processor.prepare_data(df)
    
    # Train models
    results = predictor.train_models(X_train, X_test, y_train, y_test)
    
    # Save models
    predictor.save_models()
    
    import joblib
    joblib.dump(processor, 'trained_models/data_processor.pkl')
    
    print("Training completed and models saved!")

if __name__ == "__main__":
    main()