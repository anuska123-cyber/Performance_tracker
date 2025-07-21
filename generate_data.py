import pandas as pd
import numpy as np
import os

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

# Generate synthetic student data
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(16, 25, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'parental_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'study_hours': np.random.randint(0, 50, n_samples),
    'attendance_rate': np.random.uniform(0.5, 1.0, n_samples),
    'previous_grade': np.random.uniform(40, 95, n_samples),
    'extracurricular': np.random.choice([0, 1], n_samples),
    'family_income': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'internet_access': np.random.choice([0, 1], n_samples),
    'tutoring': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

# Create target variable (pass/fail) based on logical rules
df['pass_probability'] = (
    0.3 * (df['study_hours'] / 50) +
    0.25 * df['attendance_rate'] +
    0.3 * (df['previous_grade'] / 100) +
    0.1 * df['extracurricular'] +
    0.05 * df['internet_access']
)

df['pass'] = (df['pass_probability'] + np.random.normal(0, 0.1, n_samples) > 0.6).astype(int)
df = df.drop('pass_probability', axis=1)

# Save to CSV
df.to_csv('data/student_data.csv', index=False)

print(f"Generated dataset with {len(df)} samples")
print(f"Columns: {list(df.columns)}")
print(f"Pass rate: {df['pass'].mean():.2%}")
print("\nFirst few rows:")
print(df.head())

# Verify the CSV file was created correctly
try:
    test_df = pd.read_csv('data/student_data.csv')
    print(f"\nCSV file verified successfully!")
    print(f"Shape: {test_df.shape}")
except Exception as e:
    print(f"Error reading CSV file: {e}")