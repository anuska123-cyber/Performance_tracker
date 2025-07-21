from flask import Flask, render_template, request, jsonify, flash, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs('data', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_data(self, df):
        categorical_columns = ['gender', 'parental_education', 'family_income']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        X = df.drop('pass', axis=1)
        y = df['pass']
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def transform_input(self, input_data):
        df = pd.DataFrame([input_data])
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    df[col] = 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        df_scaled = self.scaler.transform(df)
        return df_scaled

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
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
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
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        model = self.models[model_type]
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        return {
            'probability': probability,
            'prediction': prediction,
            'model_used': model_type
        }

    def save_models(self, path='trained_models/'):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'{path}{name}_model.pkl')
        print("Models saved successfully!")

    def load_models(self, path='trained_models/'):
        try:
            for name in self.models.keys():
                self.models[name] = joblib.load(f'{path}{name}_model.pkl')
            self.is_trained = True
            print("Models loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved models found. Please train the models first.")
            return False

predictor = StudentPredictor()
processor = DataProcessor()

def create_sample_data():
    if not os.path.exists('data/student_data.csv'):
        os.makedirs('data', exist_ok=True)
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
        df['pass_probability'] = (
            0.3 * (df['study_hours'] / 50) +
            0.25 * df['attendance_rate'] +
            0.2 * (df['previous_grade'] / 100) +
            0.1 * df['extracurricular'] +
            0.05 * df['internet_access'] +
            0.05 * df['tutoring'] +
            0.05 * (df['parental_education'].map({'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}) / 3)
        )
        df['pass_probability'] += np.random.normal(0, 0.1, n_samples)
        df['pass'] = (df['pass_probability'] > 0.6).astype(int)
        df = df.drop('pass_probability', axis=1)
        df.to_csv('data/student_data.csv', index=False)
        print("Sample data created successfully!")

def load_and_train_models():
    global predictor, processor
    create_sample_data()
    try:
        df = pd.read_csv('data/student_data.csv')
        required_columns = ['age', 'gender', 'parental_education', 'study_hours', 'attendance_rate', 'previous_grade', 'extracurricular', 'family_income', 'internet_access', 'tutoring', 'pass']
        if any(col not in df.columns for col in required_columns):
            os.remove('data/student_data.csv')
            create_sample_data()
            df = pd.read_csv('data/student_data.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        create_sample_data()
        df = pd.read_csv('data/student_data.csv')
    X_train, X_test, y_train, y_test = processor.prepare_data(df.copy())
    results = predictor.train_models(X_train, X_test, y_train, y_test)
    predictor.save_models()
    joblib.dump(processor, 'trained_models/data_processor.pkl')
    return results

@app.route('/')
def login():
    if session.get('admin_logged_in'):
        return redirect(url_for('home'))
    return redirect(url_for('admin_login'))

@app.route('/home')
def home():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('home.html')

@app.route('/form')
def index():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('index.html')

@app.route('/about')
def about():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template("about.html")

@app.route('/how-it-works')
def how_it_works():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template("how_it_works.html")

@app.route('/contact')
def contact():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template("contact.html")

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('admin_login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'name': request.form.get('name', 'Unknown'),
            'age': int(request.form.get('age', 18)),
            'gender': request.form.get('gender', 'Male'),
            'parental_education': request.form.get('parental_education', 'High School'),
            'study_hours': int(request.form.get('study_hours', 10)),
            'attendance_rate': float(request.form.get('attendance')) / 100,  # ✅ divide by 100
            'previous_grade': float(request.form.get('previous_grade', 75)),
            'extracurricular': int(request.form.get('extracurricular', 0)),
            'family_income': request.form.get('family_income', 'Medium'),
            'internet_access': int(request.form.get('internet_access', 1)),
            'tutoring': int(request.form.get('tutoring', 0))
        }

        model_type = request.form.get('model_type', 'random_forest')
        transformed = processor.transform_input(input_data)
        result = predictor.predict_probability(transformed, model_type)

        def get_confidence(prob):
            return "High" if prob >= 0.8 else "Moderate" if prob >= 0.6 else "Low" if prob >= 0.4 else "Very Low"

        def get_risk(prob):
            return "Low Risk" if prob >= 0.8 else "Medium Risk" if prob >= 0.5 else "High Risk"

        # 🌟 Enhanced Recommendations
        recommendations = []

        # Academic Support
        if input_data['study_hours'] < 10:
            recommendations.append(f"Increase study hours to at least 12–15 hours/week (currently {input_data['study_hours']}/week).")

        if input_data['previous_grade'] < 60:
            recommendations.append("Focus on improving weak subjects using past papers, summaries, or extra coaching.")

        # Attendance
        if input_data['attendance_rate'] < 0.8:
            recommendations.append(f"Improve attendance (currently {input_data['attendance_rate'] * 100:.0f}%) to stay in sync with classroom teaching.")

        # Parental Education
        if input_data['parental_education'] == 'High School':
            recommendations.append("Seek mentorship if parental academic support is limited.")

        # Extracurricular Activities
        if input_data['extracurricular']:
            recommendations.append("Continue extracurricular activities – they boost confidence and time management.")
        else:
            recommendations.append("Try joining a club or activity to improve soft skills and confidence.")

        # Internet Access
        if not input_data['internet_access']:
            recommendations.append("Ensure stable internet access to take advantage of digital study tools and resources.")

        # Tutoring
        if not input_data['tutoring']:
            recommendations.append("Consider joining coaching, peer groups, or online courses for additional academic support.")

        # Study Technique
        if input_data['study_hours'] > 10 and input_data['previous_grade'] < 60:
            recommendations.append("Try more effective study techniques like spaced repetition or Pomodoro for better retention.")

        # General Wellness
        recommendations.append("Maintain a healthy routine: sleep well, eat balanced meals, and manage screen time.")

        summary = {
            "prediction": "PASS" if result['prediction'] == 1 else "FAIL",
            "confidence": f"{round(result['probability'] * 100, 1)}%",
            "risk_level": get_risk(result['probability']),
            "model": f"{model_type.replace('_', ' ').title()} Algorithm",
            "recommendations": recommendations,
            "summary": f"This student has a {round(result['probability'] * 100, 1)}% probability of passing."
        }

        session['student_input'] = input_data
        session['student_result'] = summary
        return redirect(url_for('details'))

    except Exception as e:
        return f"<h2 style='color:red;'>Error: {str(e)}</h2>"


@app.route('/details')
def details():
    input_data = session.get('student_input')
    result = session.get('student_result')
    if not input_data or not result:
        return redirect(url_for('index'))
    return render_template("details.html", student=input_data, result=result)

@app.route('/train', methods=['POST'])
def train_models():
    try:
        results = load_and_train_models()
        formatted_results = {name: {'accuracy': round(result['accuracy'] * 100, 2)} for name, result in results.items()}
        return jsonify({'success': True, 'results': formatted_results, 'message': 'Models trained successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        transformed_data = processor.transform_input(data)
        model_type = data.get('model_type', 'random_forest')
        result = predictor.predict_probability(transformed_data, model_type)
        return jsonify({
            'success': True,
            'probability': result['probability'],
            'prediction': result['prediction'],
            'model_used': result['model_used']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("Initializing Student Performance Predictor...")
    if not predictor.load_models():
        try:
            processor = joblib.load('trained_models/data_processor.pkl')
            print("Data processor loaded successfully!")
        except FileNotFoundError:
            print("No saved processor found. Training from scratch...")
            load_and_train_models()
    else:
        try:
            processor = joblib.load('trained_models/data_processor.pkl')
            print("Data processor loaded successfully!")
        except FileNotFoundError:
            print("Processor not found. Retraining...")
            load_and_train_models()
    print("Application ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)
