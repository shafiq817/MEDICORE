from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from flask.json.provider import DefaultJSONProvider

# Custom JSON Provider to handle NumPy types automatically
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__, static_folder='.')
app.json = CustomJSONProvider(app)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# --- Data Loading & Model Training ---
CSV_PATH = os.path.join(os.path.dirname(__file__), 'patients.csv')

try:
    df = pd.read_csv(CSV_PATH)
    df['condition'] = df['condition'].fillna('None')
    
    # Preprocessing for ML
    le_gender = LabelEncoder()
    le_dept = LabelEncoder()
    le_cond = LabelEncoder()
    
    df_ml = df.copy()
    df_ml['gender_enc'] = le_gender.fit_transform(df['gender'])
    df_ml['dept_enc'] = le_dept.fit_transform(df['department'])
    df_ml['cond_enc'] = le_cond.fit_transform(df['condition'])
    
    # Define target: High risk if any vital is outside normal range
    df_ml['risk'] = ((df['bp_systolic'] > 140) | 
                     (df['glucose'] > 200) | 
                     (df['cholesterol'] > 240)).astype(int)
    
    features = ['age', 'gender_enc', 'dept_enc', 'cond_enc', 'bp_systolic', 'glucose', 'cholesterol', 'heart_rate']
    X = df_ml[features]
    y = df_ml['risk']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"Model accuracy: {model.score(X, y):.3f}")
    print(f"Loaded {len(df)} patient records")
except Exception as e:
    print(f"Error loading data or training model: {e}")
    df = pd.DataFrame() # Fallback

# --- Routes ---

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app.route('/api/stats')
def stats():
    if df.empty:
        return jsonify({"error": "No data available"}), 404
        
    total = len(df)
    recovered = int((df['outcome'] == 'Recovered').sum())
    
    # Age Distribution
    age_bins = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81+'])
    age_dist = age_bins.value_counts().sort_index().to_dict()

    return jsonify({
        'total_patients': total,
        'recovery_rate': round(recovered/total*100, 1) if total > 0 else 0,
        'avg_los': round(float(df['length_of_stay'].mean()), 1),
        'total_revenue': round(float(df['billing_amount'].sum()), 2),
        'emergency_cases': int((df['admission_type'] == 'Emergency').sum()),
        'dept_counts': df['department'].value_counts().to_dict(),
        'condition_counts': df['condition'].value_counts().head(6).to_dict(),
        'outcome_counts': df['outcome'].value_counts().to_dict(),
        'insurance_counts': df['insurance'].value_counts().to_dict(),
        'gender_counts': df['gender'].value_counts().to_dict(),
        'monthly_admissions': df.groupby('admission_month').size().sort_index().to_dict(),
        'avg_billing_dept': df.groupby('department')['billing_amount'].mean().round(2).to_dict(),
        'age_distribution': age_dist,
        'avg_vitals': {
            'bp_systolic': round(float(df['bp_systolic'].mean()), 1),
            'bp_diastolic': round(float(df['bp_diastolic'].mean()), 1),
            'heart_rate': round(float(df['heart_rate'].mean()), 1),
            'glucose': round(float(df['glucose'].mean()), 1),
            'cholesterol': round(float(df['cholesterol'].mean()), 1),
        }
    })

@app.route('/api/patients')
def patients():
    page = int(request.args.get('page', 1))
    per_page = 15
    dept = request.args.get('dept', '')
    search = request.args.get('search', '')
    
    filtered = df.copy()
    if dept:
        filtered = filtered[filtered['department'] == dept]
    if search:
        mask = (filtered['patient_id'].astype(str).str.contains(search, case=False) |
                filtered['doctor'].str.contains(search, case=False))
        filtered = filtered[mask]
        
    total = len(filtered)
    start = (page - 1) * per_page
    page_df = filtered.iloc[start : start + per_page].replace({np.nan: None})
    
    return jsonify({
        'patients': page_df.to_dict(orient='records'),
        'total': total,
        'pages': max(1, (total + per_page - 1) // per_page),
        'current_page': page
    })

@app.route('/api/predict_risk', methods=['POST', 'OPTIONS'])
def predict_risk():
    if request.method == 'OPTIONS':
        return '', 200
        
    d = request.json
    try:
        # Get encoding safely; default to first class if unseen
        def safe_encode(le, val):
            try:
                return int(le.transform([val])[0])
            except:
                return 0

        g_enc = safe_encode(le_gender, d.get('gender', 'Male'))
        d_enc = safe_encode(le_dept, d.get('department', 'General Medicine'))
        c_enc = safe_encode(le_cond, d.get('condition', 'None'))

        X_input = [[
            int(d.get('age', 50)), g_enc, d_enc, c_enc,
            int(d.get('bp_systolic', 120)), int(d.get('glucose', 100)),
            int(d.get('cholesterol', 180)), int(d.get('heart_rate', 75))
        ]]
        
        prob = float(model.predict_proba(X_input)[0][1])
        level = 'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low'
        
        return jsonify({'risk_probability': round(prob * 100, 1), 'risk_level': level})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/doctor_stats')
def doctor_stats():
    if df.empty:
        return jsonify([])
    s = df.groupby('doctor').agg(
        patients=('patient_id', 'count'),
        avg_los=('length_of_stay', 'mean'),
        total_billing=('billing_amount', 'sum'),
        recovery_rate=('outcome', lambda x: round((x == 'Recovered').sum() / len(x) * 100, 1))
    ).round(2).reset_index()
    return jsonify(s.to_dict(orient='records'))

if __name__ == '__main__':
    print("\n[+] MediCore Healthcare Management System")
    print("[:] Serving API at: http://localhost:5050")
    app.run(debug=True, port=5050)
