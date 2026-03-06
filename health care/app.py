from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import json
from flask.json.provider import DefaultJSONProvider

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
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# Load data
CSV = os.path.join(os.path.dirname(__file__), 'patients.csv')
df = pd.read_csv(CSV)
df['condition'] = df['condition'].fillna('None')

le_gender = LabelEncoder()
le_dept = LabelEncoder()
le_cond = LabelEncoder()
df_ml = df.copy()
df_ml['gender_enc'] = le_gender.fit_transform(df['gender'])
df_ml['dept_enc'] = le_dept.fit_transform(df['department'])
df_ml['cond_enc'] = le_cond.fit_transform(df['condition'])
df_ml['risk'] = ((df['bp_systolic'] > 140) | (df['glucose'] > 200) | (df['cholesterol'] > 240)).astype(int)
X = df_ml[['age','gender_enc','dept_enc','cond_enc','bp_systolic','glucose','cholesterol','heart_rate']]
y = df_ml['risk']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print(f"Model accuracy: {model.score(X,y):.3f}")
print(f"Loaded {len(df)} patient records")

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

@app.route('/api/stats')
def stats():
    total = len(df)
    recovered = int((df['outcome'] == 'Recovered').sum())
    avg_los = round(float(df['length_of_stay'].mean()), 1)
    total_rev = round(float(df['billing_amount'].sum()), 2)
    emergency = int((df['admission_type'] == 'Emergency').sum())
    age_bins = pd.cut(df['age'], bins=[0,20,40,60,80,100], labels=['0-20','21-40','41-60','61-80','81+'])
    age_dist = {str(k): int(v) for k,v in age_bins.value_counts().sort_index().items()}
    return jsonify({
        'total_patients': total,
        'recovery_rate': round(recovered/total*100, 1),
        'avg_los': avg_los,
        'total_revenue': total_rev,
        'emergency_cases': emergency,
        'dept_counts': df['department'].value_counts().astype(int).to_dict(),
        'condition_counts': df['condition'].value_counts().head(6).astype(int).to_dict(),
        'outcome_counts': df['outcome'].value_counts().astype(int).to_dict(),
        'insurance_counts': df['insurance'].value_counts().astype(int).to_dict(),
        'gender_counts': df['gender'].value_counts().astype(int).to_dict(),
        'monthly_admissions': df.groupby('admission_month').size().sort_index().astype(int).to_dict(),
        'avg_billing_dept': {k: round(float(v),2) for k,v in df.groupby('department')['billing_amount'].mean().items()},
        'age_distribution': age_dist,
        'avg_vitals': {
            'bp_systolic': round(float(df['bp_systolic'].mean()),1),
            'bp_diastolic': round(float(df['bp_diastolic'].mean()),1),
            'heart_rate': round(float(df['heart_rate'].mean()),1),
            'glucose': round(float(df['glucose'].mean()),1),
            'cholesterol': round(float(df['cholesterol'].mean()),1),
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
        mask = (filtered['patient_id'].str.contains(search, case=False) |
                filtered['doctor'].str.contains(search, case=False))
        filtered = filtered[mask]
    total = len(filtered)
    start = (page-1)*per_page
    page_df = filtered.iloc[start:start+per_page]
    page_df = page_df.replace({np.nan: None})
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
        g_enc = int(le_gender.transform([d.get('gender','Male')])[0])
        d_enc = int(le_dept.transform([d.get('department','General Medicine')])[0])
        c_enc = int(le_cond.transform([d.get('condition','None')])[0])
    except:
        g_enc = d_enc = c_enc = 0
    X_pred = [[int(d.get('age',50)), g_enc, d_enc, c_enc,
               int(d.get('bp_systolic',120)), int(d.get('glucose',100)),
               int(d.get('cholesterol',180)), int(d.get('heart_rate',75))]]
    prob = float(model.predict_proba(X_pred)[0][1])
    level = 'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low'
    return jsonify({'risk_probability': round(prob*100,1), 'risk_level': level})

@app.route('/api/doctor_stats')
def doctor_stats():
    s = df.groupby('doctor').agg(
        patients=('patient_id','count'),
        avg_los=('length_of_stay','mean'),
        total_billing=('billing_amount','sum'),
        recovery_rate=('outcome', lambda x: round((x=='Recovered').sum()/len(x)*100,1))
    ).round(2).reset_index()
    return jsonify(s.to_dict(orient='records'))

if __name__ == '__main__':
    print("\n[+] MediCore Healthcare Management System")
    print("[:] Visit: http://localhost:5050\n")
    app.run(debug=True, port=5050)
