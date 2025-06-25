# api.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from assets_data_prep import prepare_data

app = Flask(__name__)

model = joblib.load("trained_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {key: request.form.get(key) for key in request.form}
    df = pd.DataFrame([input_data])
    if 'garden_area' not in df.columns or df['garden_area'].isnull().all():
        df['garden_area'] = 0
    # המרת שדות מספריים ובוליאניים
    numeric_fields = ['room_num', 'floor', 'area', 'garden_area', 'days_to_enter', 'num_of_payments',
                      'monthly_arnona', 'building_tax', 'total_floors', 'num_of_images', 'distance_from_center']
    bool_fields = ['has_parking', 'has_storage', 'elevator', 'ac', 'handicap', 'has_bars',
                   'has_safe_room', 'has_balcony', 'is_furnished', 'is_renovated']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')
    for field in bool_fields:
        df[field] = df[field].astype(int)

    df_processed = prepare_data(df, st="test", with_price=False)
    prediction = model.predict(df_processed)[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
