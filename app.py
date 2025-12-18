# app.py (Orang 2 - Backend Developer)
# Tujuan: Menyediakan API untuk prediksi suhu.

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS # Diperlukan untuk menghubungkan Frontend dan Backend

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS agar browser (Frontend) bisa mengakses API

# --- 1. MEMUAT MODEL DAN SCALER ---
MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("SUCCESS: Model dan Scaler berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File model/scaler tidak ditemukan. Pastikan '{MODEL_PATH}' dan '{SCALER_PATH}' ada.")
    exit()
except Exception as e:
    print(f"ERROR: Gagal memuat model/scaler: {e}")
    exit()

@app.route('/')
def home():
    return "API Prediksi Suhu Ruangan Berjalan. Gunakan endpoint /predict (POST)."

@app.route('/predict', methods=['POST'])
def predict_suhu():
    try:
        # Menerima data JSON dari Frontend
        data = request.get_json(force=True)
        
        # Urutan fitur HARUS SAMA seperti saat training
        features = pd.DataFrame([data])
        features = features[['Suhu_Luar', 'Kelembaban', 'Jam_Hari', 'Status_AC']]
        
        # Scaling data input
        features_scaled = scaler.transform(features)
        
        # Prediksi
        prediction = model.predict(features_scaled)
        
        # Format hasil
        predicted_suhu = float(prediction[0])
        
        return jsonify({
            'prediksi_suhu': f"{predicted_suhu:.2f}",
            'satuan': 'Celcius'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Format input data salah atau ada kesalahan pada server.'}), 400

if __name__ == '__main__':
    print("--- FASE 3: API FLASK DIMULAI ---")
    # Jalankan API. Host '0.0.0.0' memungkinkan akses dari luar localhost (jika dideploy)
    app.run(debug=True, host='0.0.0.0', port=5000)