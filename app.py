from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np
from waitress import serve  # Import Waitress for production server

# Load trained model
model = pickle.load(open("ml_model.pkl", "rb"))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # User input from frontend
    features = np.array([data["study_hours_per_day"], data["mental_health_rating"], 
                         data["social_media_hours"], data["sleep_hours"], 
                         data["netflix_hours"], data["exercise_frequency"], 
                         data["attendance_percentage"]]).reshape(1, -1)
    
    predicted_score = model.predict(features)[0]  # Generate score
    final_score = max(0, min(predicted_score, 100))
    return jsonify({"predicted_exam_score": final_score})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)  # Use Waitress instead of app.run()
