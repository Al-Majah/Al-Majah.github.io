from flask import Flask, request, jsonify
import pickle
import numpy as np
from waitress import serve  # Import Waitress for production server

# Load trained model
model = pickle.load(open("ml_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # User input from frontend
    features = np.array([data["study_hours_per_day"], data["mental_health_rating"], 
                         data["social_media_hours"], data["sleep_hours"], 
                         data["netflix_hours"], data["exercise_frequency"], 
                         data["attendance_percentage"]]).reshape(1, -1)
    
    prediction = model.predict(features)[0]  # Generate score
    return jsonify({"predicted_exam_score": prediction})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)  # Use Waitress instead of app.run()
