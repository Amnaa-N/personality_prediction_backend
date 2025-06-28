from flask import Flask, request, jsonify
from model import load_trained_model, load_scaler
from feature_extractor import extract_handwriting_features
import torch
import numpy as np
import os

app = Flask(__name__)

model = load_trained_model()
scaler = load_scaler()

# Rule-based interpretation
def interpret_personality_from_features(features):
    slant_angle = features[0] * 90
    letter_spacing = features[1] * 100
    line_alignment = features[2] * 100
    loop_ratio = features[3]
    stroke_smoothness = features[4]
    word_spacing = features[5] * 100
    letter_size = features[6] * 100
    legibility = features[7]
    cursive_tendency = features[8]

    interpretations = []

    if slant_angle > 100:
        interpretations.append("You are emotionally expressive and outgoing.")
    elif slant_angle < 70:
        interpretations.append("You are more reserved and logical.")

    if letter_size > 30:
        interpretations.append("You are likely extroverted and confident.")
    elif letter_size < 20:
        interpretations.append("You may be introverted and detail-oriented.")

    if word_spacing > 10:
        interpretations.append("You value personal space and independence.")
    elif word_spacing < 7:
        interpretations.append("You are sociable and seek connection.")

    if loop_ratio > 0.1:
        interpretations.append("You have a vivid imagination and emotional richness.")

    if stroke_smoothness > 1.5:
        interpretations.append("You may have a dynamic or impulsive personality.")
    elif stroke_smoothness < 0.9:
        interpretations.append("You may be more methodical and precise.")

    if cursive_tendency == 1:
        interpretations.append("You may have a holistic or intuitive thinking style.")
    else:
        interpretations.append("You may think more analytically or systematically.")

    return " ".join(interpretations)

@app.route('/')
def home():
    return "Backend is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        temp_path = "uploaded_image.png"
        file.save(temp_path)

        features = extract_handwriting_features(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if features is None or len(features) != 9:
            return jsonify({"error": "Failed to extract valid features from handwriting"}), 400

        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features_scaled = scaler.transform([features])
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(features_tensor).numpy().flatten().tolist()

        rule_based_text = interpret_personality_from_features(features)

        return jsonify({
            "traits": {
                "Openness": prediction[0],
                "Conscientiousness": prediction[1],
                "Extraversion": prediction[2],
                "Agreeableness": prediction[3],
                "Neuroticism": prediction[4]
            },
            "rule_based": rule_based_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
