from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "sentiment": "Neutral",
            "confidence": 0
        })

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    confidence = round(max(probabilities) * 100, 2)

    return jsonify({
        "sentiment": prediction.capitalize(),
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)