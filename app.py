from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('spam_model.pkl')

# Load vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Home route
@app.route('/')
def home():

    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get message
    message = request.form['message']

    # Transform message
    transformed_message = vectorizer.transform([message])

    # Predict
    prediction = model.predict(transformed_message)[0]

    # Result
    if prediction == 1:

        result = "SPAM MESSAGE"

    else:

        result = "NOT SPAM"

    return render_template(
        'index.html',
        prediction=result,
        message=message
    )

# Run app
if __name__ == "__main__":

    app.run(debug=True)