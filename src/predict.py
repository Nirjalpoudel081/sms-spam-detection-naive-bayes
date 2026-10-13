
import joblib
from pathlib import Path

# Locate the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

# Load the saved model and TF-IDF vectorizer
model = joblib.load(MODEL_DIR / "multinomial_nb_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")


def predict_sms(message):
    """
    Predict whether an SMS message is ham or spam.

    Parameters:
        message (str): The SMS message to classify.

    Returns:
        dict: Predicted label and estimated spam probability.
    """
    # Convert the message into TF-IDF features
    message_tfidf = vectorizer.transform([message])

    # Predict the class and spam probability
    prediction = model.predict(message_tfidf)[0]
    spam_probability = model.predict_proba(message_tfidf)[0, 1]

    # Convert the numeric prediction into a readable label
    label = "Spam" if prediction == 1 else "Ham"

    return {
        "message": message,
        "prediction": label,
        "spam_probability": round(float(spam_probability), 4)
    }


if __name__ == "__main__":
    message = input("Enter an SMS message: ")

    result = predict_sms(message)

    print("\nPrediction:", result["prediction"])
    print(f"Estimated spam probability: {result['spam_probability']:.2%}")
