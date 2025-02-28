# model_prediction.py
import pickle
import streamlit
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(news_input, vectorizer, models):
    # Vectorize input text
    news_vectorized = vectorizer.transform([news_input])
    print(f"Number of features in news vectorized: {news_vectorized.shape[1]}")

    results = {}
    for model_name, model in models.items():
        try:
            prediction = model.predict(news_vectorized)
            results[model_name] = "Fake News" if prediction[0] == 0 else "Not Fake News"
        except ValueError as e:
            results[model_name] = f"Error: {str(e)}"

    return results
