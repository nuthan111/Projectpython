from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import pickle

def load_and_clean_data():
    # Load the dataset (Update the file path accordingly)
    data_fake = pd.read_csv('Fake.csv', encoding='latin1')
    data_true = pd.read_csv('True.csv', encoding='latin1')

    # Assign class labels (0 = Fake, 1 = True)
    data_fake['class'] = 0
    data_true['class'] = 1

    # Combine datasets
    data = pd.concat([data_fake, data_true], axis=0).reset_index(drop=True)

    # Drop missing values
    data.dropna(inplace=True)

    # Select only relevant columns (Modify based on dataset structure)
    if 'text' not in data.columns:
        raise ValueError("The dataset must contain a 'text' column.")

    return data

def preprocess_text(data):
    # Convert text to lowercase and remove unwanted spaces
    data['text'] = data['text'].str.lower().str.strip()
    return data

def train_models(data):
    # Split data
    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=42)

    # Text Vectorization
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Train models
    models = {
        'logistic_regression': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(random_state=0)
    }

    for model_name, model in models.items():
        model.fit(xv_train, y_train)
        pred = model.predict(xv_test)
        print(f"Model: {model_name.replace('_', ' ').title()}")
        print(classification_report(y_test, pred))

        # Save model
        with open(f'{model_name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"File created: {model_name}_model.pkl")

    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorization, f)
    print("File created: vectorizer.pkl")

if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_clean_data()
    data = preprocess_text(data)

    # Train models
    train_models(data)
