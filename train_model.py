# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")

# Step 2: Clean and prepare data
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df = df[['text', 'label']]  # Keep only relevant columns
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert to binary

# Step 3: Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_vec)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
