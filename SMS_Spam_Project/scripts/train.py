import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ===============================
# CONFIG
# ===============================
DATA_PATH = "../data/spam.csv"
MODEL_PATH = "../models/knn_spam_model.joblib"
MAX_FEATURES = 3000

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df = df[['v1','v2']]  # columns: v1=label, v2=text
df.columns = ['label', 'text']

# Convert label to numeric
df['label_num'] = df['label'].map({'ham':0, 'spam':1})

X_text = df['text'].values
y = df['label_num'].values

# ===============================
# TEXT VECTORIZATION
# ===============================
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X = vectorizer.fit_transform(X_text)

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAIN KNN MODEL
# ===============================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===============================
# SAVE MODEL & VECTORIZER
# ===============================
os.makedirs("../models", exist_ok=True)
joblib.dump((model, vectorizer), MODEL_PATH)

print("✅ Model trained and saved successfully")
