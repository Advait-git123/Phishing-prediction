import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from utils.preprocess import clean_text, vectorize_text

def extract_text_column(df):
    """Finds and returns a unified 'text' column from any valid CSV structure"""
    for text_col in ['text_combined', 'body', 'text', 'message']:
        if text_col in df.columns:
            return df[text_col]
    # Fallback: combine subject + body if both are present
    if 'subject' in df.columns and 'body' in df.columns:
        return df['subject'].fillna('') + ' ' + df['body'].fillna('')
    raise ValueError("No valid email text column found.")

def main():
    csv_files = glob.glob(os.path.join("data", "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in 'data/' folder.")

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df.columns = [col.lower().strip() for col in df.columns]

            if 'label' not in df.columns:
                print(f" Skipping {file} — missing 'label' column.")
                continue

            df['text'] = extract_text_column(df)
            df = df[['text', 'label']]
            df_list.append(df)

        except Exception as e:
            print(f" Skipping {file}: {e}")

    if not df_list:
        raise ValueError("No usable CSV files with 'label' and 'text' found.")

    df = pd.concat(df_list, ignore_index=True)

    # Clean and encode
    df['text'] = df['text'].fillna("").apply(clean_text)
    print("🧪 Unique label values BEFORE mapping:", df['label'].astype(str).str.lower().unique())
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    # Drop rows with unmapped labels
    df.dropna(subset=['label'], inplace=True)
    print("📊 Final label distribution:\n", df['label'].value_counts())
    print("🔢 Total usable emails:", len(df))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("\n Model Trained Successfully!\n")
    print(classification_report(y_test, model.predict(X_test_vec), digits=4))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/phishing_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("\n Model and vectorizer saved to 'model/' folder.")

if __name__ == "__main__":
    main()
