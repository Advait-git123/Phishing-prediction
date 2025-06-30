This project is a machine learning-based phishing email classifier built using Python, scikit-learn, and Streamlit. It enables users to paste or input email content and get instant predictions on whether the email is phishing or legitimate, along with a confidence score.

---

 Features

- TF-IDF-based text vectorization
- Logistic Regression model trained on real-world phishing datasets
- Simple Streamlit web app interface
- Confidence score for each prediction
- Easily extensible to BERT or other models

---

Dataset

I used a combination of publicly available phishing and ham (legitimate) email datasets from the following sources:

| Dataset Name      | Source                                      | Notes                         |
|-------------------|---------------------------------------------|-------------------------------|
| CEAS 2008         | Anti-Spam Conference (CEAS)                 | Removed from repo due to size |
| Enron Emails      | [Enron corpus](https://www.cs.cmu.edu/~enron/) | Cleaned & filtered            |
| Nigerian Fraud    | Nazario, SpamAssassin datasets              | Phishing scams and fraud      |
| Ling              | Academic spam email corpus                  | Academic email ham/spam       |
| SpamAssassin      | Apache SpamAssassin public dataset          | Pre-labeled spam/ham          |
| phishing_email.csv| Combined version used for training          | Over 100 MB — not included    |

⚠️ **Note:** Due to GitHub's 100 MB file size limit, large datasets like `phishing_email.csv` and `CEAS_08.csv` are not included in this repository.  
