

# 📉 Customer Churn Prediction App

A production-ready web application that predicts customer churn using a machine learning model trained on telco user behavior. Built with **Streamlit** for interactivity and deployed with scalable considerations, this project demonstrates applied machine learning, data preprocessing, and model deployment in a business-critical context.
VISIT LIVE :https://churnpredicter.streamlit.app/
---

## 🚀 Overview

Customer retention is a key driver of profitability in subscription-based businesses. This application enables business stakeholders to:

* Upload customer data via a CSV file or input key metrics manually.
* Generate real-time churn predictions using a trained **XGBoost** classifier.
* View a churn risk label: `✅ Likely to Stay` or `⚠️ High Risk of Churn`.

---

## 🗃️ Source Dataset

This project uses the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from **Kaggle**, which contains demographic, account, and service usage information of a telecom company's customers.

**Dataset Highlights:**

* 7,043 customer records
* Features include tenure, contract type, billing methods, and service usage
* Binary target: `Churn` (Yes/No)

Credit: [Telco Customer Churn Dataset by IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🎯 Key Features

* ✅ **User-Friendly Interface:** Streamlit-based UI for quick predictions.
* 🔄 **Batch & Single Input Modes:** Upload entire datasets or input customer details manually.
* 📈 **Scalable ML Backend:** Trained using XGBoost and robust preprocessing pipelines.
* 🧪 **Production Ready:** Uses `joblib` to persist models and scalers.
* 🧠 **Churn Risk Labeling:** Visual and textual indicators for churn likelihood.
* 📊 **Preprocessing Pipeline:** Includes scaling, encoding, and input validation.

---

## 📂 Project Structure

```
Customer-Churn-Prediction-App/
│
├── app.py                    # Streamlit web app
├── churn_prediction.ipynb    # Data preprocessing, EDA, and model training
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   ├── scaler.pkl            # StandardScaler object
│   └── feature_names.pkl     # Feature schema used during training
├── requirements.txt
└── README.md
```

---

## 🧪 Model & Data

* **Dataset:** Telco Customer Churn dataset (Kaggle)
* **Algorithm:** XGBoost Classifier
* **Target:** `Churn` (binary classification)
* **Training Pipeline:**

  * Missing value imputation
  * Encoding categorical variables
  * Scaling numerical features
* **Evaluation Metrics:**

  * Accuracy
  * ROC-AUC Score
  * Precision/Recall

---

## 🛠️ How to Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/NAIDU0019/Customer-Churn-Prediction-App.git
cd Customer-Churn-Prediction-App
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 📥 Sample Input Format

For batch prediction, upload a CSV file with the following format:

```csv
tenure,MonthlyCharges,TotalCharges,Contract,...
12,75.6,890.5,Month-to-month,...
45,110.9,4530.1,Two year,...
```

---

## 🧠 Risk Label Logic

| Churn Prediction | Risk Label            |
| ---------------- | --------------------- |
| 1 (Churn Likely) | ⚠️ High Risk of Churn |
| 0 (No Churn)     | ✅ Likely to Stay      |

---

## 💼 Use Case Relevance

This project is directly aligned with real-world applications in:

* **E-commerce (Meesho)**: Retaining vendors or customers through data-driven insights.
* **Telecom**: Reducing user churn through proactive engagement.
* **SaaS Platforms**: Predictive user analytics and personalized retention strategies.

---

## 📚 Tech Stack

| Layer            | Tools & Frameworks          |
| ---------------- | --------------------------- |
| Language         | Python                      |
| ML Model         | XGBoost                     |
| Web Framework    | Streamlit                   |
| Data Processing  | Pandas, NumPy, Scikit-learn |
| Deployment Ready | joblib, streamlit, pickle   |

---

## 👨‍💻 Author

**Rajappa Adabala**
Final Year B.Tech | Data Scientist | ML Engineer
📧 [rajappaadabala@gmail.com](mailto:rajappaadabala@gmail.com)
🔗 [GitHub](https://github.com/NAIDU0019) • [LinkedIn](https://linkedin.com/in/rajappaadabala)

---

## 📌 License

This project is released under the MIT License.

---


