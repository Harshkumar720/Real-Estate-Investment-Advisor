
# 🏠 Real Estate Investment Advisor

An AI-powered web application that helps users analyze whether a property is a **good investment** and predicts its **future price growth over 5 years**.

---

## 🚀 Features

* 🔍 Predict whether a property is a **Good Investment**
* 📈 **Year-wise price prediction (1 to 5 years)**
* 💡 **Smart suggestions**

  * Reduce price to make it profitable
  * Increase size to improve investment value
* 📊 Confidence score & risk level (Low / Moderate / High)
* 🧠 Dynamic reasoning (why the model gave this result)
* 🎯 Clean and interactive Streamlit UI

---

## 🛠️ Languages & Tools Used

* **Python** 🐍
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine Learning models
* **Streamlit** – Web app framework
* **Matplotlib** – Visualization
* **Jupyter Notebook – EDA & experimentation
* **Git & GitHub** – Version control

---

## 🧠 Machine Learning Models

* **Classification Model**

  * Algorithm: Random Forest Classifier
  * Purpose: Predict if property is a good investment

* **Regression Model**

  * Algorithm: Random Forest Regressor
  * Purpose: Predict future property price

---

## 📂 Project Structure

```
Real-Estate-Investment-Advisor/
│
├── app/
│   └── app.py                  # Streamlit UI
│
├── src/
│   ├── train_classification_model.py
│   ├── train_regression_model.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   └── evaluate_models.py
│
├── models/
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   ├── city_encoder.pkl
│   ├── type_encoder.pkl
│   ├── furnish_encoder.pkl
│   ├── clf_features.json
│   └── reg_features.json
│
├── data/
│   ├── raw/
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   ├── eda_analysis.ipynb
│   └── fix_dataset.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Harshkumar720/Real-Estate-Investment-Advisor.git
cd Real-Estate-Investment-Advisor
```

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the app

```bash
streamlit run app/app.py
```

---

## 📊 How It Works

1. User inputs property details (city, size, price, etc.)
2. Data is preprocessed and encoded
3. Classification model predicts:

   * ✅ Good Investment
   * ❌ Not a Good Investment
4. Regression model predicts future price
5. App displays:

   * Year-wise price growth
   * Confidence score
   * Risk level
   * Suggestions for improvement

---

## 💡 Example Output

* ✅ Good Investment
* 📈 Year 1 → Year 5 price prediction
* 📊 Confidence: 82%
* 🟢 Low Risk
* 💡 Suggestion: Reduce price by ₹X to improve ROI

---

## ⚠️ Notes

* Large datasets may not preview directly on GitHub (normal behavior)
* Models are optimized for deployment (reduced size for performance)

---

## 👨‍💻 Author

**Harsh Kumar**
🔗 GitHub: https://github.com/Harshkumar720

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub and share it!
