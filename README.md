# AI Customer Churn Intelligence Platform

A machine learning powered customer churn prediction web application built with **Python, Scikit-learn, Flask, Pandas, HTML, CSS, and Chart.js**.

The application predicts the probability of a customer leaving a bank and presents the result through an interactive dashboard.

## Project Overview

Customer churn is an important business problem for banks and other customer-focused companies.

This project uses machine learning to analyze customer information and predict whether a customer is likely to churn.

The trained model is integrated into a **Flask web application** where users can enter customer information and receive:

* Churn prediction
* Churn probability
* Risk level
* Customer analytics
* Country-wise churn insights
* Gender-wise churn insights
* Active vs inactive customer churn
* Age-group churn analysis
* Product-wise churn analysis
* Customer search and filtering

## Features

### Customer Churn Prediction

Users can enter:

* Credit Score
* Country
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary

The application processes the input using the same preprocessing pipeline used during model training and generates a churn prediction.
## Project Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)
![Dashboard](screenshots/dashboard(2).png)

### Churn Prediction
![Prediction](screenshots/prediction.png)

### Analytics
![Analytics](screenshots/analytics.png)
![Analytics](screenshots/analytics(2).png)

### AI Risk Analysis

The application displays:

* Churn Probability
* Low Risk
* Medium Risk
* High Risk
* Customer retention indication

### Analytics Dashboard

The dashboard provides visual insights into customer churn using charts.

Analytics include:

* Overall churn distribution
* Country-wise churn
* Gender-wise churn
* Active member churn
* Age-group churn
* Product-wise churn

### Customer Management

The application includes a customer table with:

* Customer ID
* Country
* Age
* Balance
* Products
* Churn Status

Users can search and filter customers by:

* Customer ID
* Country
* Churn Status

Pagination is also included for easier data browsing.

## Machine Learning Workflow

The machine learning workflow used in this project:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Flask Integration
   ↓
Web Prediction
```

## Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* StandardScaler

### Model Deployment

* Flask
* Joblib

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

### Development

* VS Code
* Google Colab

## Project Structure

```text
customer_churn_flask/
│
├── app.py
├── Bank Customer Churn Prediction.csv
├── churn_model (1).joblib
├── scaler (2).pkl
├── feature_names.pkl
│
├── templates/
│   └── index.html
│
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project

```bash
cd customer_churn_flask
```

### 3. Install dependencies

```bash
pip install flask pandas numpy scikit-learn joblib
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open in browser

```text
http://127.0.0.1:5000
```

## Example Prediction

Example output:

```text
Churn Probability: 24.0%
Risk Level: LOW RISK
Status: Customer likely to stay
```

## Dashboard

The application provides a complete dashboard for exploring customer churn patterns and testing individual customer predictions.

## What I Learned

Through this project, I practiced:

* Data preprocessing
* Categorical encoding
* Feature scaling
* Machine learning model training
* Model prediction
* Model serialization with Joblib
* Flask application development
* Connecting ML models with Flask
* Building interactive dashboards
* Data visualization
* Customer filtering and search
* Deploying machine learning functionality into a web application

## Future Improvements

Possible future improvements include:

* Model comparison
* Customer retention recommendations
* Authentication system
* Database integration
* Real-time customer data
* Advanced model explainability
* Cloud deployment
* REST API
* Automated retraining pipeline

## Author

**Muhammad Aslam**

Aspiring **AI Specialist | Python Developer | Machine Learning & Automation**

---

⭐ If you find this project useful, feel free to star the repository.
