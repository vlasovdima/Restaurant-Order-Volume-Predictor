# Restaurant-Order-Volume-Predictor

Markdown
# QSR Traffic Forecast 🍔📈

An end-to-end Machine Learning pipeline built to predict hourly customer traffic in quick-service restaurants. This project demonstrates how data-driven forecasting can solve real-world operational challenges, such as staff scheduling, peak hour management, and workload distribution.

 📌 Problem Statement
In high-intensity food service environments, understaffing during rush hours leads to SLA violations and customer dissatisfaction, while overstaffing during calm hours results in financial losses. This ML system predicts the exact order volume for any given hour, allowing managers to optimize shifts dynamically.

 🧠 Technical Overview
Algorithm:** `RandomForestRegressor` from scikit-learn.
Feature Engineering:** Extracts temporal patterns (hour, day of the week, weekend flags) and integrates external factors (weather conditions).
Data Pipeline:** Includes a custom synthetic data generator that simulates realistic restaurant traffic over a 6-month period, accounting for lunch/dinner rush multipliers and weather impact.
Validation:** Implements time-series-aware splitting to prevent data leakage during model training and evaluation.

 Quick Start

 1. Setup Environment
Clone the repository and set up a virtual environment:
bash
git clone
cd QSR-Traffic-Forecast
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn
1. Run the ML Pipeline
Execute the main script to generate data, train the model, and run randomized inference tests:

Bash
python restaurant_ml.py
3. Example Output
The CLI interface will output the training metrics (MAE) and run inference scenarios demonstrating the model's logic:

Plaintext
💼 Будний   | Вторник     | 🕒 13:00 | ☀️ Супер    -> Ждем ~128 заказов
🌴 Выходной | Суббота     | 🕒 23:00 | 🌧️ Ливень   -> Ждем ~21 заказов
💼 Будний   | Пятница     | 🕒 19:00 | ☁️ Норм     -> Ждем ~135 заказов
