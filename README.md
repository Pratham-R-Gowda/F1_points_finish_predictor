# 🏎️ F1 Race Outcome Predictor: Grid to Checkered Flag

## 📌 Project Overview

This project applies Machine Learning to Formula 1 historical data to predict whether a driver will finish in the points (Top 10). Rather than relying purely on driver skill, this model quantifies the impact of Starting Grid Position and Constructor Strength on race outcomes.

## 🛠️ Tech Stack & Tools

- **Data Ingestion:** `FastF1` API (with local caching for rapid execution)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Random Forest Classifier)
- **Visualization:** `seaborn`, `matplotlib`
- **Environment:** Jupyter Notebook

## 💡 Key Engineering Highlights

1. **Target Leakage Prevention:** Strictly removed post-race variables (Status, Finishing Position) prior to training to simulate true pre-race prediction.
2. **Chronological Validation:** Implemented a non-shuffled `train_test_split` to ensure the model trains on historical races and tests on chronological future races, avoiding data time-travel.
3. **Domain Feature Engineering:** Engineered an `is_top_team` feature using One-Hot Encoding to account for the massive disparity in car pace between leading constructors and backmarkers.

## 📊 Quick Results & Insights

- **Algorithm Choice:** A Random Forest Classifier with a limited `max_depth` to capture non-linear relationships (e.g., P1 vs P2 matters more than P19 vs P20) while preventing overfitting.
- **Key Insight:** The Exploratory Data Analysis (EDA) revealed a massive drop-off in point probability for drivers starting outside the Top 8, heavily skewed by the specific circuit's overtaking difficulty.
- **Performance:** Accuracy: 0.5500, Precision: 0.5455, Recall: 0.6000, F1-Score: 0.5714

## 🚀 How to Run Locally

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the `model.ipynb` notebook.
   _(Note: The first run will take a moment as the FastF1 cache downloads historical telemetry/timing data.)_
