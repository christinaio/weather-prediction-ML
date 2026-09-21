# Rain Prediction Using Machine Learning

A supervised machine learning project for predicting whether it will rain the following day based on historical weather observations.

## Project Overview

This project develops and evaluates multiple classification models for next-day rain prediction. The workflow covers exploratory data analysis, preprocessing, model comparison, hyperparameter tuning, and final evaluation on an unseen test set.

Since rainy days represent the minority class, model performance is evaluated using precision, recall, and F1 score in addition to accuracy.

## Machine Learning Workflow

- Exploratory data analysis and missing-value assessment
- Train/validation split with stratification
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standardization of numerical features
- One-hot encoding of categorical features
- Comparison of seven classification algorithms
- Hyperparameter tuning with GridSearchCV
- 5-fold stratified cross-validation
- Final evaluation on an unseen test set

## Models Evaluated

- Gaussian Naive Bayes
- K-Nearest Neighbors
- Logistic Regression
- Multi-Layer Perceptron
- Support Vector Classifier
- Decision Tree
- Random Forest

## Model Selection

Among the baseline models, Logistic Regression achieved the highest F1 score on the validation set.

Hyperparameter tuning was performed using GridSearchCV with 5-fold stratified cross-validation, optimizing for F1 score. The selected Logistic Regression model used:

- `C = 0.1`
- `class_weight = "balanced"`

## Results

On the unseen test set, the tuned Logistic Regression model achieved:

| Metric | Baseline | Tuned |
|---|---:|---:|
| F1 Score | 0.628 | **0.656** |
| Recall | 0.551 | **0.795** |
| Precision | 0.732 | 0.558 |

The tuned model substantially improved recall, correctly identifying 847 of 1,066 rainy days. This improvement came with lower precision, illustrating the trade-off between identifying more rainy days and generating additional false-positive predictions.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Repository Structure

    weather_prediction_ML/
    │
    ├── data/
    │   ├── train-val.csv
    │   └── test.csv
    │
    ├── weather_prediction.ipynb
    ├── README.md
    ├── .gitignore
    └── LICENSE

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Open `weather_prediction.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells sequentially.

## Key Takeaway

This project demonstrates an end-to-end supervised machine learning workflow, with particular emphasis on preventing data leakage, handling class imbalance, comparing multiple classifiers, and evaluating model performance beyond accuracy.
