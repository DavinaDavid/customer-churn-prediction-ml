# Customer Churn Prediction with Machine Learning

This is my first assignment for Marketing Analytics. The task: predict whether a customer will churn (stop using the service) from their demographics, spending, engagement and marketing history. I compared three models, Logistic Regression, Decision Tree and Random Forest, on two separate churn datasets.

The slides are in `Churn_Prediction_using_ML_Models (slides).pdf`.

## The data

There are two datasets, with 40,000 and 50,000 customers. Each has 13 features and a binary `churn` target (1 = churned, 0 = stayed).

| Group | Features |
|---|---|
| Demographics | age, gender, region |
| Financial | income, monthly spend |
| Engagement | website visits, email click rate, support tickets |
| Marketing | campaign type, discount used, days since last campaign |
| Customer value | tenure (months), loyalty score |

Ten features are numeric and three are categorical (`gender`, `region`, `campaign_type`). Neither dataset has missing values. Dataset 2 has a noticeably higher share of churners than Dataset 1.

The two datasets are included as `dataset1_HW1` and `dataset2_HW1`.

## Approach

I built one scikit-learn preprocessing pipeline and used it for all three models so the comparison is fair:

- **Numeric features:** `SimpleImputer`, then `StandardScaler`
- **Categorical features:** most-frequent imputation, then `OneHotEncoder`
- Combined with `ColumnTransformer` and `Pipeline`

The imputers weren't actually needed since there's no missing data, but I left them in so the pipeline would still work on messier data.

After training, I pulled feature importances from each model (`coef_` for Logistic Regression, `feature_importances_` for the trees) and plotted the top 10. I evaluated everything with accuracy, F1 and ROC-AUC, plus ROC and precision-recall curves.

## Results

| Dataset | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| 1 | Logistic Regression | 0.815 | 0.822 | 0.899 |
| 1 | Decision Tree | 1.000 | 1.000 | 1.000 |
| 1 | Random Forest | 1.000 | 1.000 | 1.000 |
| 2 | Logistic Regression | 0.942 | 0.967 | 0.970 |
| 2 | Decision Tree | 1.000 | 1.000 | 1.000 |
| 2 | Random Forest | 1.000 | 1.000 | 1.000 |

The tree models scoring a perfect 1.00 on both datasets is not something I trust. Real churn data is noisy, so a perfect score usually means overfitting or data leakage, for example a feature that gives away the answer. I haven't tracked down the cause yet. Because of this I recommend Logistic Regression, whose curves look realistic and whose results are easier to interpret.

**Top features:** monthly spend and days since last campaign came out on top in every model on both datasets. Tenure, email click rate and loyalty score came next. The trees leaned heavily on just one or two features (especially monthly spend on Dataset 2), while Logistic Regression spread its weight across more of them.

## Caveats

- **The perfect tree scores need investigating.** Next steps would be checking for leaking columns, looking at train vs. test scores, and using cross-validation with a limited tree depth.
- **Dataset 2 isn't necessarily "easier".** Its churn rate is higher, and with more churners in the data, accuracy and F1 look better even for a mediocre model. Comparing across datasets this way is a bit unfair.
- I used default hyperparameters for all models and didn't do any tuning.
- Accuracy alone can hide a lot on imbalanced data, so look at the precision-recall curves too.

## Running it

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy scikit-learn matplotlib
python HW1_Code.py
```

Run it from the repo folder so the script can find the two dataset files sitting next to it.

## What's in the repo

```
HW1_Code.py                                  main script (preprocessing, models, plots)
dataset1_HW1                                 customer data, 40,000 rows
dataset2_HW1                                 customer data, 50,000 rows
churn_dataset1_logistic_regression           model output for Dataset 1
churn_dataset1_decision_tree                 model output for Dataset 1
churn_dataset1_random_forest                 model output for Dataset 1
churn_dataset2_logistic_regression           model output for Dataset 2
churn_dataset2_decision_tree                 model output for Dataset 2
churn_dataset2_random_forest                 model output for Dataset 2
Churn_Prediction_using_ML_Models (slides).pdf   project presentation
```

## Author

Davina David
