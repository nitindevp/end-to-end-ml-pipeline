# End-to-End ML Pipeline: House Price Prediction

In this project I built a complete machine learning pipeline, from loading the data to evaluating and saving the model, to predict house prices.

## Dataset
I used `Housing.csv`: 545 houses, 12 input features (area, bedrooms, bathrooms, stories, parking and yes/no features like air conditioning) and the target `price`.
There are no missing values and no duplicate rows.

## Tools used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib and Jupyter Notebook.

## What the pipeline does
1. Loads and checks the data (missing values, duplicates, price distribution, correlations).
2. Splits it into 80% training (436 houses) and 20% test (109 houses) with a fixed seed.
3. Preprocesses with a scikit-learn Pipeline and ColumnTransformer: numeric columns are filled and scaled, text columns are filled and converted to 0/1 columns.
4. Trains 5 models: Linear Regression, Ridge, Decision Tree, Random Forest and Gradient Boosting, and compares them with a baseline that always guesses the average price.
5. Evaluates them with RMSE, MAE and R2, using 5-fold cross-validation and the test set.
6. Picks the final model by cross-validation, checks feature importance, and tries tuning.
7. Saves the full pipeline with joblib and checks that the reloaded model gives identical predictions.

## Results
| Model | CV RMSE | Test RMSE | Test R2 |
|---|---|---|---|
| Baseline (average) | - | 2,268,298 | -0.018 |
| Linear Regression | 1,023,579 | 1,324,507 | 0.653 |
| Ridge | 1,021,738 | 1,333,155 | 0.648 |
| Decision Tree | 1,363,109 | 1,616,965 | 0.483 |
| Random Forest | 1,085,010 | 1,398,239 | 0.613 |
| Gradient Boosting | 1,084,589 | 1,299,730 | 0.666 |

## What I found
- Ridge was the best model by cross-validation, so I chose it as the final model. Gradient Boosting was best on the test set, but the gap is small (about 2.5%) and the test set has only 109 houses.
- Simple linear models matched the forests, and the Decision Tree was the weakest.
- Area, bathrooms, stories and air conditioning were the most important features.
- Tuning did not change the winner.

## Limitations
- Small dataset (545 houses), so small differences between models are not reliable.
- R2 is about 0.65, so a third of the variation in price is unexplained. A typical error is around 1.3 million.
- Expensive houses are under-predicted.
- Feature importance shows what the model uses, not what causes prices to change.

## How to run
Install the libraries and open the notebook:

    pip install -r requirements.txt
    jupyter notebook end_to_end_ml_pipeline.ipynb

## Using the saved model
The saved pipeline takes a DataFrame with the 12 original columns and does all preprocessing itself:

    import joblib
    model = joblib.load("housing_price_model.joblib")
    model.predict(new_house)

The saved model needs the same library versions as in requirements.txt.
