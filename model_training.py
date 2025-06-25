# model_training.py
import pandas as pd
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from assets_data_prep import prepare_data

df = pd.read_csv("train.csv")
df_cleaned = prepare_data(df, st="train", with_price=True)

X = df_cleaned.drop("price", axis=1)
y = df_cleaned["price"]

param_grid = {
    'alpha': [0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]
}
model = ElasticNet()
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
joblib.dump(best_model, "trained_model.pkl")
