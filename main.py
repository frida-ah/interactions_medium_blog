from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import PolynomialFeatures
import re
from typing import List
import shap
import lightgbm

import prepare_data


def split_train_test(pdf, numerical_cols, categorical_cols, target_col, date_col):
    start_train = datetime(2018, 1, 1)
    end_train = datetime(2020, 11, 30)
    start_test = datetime(2020, 6, 1)
    end_test = datetime(2020, 9, 6)

    pdf_filtered = pdf[numerical_cols + categorical_cols + [target_col, date_col]]
    train_pdf = pdf_filtered.loc[(pdf_filtered[date_col] >= start_train) & (pdf_filtered[date_col] <= end_train)]
    test_pdf = pdf_filtered.loc[(pdf_filtered[date_col] > start_test) & (pdf_filtered[date_col] < end_test)]

    train_pdf = train_pdf.set_index(date_col)
    test_pdf = test_pdf.set_index(date_col)
    return train_pdf, test_pdf


def select_features_target(train_pdf, test_pdf, numerical_cols, categorical_cols, target_col):
    x_train = train_pdf[numerical_cols + categorical_cols]
    x_test = test_pdf[numerical_cols + categorical_cols]

    y_train = train_pdf[target_col]
    y_test = test_pdf[target_col]
    return x_train, y_train, x_test, y_test


def get_column_type(pdf: pd.DataFrame, suffix: str) -> List:
    column_names = pdf.columns
    r = re.compile(".*" + suffix)
    filtered_list = list(filter(r.match, column_names))
    return filtered_list


def select_features(pdf):
    target_col = "searches"
    date_col = "date"
    pdf = pdf.sort_values(by=[date_col], ascending=True)

    # this ensures that the date column is not part of the feature selection
    pdf = pdf.set_index(date_col)

    y = pdf[target_col]
    X = pdf[
        [
            "total_rainfall_num",
            "avg_sunshine_num",
            "avg_temperature_num",
            "holiday_cat",
        ]
    ]
    model = OLS(y, X).fit()

    model.summary()

    # generating interaction terms
    x_interaction = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(X)

    pdf_interaction = pd.DataFrame(
        x_interaction,
        columns=[
            "total_rainfall_num",
            "avg_sunshine_num",
            "avg_temperature_num",
            "holiday_cat",
            "rainfall_sunshine_num",
            "rainfall_temperature_num",
            "rainfall_holiday_num",
            "sunshine_temperature_num",
            "sunshine_holiday_num",
            "temperature_holiday_num",
        ],
    )
    pdf_dates = pdf.copy().reset_index(drop=False)[[date_col]]

    pdf_interaction_date = pd.concat([pdf_interaction, pdf_dates], axis=1).set_index(date_col)

    interaction_model = OLS(y, pdf_interaction_date).fit()
    # print("pdf_interaction_date:", pdf_interaction_date.columns)
    # print("y:", y.head())
    pdf_significant_features = interaction_model.pvalues[interaction_model.pvalues < 0.05]

    list_significant_features = list(pdf_significant_features.index.values)
    pdf_model = pdf_interaction[list_significant_features]
    pdf_searches = pdf.copy().reset_index(drop=False)[[date_col, target_col]]
    pdf_model = pd.concat([pdf_model, pdf_searches], axis=1)
    return pdf_model


def prepare_train_test_datasets(pdf, include_interaction):
    target_col = "searches"
    date_col = "date"

    if include_interaction:
        pdf_model = select_features(pdf)
    else:
        pdf_model = pdf.copy()

    numerical_cols = get_column_type(pdf_model, "num")
    categorical_cols = get_column_type(pdf_model, "cat")
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    pdf_model.reset_index(inplace=True)
    # print("pdf_model:", pdf_model.columns)
    train_pdf, test_pdf = split_train_test(pdf_model, numerical_cols, categorical_cols, target_col, date_col)
    x_train, y_train, x_test, y_test = select_features_target(train_pdf, test_pdf, numerical_cols, categorical_cols, target_col)
    return (x_train, y_train, x_test, y_test)


def fit_linear_regression(pdf, include_interaction=True):
    (x_train, y_train, x_test, y_test) = prepare_train_test_datasets(pdf, include_interaction=include_interaction)
    model = LinearRegression()
    fitted_model = model.fit(x_train, y_train)
    y_pred = pd.Series(fitted_model.predict(x_test))
    regression_shap_values = decompose_shap_values(fitted_model, x_train, x_test, shap.LinearExplainer)
    return y_test, y_pred, regression_shap_values


def fit_lightgbm(pdf, include_interaction=False):
    (x_train, y_train, x_test, y_test) = prepare_train_test_datasets(pdf, include_interaction=include_interaction)
    params = get_gbm_parameters()
    model = lightgbm.sklearn.LGBMRegressor(**params)
    fitted_model = model.fit(x_train, y_train)
    y_pred = pd.Series(fitted_model.predict(x_test))
    regression_shap_values = decompose_shap_values(fitted_model, x_train, x_test, shap.TreeExplainer)
    return y_test, y_pred, regression_shap_values


def get_gbm_parameters():
    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "rmse",
        "metric": ["l2", "rmse"],
        "learning_rate": 0.005,
        "num_leaves": 128,
        "max_bin": 512,
    }
    return params


def calc_shap_values(fitted_model, X, shap_explainer):
    explainer = shap_explainer(fitted_model, X)
    shapley_values = explainer.shap_values(X)
    shapley_values_df = pd.DataFrame(shapley_values, index=X.index, columns=X.columns)
    return shapley_values_df


def decompose_shap_values(fitted_model, x_train, x_test, shap_explainer):
    shapley_values_train = calc_shap_values(fitted_model, x_train, shap_explainer)
    shapley_values_test = calc_shap_values(fitted_model, x_test, shap_explainer)
    decomposed_shapley_values = pd.concat([shapley_values_train, shapley_values_test])
    return decomposed_shapley_values


def join_preds_to_actuals(y_pred, y_test, predictions_column="preds", observed_column="actuals"):
    # returns pd DataFrame with predictions and actuals
    eval_table = pd.concat([y_pred.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    eval_table.columns = [predictions_column, observed_column]
    return eval_table


def calculate_rmse(y_pred, y_test, predictions_column="preds", observed_column="actuals"):
    eval_table = join_preds_to_actuals(y_pred, y_test)
    mse_score = round(
        mean_squared_error(eval_table[observed_column], eval_table[predictions_column]),
        2,
    )
    rmse_score = round(np.sqrt(mse_score), 3)
    print(f"RMSE: {rmse_score}")


def calculate_variance(y_pred, y_test):
    variance_score = explained_variance_score(y_test, y_pred)
    print(f"Variance score: {variance_score}")


if __name__ == "__main__":
    pdf = prepare_data.prepare_input_data()

    print("Scores for linear regression without interactions:")
    y_test, y_pred, regression_shap_values = fit_linear_regression(pdf=pdf, include_interaction=False)
    calculate_rmse(y_pred, y_test)
    # print(regression_shap_values)

    print("Scores for linear regression with interactions:")
    y_test_interactions, y_pred_interactions, interactions_shap_values = fit_linear_regression(pdf=pdf, include_interaction=True)
    calculate_rmse(y_pred_interactions, y_test_interactions)
    # print(interactions_shap_values)

    print("Scores for LightGBM model:")
    y_test_gbm, y_pred_gbm, gbm_shap_values = fit_lightgbm(pdf=pdf)
    calculate_rmse(y_pred_gbm, y_test_gbm)
    # print(gbm_shap_values)
