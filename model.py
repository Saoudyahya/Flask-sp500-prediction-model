import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def calculate_metrics(predictions):
    try:
        metrics = {}
        metrics['Precision'] = precision_score(predictions["Target"], predictions["Prediction"])
        metrics['Recall'] = recall_score(predictions["Target"], predictions["Prediction"])
        metrics['F1-Score'] = f1_score(predictions["Target"], predictions["Prediction"])
        metrics['Accuracy'] = accuracy_score(predictions["Target"], predictions["Prediction"])
        metrics['Confusion Matrix'] = confusion_matrix(predictions["Target"], predictions["Prediction"])
        return metrics
    except Exception as e:
        logger.error(f"An error occurred in calculate_metrics function: {e}")
        traceback.print_exc()
def predict(train, test, predictors, model):
    try:
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:, 1]
        preds[preds >= 0.6] = 1
        preds[preds < 0.6] = 0
        preds = pd.Series(preds, index=test.index, name="Prediction")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined
    except Exception as e:
        logger.error(f"An error occurred in predict function: {e}")
        traceback.print_exc()

def backtest(data, model, predictors, start=2500, step=250):
    try:
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            prediction = predict(train, test, predictors, model)
            if prediction is not None:
                all_predictions.append(prediction)
        return pd.concat(all_predictions)
    except Exception as e:
        logger.error(f"An error occurred in backtest function: {e}")
        traceback.print_exc()

def create_predictors(data, horizons):
    try:
        new_predictors = []
        for horizon in horizons:
            rolling_average = data.rolling(horizon).mean()
            ratio_column = f"close_ratio_{horizon}"
            data[ratio_column] = data["Close"] / rolling_average["Close"]
            trend_column = f"TREND_{horizon}"
            data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
            new_predictors += [ratio_column, trend_column]
        return new_predictors
    except Exception as e:
        logger.error(f"An error occurred in create_predictors function: {e}")
        traceback.print_exc()

def train_pipeline(data, model, base_predictors, horizons):
    try:
        new_predictors = create_predictors(data, horizons)
        all_predictors = base_predictors + new_predictors
        data = data.dropna()
        train_data = data.iloc[:-100]
        test_data = data.iloc[-100:]
        predictions = backtest(train_data, model, all_predictors)
        if predictions is not None:
            metrics = calculate_metrics(predictions)
            logger.info("Performance Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
            return metrics['Precision']
    except Exception as e:
        logger.error(f"An error occurred in train_pipeline function: {e}")
        traceback.print_exc()

def main():
    try:
        # Data preparation
        sp500 = yf.Ticker("^GSPC").history(period="max")
        del sp500["Dividends"]
        del sp500["Stock Splits"]
        sp500["Tomorrow"] = sp500["Close"].shift(-1)
        sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
        sp500_after_1990_04_03 = sp500.loc["1990-04-03":]

        # Model initialization
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

        # Base predictors
        base_predictors = ["Close", "Volume", "Open", "High", "Low"]

        # Horizons for feature creation
        horizons = [2, 5, 60, 250, 1000]

        # Train pipeline
        precision = train_pipeline(sp500_after_1990_04_03, model, base_predictors, horizons)
        return precision
    except Exception as e:
        logger.error(f"An error occurred in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
