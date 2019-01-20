from sklearn.ensemble import RandomForestRegressor
import reader
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima_model import ARIMA
from utils import eval_res


def construct_model_params():
    model_params={
                    'rf': {
                        "model": RandomForestRegressor(random_state=0),
                        "params": {
                            'model' + "__max_depth": [5, 10, 20],
                            'model' + "__n_estimators": [10, 30, 50],
                        }
                    },
                    'xgb': {
                        "model": XGBRegressor(),
                        "params": {
                            "model" + "__max_depth": [3, 5],
                            "model" + "__n_estimators": [3, 10],
                        }
                    }
                  }
    return model_params


def construct_model(alg="rf"):
    model_params = construct_model_params()
    model = {}
    for key, value in model_params.items():
        if key == alg:
            model = model_params[key]

    if len(model) == 0:
        raise Exception(f"{alg} model not implement")
    return model


def grid_search(alg, train_features, train_labels):
    model_params = construct_model(alg)
    step2 = ('model', model_params['model'])
    pipeline = Pipeline([step2])

    search = GridSearchCV(pipeline, param_grid=model_params['params'], cv=TimeSeriesSplit(n_splits=3),
                          n_jobs=3, scoring='neg_mean_squared_error')
    search.fit(train_features, train_labels)
    best_estimators = search.best_estimator_

    return best_estimators


def train_test(alg):
    dataset = reader.Dataset("tweets")
    train_feature, train_rate = dataset.rf_feature_union("train")
    test_feature, test_rate = dataset.rf_feature_union("test")
    estimator = grid_search(alg, train_feature, train_rate)
    pred_rate = estimator.predict(test_feature)
    print(pred_rate, test_rate)
    print(alg, eval_res(test_rate, pred_rate))


def arima():
    dataset = reader.Dataset("tweets")
    history, test = dataset.series_feature_union()
    predictions = []
    all_test = []
    i = 1
    for one_stock_his, one_stock_test in zip(history, test):
        for t in one_stock_test:
            model = ARIMA(one_stock_his, order=(10, 0, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            all_test.append(t)
            # print(one_stock_his)
            one_stock_his += [t]
        print(i+1, predictions, all_test)
    eval_res(all_test, predictions)


if __name__ == "__main__":
    train_test('rf')
    # arima()