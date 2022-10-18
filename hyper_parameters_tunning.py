import time
import warnings

import optuna
from optuna.trial import Trial

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score

from main import preprocessing, import_data, EXPERIMENT_CONFIG

warnings.filterwarnings("ignore")
TREE_MODEL = 'xgboost'


def load_data():
    # 导入数据集
    wine = import_data(f'./dataset/winequality-{EXPERIMENT_CONFIG["dataset"]}.csv')

    # 预处理
    processed_wine, X_train, X_test, Y_train, Y_test = preprocessing(wine)
    return X_train, X_test, Y_train, Y_test


# def lightgbm_objective(trial: Trial) -> float:
#     # load original_data
#     X_train, X_holdout, X_test, Y_train, Y_holdout, price_encoder = load_data(ORIGINAL_DATA_DIC, TRAIN_FILE, TEST_FILE,
#                                                                               TEXT_FEA_DIC)
#     max_depth = trial.suggest_int("max_depth", 3, 5)
#     num_leaves = trial.suggest_int("num_leaves", 5, 2 ** max_depth - 1)
#     LightGBM_params = {
#         "n_estimators": 1000,
#         "metric": "rmse",
#         'early_stopping_rounds': 20,
#         "learning_rate": 0.1,
#
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
#         "max_depth": max_depth,
#         "num_leaves": num_leaves,
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
#         "subsample": trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]),
#         "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
#     }
#
#     cat_col = CATEGORY_COLUMN + ['k_means_fea']
#     X_train[cat_col] = X_train[cat_col].astype('category')
#     X_holdout[cat_col] = X_holdout[cat_col].astype('category')
#     X_test[cat_col] = X_test[cat_col].astype('category')
#
#     model = LightgbmModel(MODEL_DIC, price_encoder)
#     param = LightGBM_params
#     pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
#
#     model.train_model(X_train, X_holdout, Y_train, Y_holdout, param, [pruning_callback])
#     return model.eval_models(X_holdout, Y_holdout)


def xgboost_objective(trial: Trial, ) -> float:
    # load original_data

    X_train, X_test, Y_train, Y_test = load_data()

    XGBoost_params = {"verbosity": 0,
                      'early_stopping_rounds': 20,
                      'eval_metric': "rmse",
                      'num_boost_round': 1000,
                      # defines booster, gblinear for linear functions.
                      "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                      # L2 regularization weight.
                      "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                      # L1 regularization weight.
                      "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                      # sampling ratio for training data.
                      "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                      # sampling according to each tree.
                      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                      }
    if XGBoost_params["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        XGBoost_params["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        XGBoost_params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        XGBoost_params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        XGBoost_params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        XGBoost_params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if XGBoost_params["booster"] == "dart":
        XGBoost_params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        XGBoost_params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        XGBoost_params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        XGBoost_params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    param = XGBoost_params
    model = XGBClassifier(**param)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-mlogloss")

    model.fit(X_train, Y_train,
              # eval_set=[(X_train, Y_train)],
              # callbacks=[pruning_callback],
    )

    Y_pred = model.predict(X_test)

    return f1_score(Y_test, Y_pred,average='weighted')


# def catboost_objective(trial: Trial) -> float:
#     # load original_data
#     X_train, X_holdout, X_test, Y_train, Y_holdout, price_encoder = load_data(ORIGINAL_DATA_DIC, TRAIN_FILE, TEST_FILE,
#                                                                               TEXT_FEA_DIC)
#
#     CatBoost_params = {
#         'od_type': 'Iter',
#         'od_wait': 20,
#         'loss_function': 'RMSE',
#         'eval_metric': 'RMSE',
#         'iterations': 700,
#
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
#         "depth": trial.suggest_int("depth", 1, 12),
#         "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
#         "bootstrap_type": trial.suggest_categorical(
#             "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
#         ),
#     }
#     if CatBoost_params["bootstrap_type"] == "Bayesian":
#         CatBoost_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
#     elif CatBoost_params["bootstrap_type"] == "Bernoulli":
#         CatBoost_params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
#
#     cat_col = CATEGORY_COLUMN + ['k_means_fea']
#     X_train[cat_col] = X_train[cat_col].astype(str)
#     X_holdout[cat_col] = X_holdout[cat_col].astype(str)
#     X_test[cat_col] = X_test[cat_col].astype(str)
#
#     param = CatBoost_params
#     model = CatBoostModel(MODEL_DIC, price_encoder)
#
#     model.train_model(X_train, X_holdout, Y_train, Y_holdout, param)
#     return model.eval_models(X_holdout, Y_holdout)


if __name__ == "__main__":
    time_str = f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}'
    study = optuna.create_study(
        pruner=optuna.pruners.HyperbandPruner(),
        direction="minimize",
        storage='sqlite:///db.sqlite3',
        study_name=f'ST5188_{TREE_MODEL}_{time_str}'
    )
    # if TREE_MODEL == 'lightgbm':
    #     objective = lightgbm_objective
    # elif TREE_MODEL == 'catboost':
    #     objective = catboost_objective
    # else:
    #     objective = xgboost_objective
    objective = xgboost_objective
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  RMSE: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
