import warnings

import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report,mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import *
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from feature_engine.selection.smart_correlation_selection import SmartCorrelatedSelection
from feature_engine.selection.drop_psi_features import DropHighPSIFeatures
# from feature_engine.creation import MathematicalCombination, CombineWithReferenceFeature
from feature_engine.transformation import *

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor,AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 实验配置
EXPERIMENT_CONFIG = {
    'dataset': 'white',  # 数据集 必填 ['red', 'white']
    'task': 'classification',  # 任务 必填 ['classification', 'regression', 'coarse_grain']
    # 'outlier_detect': 'IForest',  # 异常点检测算法(启动了这个也必须启动outlier_process) ['ECOD', 'IForest']
    # 'outlier_process': 'drop_outliers',  # 异常点处理算法 ['impute_outliers', 'drop_outliers']
    # 'feature_creation': 'true',  # 创造额外特征 ['true']
    # 'feature_selection': 'SmartCorrelatedSelection',  # 特征选择算法 ['SmartCorrelatedSelection', 'DropHighPSIFeatures']
    'transformer': 'YeoJohnsonTransformer',  # scale算法 ['LogTransformer', 'LogCpTransformer', 'ArcsinTransformer', 'PowerTransformer', 'YeoJohnsonTransformer', 'BoxCoxTransformer']
    'sampler': 'BorderlineSMOTE', # 重采样算法 ['ADASYN', 'RandomOverSampler', 'SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'SMOTEN']
}

# 分类模型
CLF_MODELS = {
    # 'KNN': KNeighborsClassifier(3),
    # 'SVC_linear': SVC(kernel="linear"),
    # 'SVC_RBF': SVC(),
    # "Gaussian_Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    # "Decision_Tree": DecisionTreeClassifier(),
    # "Random_Forest": RandomForestClassifier(),
    # "MLP": MLPClassifier(),
    # "AdaBoost": AdaBoostClassifier(),
    # "Naive_Bayes": GaussianNB(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    # "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    # "CatBoost": CatBoostClassifier(silent=True),
}

# 回归模型
REG_MODELS = {
    # 'KNN': KNeighborsRegressor(3),
    # 'SVC_linear': SVR(kernel="linear", C=0.025),
    # 'SVC_RBF': SVR(gamma=2, C=1),
    # "Gaussian_Process": GaussianProcessRegressor(1.0 * RBF(1.0)),
    "Decision_Tree": DecisionTreeRegressor(max_depth=5),
    "Random_Forest": RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
    "MLP": MLPRegressor(alpha=1, max_iter=1000),
    "AdaBoost": AdaBoostRegressor(),
    # "Naive_Bayes": GaussianNB(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(silent=True),
}

# 初始化配置
def init_config() -> None:
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度
    pd.set_option('max_colwidth', 100)
    # 设置1000列时才换行
    pd.set_option('display.width', 1000)
    # 忽略各种警告
    warnings.filterwarnings('ignore')
    # seaborn 初始化
    sns.set()

# 导入数据集
def import_data(dataset_path: str) -> pd.DataFrame:
    task = EXPERIMENT_CONFIG['task']

    # 读csv
    wine = pd.read_csv(dataset_path, sep=';')

    if task == 'coarse_grain':
        # 粗分类
        wine['quality'] = wine['quality'].apply(lambda value: 'need_to_improve'
        if value <= 5 else 'good'
        if value <= 7 else 'outstanding')
        wine['quality'] = pd.Categorical(wine['quality'],
                                         categories=['need_to_improve', 'good', 'outstanding'])
    elif task == 'regression':
        # 回归
        wine['quality'].astype('float64')
    else:
        # 普通分类
        wine['quality'].astype('int')

    return wine

# 数据可视化分析
def EDA(wine: pd.DataFrame) -> None:
    # 打印数据集基本信息
    print(wine.info())
    print(wine.describe().T)

    X = wine.drop(['quality'], axis=1)
    Y = wine[['quality']]

    # # pairplot
    # sns.pairplot(wine, height=5, kind='scatter', diag_kind='kde')
    # plt.show()
    #
    # # quality分布圆形图
    # fig = go.Figure(
    #     data=[go.Pie(labels=Y['quality'].value_counts().index, values=Y['quality'].value_counts(), hole=.3)])
    # fig.update_layout(legend_title_text='Quality')
    # fig.show()
    #
    # feature分布图
    fig, ax = plt.subplots(X.shape[1], 3, figsize=(30, 90))
    for index, col_name in enumerate(X.columns):
        sns.distplot(X[col_name], ax=ax[index, 0], color='green')
        sns.boxplot(X[col_name], ax=ax[index, 1], color='yellow')
        stats.probplot(X[col_name], plot=ax[index, 2])
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.suptitle("Visualizing continuous columns", fontsize=50)
    fig.show()

    # correlation
    # correlation = wine.corr()
    # cols = correlation.nlargest(X.shape[1], 'quality')['quality'].index
    # cm = np.corrcoef(wine[cols].values.T)
    # _, _ = plt.subplots(figsize=(14, 12))
    # sns.heatmap(cm, vmax=.8, linewidths=0.01, square=True, annot=True, cmap='viridis',
    #             linecolor="white", xticklabels=cols.values, annot_kws={'size': 12}, yticklabels=cols.values)
    # plt.show()

    # # quality在feature中的分布
    # fig_col = int(X.shape[1] ** 0.5)
    # fig_row = fig_col
    # if X.shape[1] ** 0.5 != fig_row:
    #     fig_row += 1
    # fig, ax = plt.subplots(fig_row, fig_col, figsize=(fig_col * 8, fig_row * 8))
    # k = 0
    # columns = list(X.columns)
    # for i in range(fig_row):
    #     for j in range(fig_col):
    #         if k >= len(columns):
    #             break
    #         sns.boxplot(Y['quality'], X[columns[k]], ax=ax[i][j], palette='pastel')
    #         k += 1
    # plt.show()

# 异常处理算法-用中位数替换异常值
def impute_outliers(df, outlier_index):
    for col in df.columns:
        if col == 'quality':
            continue
        med = np.median(df[col])
        for i in outlier_index:
            df.loc[i, col] = med
    return df

# 异常处理算法-丢弃异常点数据
def drop_outliers(df, outlier_index):
    return df.drop(outlier_index)

# 异常处理
def handle_outlier(wine):
    X = wine.drop(['quality'], axis=1)
    # outlier 识别
    # clf = ECOD()
    clf = eval(EXPERIMENT_CONFIG['outlier_detect'])()
    outlier = clf.fit_predict(X)
    outlier_index = np.where(outlier == 1)[0]

    # outlier 处理
    wine = eval(EXPERIMENT_CONFIG['outlier_process'])(wine, outlier_index)
    return wine

# 特征选择
def feature_selection(wine):
    X = wine.drop(['quality'], axis=1)
    Y = wine[['quality']]

    primed_wine = pd.concat([eval(EXPERIMENT_CONFIG['feature_selection'])().fit_transform(X), Y], axis=1)

    return primed_wine

# 特征创造
def feature_creation(wine):
    X = wine.drop(['quality'], axis=1)
    Y = wine[['quality']]

    feature_creation_pipeline = Pipeline([
        ('acidity', MathematicalCombination(
            variables_to_combine=['fixed acidity', 'volatile acidity'],
            math_operations=['sum', 'mean'],
            new_variables_names=['total_acidity', 'average_acidity']
        )
         ),

        ('total_minerals', MathematicalCombination(
            variables_to_combine=['chlorides', 'sulphates'],
            math_operations=['sum', 'mean'],
            new_variables_names=['total_minerals', 'average_minearals'],
        )
         ),

        ('non_free_sulfur', CombineWithReferenceFeature(
            variables_to_combine=['total sulfur dioxide'],
            reference_variables=['free sulfur dioxide'],
            operations=['sub'],
            new_variables_names=['non_free_sulfur_dioxide'],
        )
         ),

        ('perc_free_sulfur', CombineWithReferenceFeature(
            variables_to_combine=['free sulfur dioxide'],
            reference_variables=['total sulfur dioxide'],
            operations=['div'],
            new_variables_names=['percentage_free_sulfur'],
        )
         ),

        ('perc_salt_sulfur', CombineWithReferenceFeature(
            variables_to_combine=['sulphates'],
            reference_variables=['free sulfur dioxide'],
            operations=['div'],
            new_variables_names=['percentage_salt_sulfur'],
        )
         ),
    ])
    X = feature_creation_pipeline.fit_transform(X)
    augmented_wine = pd.concat([X, Y], axis=1)

    return augmented_wine

# 数据预处理
def preprocessing(wine):
    # 处理outlier
    if (EXPERIMENT_CONFIG.get('outlier_detect')):
        wine = handle_outlier(wine)

    # 特征创造
    if (EXPERIMENT_CONFIG.get('feature_creation')):
        wine = feature_creation(wine)

    # 特征选择
    if (EXPERIMENT_CONFIG.get('feature_selection')):
        wine = feature_selection(wine)

    # 重采样
    if (EXPERIMENT_CONFIG.get('sampler')):
        sampler = eval(EXPERIMENT_CONFIG['sampler'])()
        X, Y = wine.drop(['quality'], axis=1), wine[['quality']]
        X, Y = sampler.fit_resample(X, Y)
        wine = pd.concat([X, Y], axis=1)

    # scale
    X = wine.drop(['quality'], axis=1)
    Y = wine[['quality']]
    scaler = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    if (EXPERIMENT_CONFIG.get('transformer')):
        scaler = eval(EXPERIMENT_CONFIG['transformer'])()
        X = scaler.fit_transform(X)
    wine = pd.concat([X, Y], axis=1)

    # 切分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,
                                                        random_state=66)

    return wine, X_train, X_test, Y_train, Y_test

# 模型评估
def evaluation(model, X_test, Y_test):
    predict = model.predict(X_test)
    if EXPERIMENT_CONFIG['task'] == 'regression':
        print(mean_squared_error(Y_test,predict))
    else:
        print(classification_report(Y_test, predict))


if __name__ == '__main__':
    # 初始化各种设置
    init_config()

    # 导入数据集
    wine = import_data(f'./dataset/winequality-{EXPERIMENT_CONFIG["dataset"]}.csv')

    # 预处理
    processed_wine, X_train, X_test, Y_train, Y_test = preprocessing(wine)

    # # 可视化
    # EDA(wine)
    # EDA(processed_wine)

    # 训练
    models = CLF_MODELS
    if EXPERIMENT_CONFIG['task'] == 'regression':
        models = REG_MODELS
    for name,model in models.items():
        model.fit(X_train, Y_train)

        print('-'*20 + name + '-'*20)
        # 评估
        evaluation(model, X_test, Y_test)
