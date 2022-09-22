import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
import plotly.graph_objects as go
import warnings
from scipy import stats
import numpy as np
from imblearn.over_sampling import SMOTE


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


def import_data(dataset_path: str, task:str='classification') -> pd.DataFrame:
    if task not in ['classification','regression','coarse_grain']:
        print('Invalid task')
        exit(-1)

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


def EDA(wine:pd.DataFrame) -> None:
    # 打印数据集基本信息
    print(wine.info())
    print(wine.describe().T)

    X = wine.drop(['quality'],axis=1)
    Y = wine[['quality']]

    # pairplot
    sns.pairplot(wine, height=5, kind='scatter', diag_kind='kde')
    plt.show()

    # quality分布圆形图
    fig = go.Figure(
        data=[go.Pie(labels=Y['quality'].value_counts().index, values=Y['quality'].value_counts(), hole=.3)])
    fig.update_layout(legend_title_text='Quality')
    fig.show()

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
    correlation = wine.corr()
    cols = correlation.nlargest(10, 'quality')['quality'].index
    cm = np.corrcoef(wine[cols].values.T)
    _, _ = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, vmax=.8, linewidths=0.01, square=True, annot=True, cmap='viridis',
                linecolor="white", xticklabels=cols.values, annot_kws={'size': 12}, yticklabels=cols.values)
    plt.show()

    # quality在feature中的分布
    fig, ax = plt.subplots(3, 4, figsize=(24, 30))
    k = 0
    columns = list(X.columns)
    while k < len(columns):
        for i in range(3):
            for j in range(4):
                if k < len(columns):
                    sns.boxplot(Y['quality'], X[columns[k]], ax=ax[i][j], palette='pastel')
                    k += 1
    plt.show()


def preprocessing(wine):
    # TODO outlier去除

    # 重采样
    oversample = SMOTE()
    resample_features, resample_labels = oversample.fit_resample(wine.drop(["quality"], axis=1), wine["quality"])

    # TODO 特征选取

    # scaler
    scaler = StandardScaler()
    scaled_feature = pd.DataFrame(scaler.fit_transform(resample_features), columns=resample_features.columns)
    processed_wine = pd.concat([scaled_feature,resample_labels],axis=1)

    # 切分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(scaled_feature, resample_labels, test_size=0.1, random_state=42)

    return processed_wine, X_train, X_test, Y_train, Y_test


def training(X_train, Y_train):
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train)

    return xgb


def evaluation(model, X_test, Y_test):
    predict = model.predict(X_test)
    print(classification_report(Y_test, predict))


if __name__ == '__main__':
    # 初始化各种设计
    init_config()

    # 导入数据集
    wine = import_data('./dataset/winequality-red.csv','classification')

    # 预处理
    processed_wine, X_train, X_test, Y_train, Y_test = preprocessing(wine)

    # # 可视化
    EDA(wine)
    EDA(processed_wine)

    # 训练
    model = training(X_train, Y_train)

    # 评估
    evaluation(model, X_test, Y_test)
