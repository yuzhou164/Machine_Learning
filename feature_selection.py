import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np


def _read_data():
    all_data = []
    for parents, path, filenames in os.walk("input/rri_feature_two"):
        for filename in filenames:
            temp_rri = pd.read_csv(os.path.join(parents, filename))
            temp_rri.drop(labels=[u'Unnamed: 0'], axis=1, inplace=True)
            all_data.extend(temp_rri.values.tolist())
            name = temp_rri.columns
    return all_data, name


def get_corr_feature(df_result):
    corr = df_result.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()
    # plt.figure(figsize=(12, 12))
    # sns.heatmap(corr, vmax=1, square=True)
    cor_dict = corr['sleep_stage'].to_dict()
    del cor_dict['sleep_stage']
    # print("List the numerical features decendingly by their correlation with Sale Price:\n")
    for ele in sorted(cor_dict.items(), key=lambda x: -abs(x[1])):
        print("{0}: \t{1}".format(*ele))
    cor_top_10 = [value[0] for value in sorted(cor_dict.items(), key=lambda x: -abs(x[1]))[:10]]
    # print (cor_top_10)
    return cor_top_10


def get_rf_important_feature(X_train_std, y_train, df_result):
    model = RandomForestClassifier()
    model.fit(X_train_std, y_train)

    feature_imp = pd.DataFrame(model.feature_importances_, index=df_result.columns[:-1], columns=["importance"])
    feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(10).index
    return feat_imp_20


def remov_low_variance_features(X_train_std):
    # Find all features with more than 90% variance in values.
    threshold = 0.90
    vt = VarianceThreshold().fit(X_train_std)
    # Find feature names
    feat_var_threshold = df.columns[:-1][vt.variances_ > threshold * (1 - threshold)]
    # select the top 20
    # print(feat_var_threshold[0:10])
    # print(vt.variances_)
    # print(df.columns[:-1])
    return feat_var_threshold[0:10]


def pick_top_k_best_features(X_train_std, y_train, df):
    X_scored = SelectKBest(score_func=f_regression, k='all').fit(X_train_std, y_train)
    feature_scoring = pd.DataFrame({
        'feature': df.columns[:-1],
        'score': X_scored.scores_
    })

    feat_scored_10 = feature_scoring.sort_values('score', ascending=False).head(10)['feature'].values
    return feat_scored_10


#Select 20 features from using recursive feature elimination (RFE) with logistic regression model.
def rfe_features_selection(X_train_std, y_train, df):
    rfe = RFE(LogisticRegression(), 10)
    rfe.fit(X_train_std, y_train)

    feature_rfe_scoring = pd.DataFrame({
        'feature': df.columns[:-1],
        'score': rfe.ranking_
    })

    feat_rfe_10 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
    return feat_rfe_10


if __name__ == '__main__':
    all_data, name = _read_data()
    df_result = pd.DataFrame(all_data, columns=name)
    X_train_std = df_result.iloc[:, :-1]
    y_train = df_result.iloc[:, -1]
    feat_rfe_10 = rfe_features_selection(X_train_std, y_train, df_result)
    feat_scored_10 = pick_top_k_best_features(X_train_std, y_train, df_result)
    feat_var_threshold_10 = remov_low_variance_features
    feat_imp_10 = get_rf_important_feature(X_train_std, y_train, df_result)
    get_corr_feature = get_corr_feature(df_result)
    features = np.hstack([
        feat_rfe_10,
        feat_scored_10,
        feat_var_threshold_10,
        feat_imp_10,
        get_corr_feature
    ])
    print(features)
    print (len(features))