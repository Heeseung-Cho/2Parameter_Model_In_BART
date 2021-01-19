###Import libraries
#Basic
import numpy as np
import pandas as pd
import seaborn as sns    
import matplotlib.pyplot as plt
#AutoML
from pycaret import classification
from pycaret import regression
#Skicit-Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,cross_validate,cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,mean_squared_error
#LightGBM
import lightgbm as lgb


def FullFoldLGBC(Input,Label, Fold = 'KFold', nfolds = 10, optimize = 'AUC', iteration = 2000, threshold = 0.5, random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds (Default: 10)
    iteration: default = 2000
    threshold: default = 0.5
    optimize: Optimizing target in Pycaret (Default: 'AUC')
    '''
    #Define Model
    X = Input
    y = Label
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Stratified':
        folds = StratifiedKFold(n_splits=nfolds, shuffle = True, random_state=random_state)
    elif Fold == 'Shuffle':
        folds = ShuffleSplit(n_splits=nfolds, test_size = 0.25, random_state=random_state)
    elif Fold == 'LeaveOne':
        folds = LeaveOneOut()
        nfolds = len(X_train)
        
    #Setup tuned model from pycaret
    setup_data = pd.concat([X,y], axis = 1)
    clf = classification.setup(data = setup_data
                           , target = y.name
                           , silent = True
                           , numeric_features = X.columns.drop('Sex')                           
                           , session_id = random_state)
    lightgbm = classification.create_model('lightgbm')
    LGB_param_grid = {'application': ['binary'],
     'bagging_fraction': [0.3,0.4,0.5],
     'bagging_freq': [5,10,20],
     'boosting': ['gbdt'],
     'feature_fraction': [0.1,0.2,0.3,0.4,0.5],
     'is_unbalance': ['False'],
     'learning_rate': [0.001,0.01,0.05],
     'metric': ['auc'],
     'min_data': [1],
     'min_hess': [0],
     'objective': ['binary']}
    tuned_lgb = classification.tune_model(lightgbm, n_iter = 500, optimize = optimize, custom_grid= LGB_param_grid)
    final_model_lgb = classification.finalize_model(tuned_lgb)
    parameters = final_model_lgb.get_params()    
    
    print(parameters)
    fold_pred = np.zeros(len(X))
    lgb_preds = np.zeros(len(X))
    feature_importance_df = pd.DataFrame()
    auc_score = []
    valid_score = []
    #
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values,y.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx]) #categorical_feature=categorical_feats
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx]) #categorical_feature=categorical_feats

        iteration = iteration
        lgb_m = lgb.train(parameters, trn_data, iteration, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
        fold_pred[val_idx] = lgb_m.predict(X.iloc[val_idx], num_iteration=lgb_m.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = lgb_m.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        lgb_preds += lgb_m.predict(X, num_iteration=lgb_m.best_iteration) / (nfolds)
        
        auc_score.append(lgb_m.best_score['training']['auc'])
        valid_score.append(lgb_m.best_score['valid_1']['auc'])
    print("Best CV score: {:<8.5f}".format(np.amax(auc_score)))
    print("Average CV score:{:<8.5f}".format(np.mean(auc_score)))
    print("Best Valid1 score: {:<8.5f}".format(np.amax(valid_score)))
    print("Average Valid1 score:{:<8.5f}".format(np.mean(valid_score)))
    
    #Scaling
    scaler=MinMaxScaler()
    #prob_list = lgb_m.predict(X, num_iteration=lgb_m.best_iteration)
    prob_list_scaled = scaler.fit_transform(lgb_preds.reshape(-1,1))
    pd.options.display.float_format = '{:.2f}'.format
    df = pd.DataFrame(prob_list_scaled,columns=['scaled'], index = X.index)
    df['Predict'] = df['scaled']> threshold
    df['T/F'] = y.values
    
    #Plot Confusion Matrix
    conf_mat = confusion_matrix(df['T/F'], df['Predict'])

    cm_matrix = pd.DataFrame(data=conf_mat, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    
    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = feature_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}        
    order = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = feature_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('LightGBM over {} Folds'.format(nfolds))
    plt.tight_layout()
    
    return [final_model_lgb,pd.concat([X,df],axis = 1)]
	
def FullFoldLGBR(Input,Label, Fold = 'KFold', nfolds = 10, optimize = 'R2', iteration = 2000, threshold = 0.5, random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds (Default: 10)
    iteration: default = 2000
    optimize: Optimizing target in Pycaret (Default: 'R2')
    '''
    X = Input
    y = Label
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Shuffle':
        folds = ShuffleSplit(n_splits=nfolds, test_size = 0.25, random_state=random_state)

    #GridSearch
    setup_data = pd.concat([X,y], axis = 1)
    clf = regression.setup(data = setup_data
                           , target = y.name
                           , silent = True
                           , numeric_features = X.columns.drop('Sex')
                           , session_id = random_state)
    lightgbm = regression.create_model('lightgbm')

    LGB_param_grid = {
     'bagging_fraction': [0.3,0.4,0.5],
     'bagging_freq': [5,10,20],
     'boosting': ['gbdt'],
     'feature_fraction': [0.1,0.2,0.3,0.4,0.5],
     'is_unbalance': ['False'],
     'learning_rate': [0.001,0.01,0.05],
     'min_data': [1],
     'min_hess': [0]
    }
    tuned_lgb = regression.tune_model(lightgbm, n_iter = 500, optimize = optimize, custom_grid= LGB_param_grid)
    final_model_lgb = regression.finalize_model(tuned_lgb)
    parameters = final_model_lgb.get_params()    
    
    #Cross Validation
    fold_pred = np.zeros(len(X))
    lgb_preds = np.zeros(len(X))
    feature_importance_df = pd.DataFrame()
    auc_score = []
    valid_score = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values,y.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx]) #categorical_feature=categorical_feats
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx]) #categorical_feature=categorical_feats

        iteration = iteration
        lgb_m = lgb.train(parameters, trn_data,
                           num_boost_round = iteration,
                           valid_sets = [trn_data, val_data],
                           valid_names = ['train', 'valid'],
                           verbose_eval=200)
        fold_pred[val_idx] = lgb_m.predict(X.iloc[val_idx], num_iteration=lgb_m.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = lgb_m.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        lgb_preds += lgb_m.predict(X, num_iteration=lgb_m.best_iteration) / (nfolds)

        valid_score.append(mean_squared_error(y.iloc[val_idx],fold_pred[val_idx]))
    print("Best Valid score: {}".format(np.amin(valid_score)))
    print("Average Valid score:{}".format(np.mean(valid_score)))
    
    print('Total accuracy score: {}'.format(mean_squared_error(y, lgb_preds)))

    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = feature_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}            
    order = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = feature_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('LightGBM over {} Folds'.format(nfolds))
    plt.tight_layout()

    return [final_model_lgb,y,lgb_preds]
    
### Prediction - LightGBM Classification
def PredCVLGBC(Input,Label, Fold = 'KFold', nfolds = 10, optimize = 'AUC', iteration = 2000, threshold = 0.5, random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds (Default: 10)
    iteration: default = 2000
    threshold: default = 0.5
    optimize: Optimizing target in Pycaret (Default: 'AUC')
    '''
    #Define Model
    X = Input
    y = Label
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=random_state)
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Stratified':
        folds = StratifiedKFold(n_splits=nfolds, shuffle = True, random_state=random_state)
    elif Fold == 'Shuffle':
        folds = ShuffleSplit(n_splits=nfolds, test_size = 0.25, random_state=random_state)
    elif Fold == 'LeaveOne':
        folds = LeaveOneOut()
        nfolds = len(X_train)
        
    #Setup tuned model from pycaret
    setup_data = pd.concat([X_train,y_train], axis = 1)
    clf = classification.setup(data = setup_data
                           , target = y_train.name
                           , silent = True
                           , numeric_features = X_train.columns.drop('Sex')                           
                           , session_id = random_state)
    lightgbm = classification.create_model('lightgbm')
    LGB_param_grid = {'application': ['binary'],
     'bagging_fraction': [0.3,0.4,0.5],
     'bagging_freq': [5,10,20],
     'boosting': ['gbdt'],
     'feature_fraction': [0.1,0.2,0.3,0.4,0.5],
     'is_unbalance': ['False'],
     'learning_rate': [0.001,0.01,0.05],
     'metric': ['auc'],
     'min_data': [1],
     'min_hess': [0],
     'objective': ['binary']}
    tuned_lgb = classification.tune_model(lightgbm, n_iter = 500, optimize = optimize, custom_grid= LGB_param_grid)
    final_model_lgb = classification.finalize_model(tuned_lgb)
    parameters = final_model_lgb.get_params()    
    
    fold_pred = np.zeros(len(X_train))
    lgb_preds = np.zeros(len(X_test))
    feature_importance_df = pd.DataFrame()
    auc_score = []
    valid_score = []
    #
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx]) #categorical_feature=categorical_feats
        val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx]) #categorical_feature=categorical_feats

        iteration = iteration
        lgb_m = lgb.train(parameters, trn_data, iteration, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
        fold_pred[val_idx] = lgb_m.predict(X.iloc[val_idx], num_iteration=lgb_m.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train.columns
        fold_importance_df["importance"] = lgb_m.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        lgb_preds += lgb_m.predict(X_test, num_iteration=lgb_m.best_iteration) / (nfolds)
        
        auc_score.append(lgb_m.best_score['training']['auc'])
        valid_score.append(lgb_m.best_score['valid_1']['auc'])
    print("Best CV score: {:<8.5f}".format(np.amax(auc_score)))
    print("Average CV score:{:<8.5f}".format(np.mean(auc_score)))
    print("Best Valid1 score: {:<8.5f}".format(np.amax(valid_score)))
    print("Average Valid1 score:{:<8.5f}".format(np.mean(valid_score)))
    
    #Scaling
    scaler=MinMaxScaler()
    #prob_list = lgb_m.predict(X, num_iteration=lgb_m.best_iteration)
    prob_list_scaled = scaler.fit_transform(lgb_preds.reshape(-1,1))
    pd.options.display.float_format = '{:.2f}'.format
    df = pd.DataFrame(prob_list_scaled,columns=['scaled'], index = X_test.index)
    df['Predict'] = df['scaled']> threshold
    df['T/F'] = y_test.values
    
    #Plot Confusion Matrix
    conf_mat = confusion_matrix(df['T/F'], df['Predict'])

    cm_matrix = pd.DataFrame(data=conf_mat, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    
    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = feature_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}        
    order = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = feature_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('LightGBM over {} Folds'.format(nfolds))
    plt.tight_layout()
    
    return [final_model_lgb,pd.concat([X_test,df],axis = 1)]
    
### Prediction - LightGBM Regression
def PredCVLGBR(Input,Label, split = 0.3, Fold = 'KFold', nfolds = 10, optimize = 'R2', iteration = 2000, threshold = 0.5, random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds (Default: 10)
    iteration: default = 2000
    optimize: Optimizing target in Pycaret (Default: 'R2')
    '''
    X = Input
    y = Label
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split, random_state=random_state)
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Shuffle':
        folds = ShuffleSplit(n_splits=nfolds, test_size = 0.25, random_state=random_state)

    #GridSearch
    setup_data = pd.concat([X_train,y_train], axis = 1)
    clf = regression.setup(data = setup_data
                           , target = y_train.name
                           , silent = True
                           , numeric_features = X_train.columns.drop('Sex')
                           , session_id = random_state)
    lightgbm = regression.create_model('lightgbm')

    LGB_param_grid = {
     'bagging_fraction': [0.3,0.4,0.5],
     'bagging_freq': [5,10,20],
     'boosting': ['gbdt'],
     'feature_fraction': [0.1,0.2,0.3,0.4,0.5],
     'is_unbalance': ['False'],
     'learning_rate': [0.001,0.01,0.05],
     'min_data': [1],
     'min_hess': [0]
    }
    tuned_lgb = regression.tune_model(lightgbm, n_iter = 500, optimize = optimize, custom_grid= LGB_param_grid)
    final_model_lgb = regression.finalize_model(tuned_lgb)
    parameters = final_model_lgb.get_params()    
    
    #Cross Validation
    fold_pred = np.zeros(len(X_train))
    lgb_preds = np.zeros(len(X_test))
    feature_importance_df = pd.DataFrame()
    auc_score = []
    valid_score = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], label=y_train.iloc[trn_idx]) #categorical_feature=categorical_feats
        val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx]) #categorical_feature=categorical_feats

        iteration = iteration
        lgb_m = lgb.train(parameters, trn_data,
                           num_boost_round = iteration,
                           valid_sets = [trn_data, val_data],
                           valid_names = ['train', 'valid'],
                           verbose_eval=100)
        fold_pred[val_idx] = lgb_m.predict(X.iloc[val_idx], num_iteration=lgb_m.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train.columns
        fold_importance_df["importance"] = lgb_m.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        lgb_preds += lgb_m.predict(X_test, num_iteration=lgb_m.best_iteration) / (nfolds)
        
        valid_score.append(mean_squared_error(y.iloc[val_idx],fold_pred[val_idx]))
        
    print("Best Valid score: {}".format(np.amin(valid_score)))
    print("Average Valid score:{}".format(np.mean(valid_score)))
    
    #Predict    
    plt.hist(y_test, label = 'True')
    plt.hist(lgb_preds, label = 'Pred', alpha = 0.5)
    plt.legend(loc='upper right')
    plt.show()
    print(f'MSE: {mean_squared_error(y_test,lgb_preds):.6}')
    print(f'R2: {r2_score(y_test,lgb_preds):.3}')
    
    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = feature_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}            
    order = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = feature_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('LightGBM over {} Folds'.format(nfolds))
    plt.tight_layout()

    return [final_model_lgb,y_test,lgb_preds]
