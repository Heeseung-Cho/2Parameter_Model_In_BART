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
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,cross_validate,cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#shap
import shap

### Explainable - Random Forest Classifier
def FullFoldRFC(Input,Label, Fold = 'KFold', nfolds = 10, optimize = 'AUC', threshold = 0.5, random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds
    optimize: Optimizing target in Pycaret (Default: 'AUC')
    '''
    X = Input
    y = Label
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Stratified':
        folds = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state=random_state)

    #Setup tuned model from pycaret        
    setup_data = pd.concat([X,y], axis = 1)
    clf = classification.setup(data = setup_data
                           , target = y.name
                           , numeric_features = X.columns.drop('Sex')                           
                           , silent = True
                           , session_id = random_state)       
    rf = classification.create_model('rf')
    tuned_rf = classification.tune_model(rf, n_iter = 18, optimize = optimize, choose_better = True)
    final_model_rf = classification.finalize_model(tuned_rf)
    RFC_best = RandomForestClassifier(**final_model_rf.get_params())

    feature_names = ['F{}'.format(i) for i in range(X.shape[1])]
    fold_pred = np.zeros(len(X))
    rf_preds = np.zeros(len(X))
    rf_importance_df = pd.DataFrame()
    shap_values = np.zeros(shape = X.shape)
    #oob_score = []
    acc_score = []
    best_score = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values,y.values)):
        print("fold n°{}".format(fold_))
        trn_data = X.iloc[trn_idx]; trn_label = y.iloc[trn_idx]
        val_data = X.iloc[val_idx]; val_label = y.iloc[val_idx]
        
        ## Fitting
        RFC_best.fit(trn_data,trn_label)
        fold_pred[val_idx] = RFC_best.predict(val_data)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = RFC_best.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        rf_importance_df = pd.concat([rf_importance_df, fold_importance_df], axis=0)

        accuracy = accuracy_score(val_label, RFC_best.predict(val_data))
        rf_preds += RFC_best.predict_proba(X)[:,1] / (nfolds)
        acc_score.append(accuracy)    
        print(f'Mean accuracy score: {accuracy:.3}')    
        
        ## Shap        
        explainer = shap.TreeExplainer(RFC_best)
        shap_values += explainer.shap_values(X) / (nfolds)        
        
    #Plot Confusion Matrix    
    predicted = rf_preds > 0.5
    conf_mat = confusion_matrix(y, predicted)

    cm_matrix = pd.DataFrame(data=conf_mat, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = rf_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}    
    order = rf_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = rf_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('Random Forest over {} Folds'.format(nfolds))
    plt.tight_layout()
    plt.show()    
    print("Best Acc score: {:<8.5f}".format(np.amax(acc_score)))
    
    ### SHAP Analysis
    print('========================================================================')
    print("SHAP Analysis : Classifier")
    shap.summary_plot(shap_values, features=X, feature_names=X.columns)
    plt.show()    
    return [RFC_best,y,predicted, explainer, shap_values]
    
### Explainable -Random Forest Regressor
def FullFoldRFR(Input,Label, Fold = 'KFold', nfolds = 10, optimize = 'R2', random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds
    optimize: Optimizing target in Pycaret (Default: 'R2')
    '''
    X = Input
    y = Label
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Shuffle':
        folds = ShuffleSplit(n_splits=nfolds, test_size = 0.25, random_state=random_state)

    #Setup tuned model from pycaret
    setup_data = pd.concat([X,y], axis = 1)
    reg = regression.setup(data = setup_data
                           , target = y.name
                           , silent = True
                           , numeric_features = X.columns.drop('Sex')                           
                           , session_id = random_state)       
    rf = regression.create_model('rf')
    tuned_rf = regression.tune_model(rf, n_iter = 18, optimize = optimize, choose_better = True)
    final_model_rf = regression.finalize_model(tuned_rf)
    RFR_best = RandomForestRegressor(**final_model_rf.get_params())

    feature_names = ['F{}'.format(i) for i in range(X.shape[1])]
    fold_pred = np.zeros(len(X))
    rf_preds = np.zeros(len(X))
    shap_values = np.zeros(shape = X.shape)
    rf_importance_df = pd.DataFrame()
    acc_score = []
    best_score = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values,y.values)):
        print("fold n°{}".format(fold_))
        trn_data = X.iloc[trn_idx]; trn_label = y.iloc[trn_idx]
        val_data = X.iloc[val_idx]; val_label = y.iloc[val_idx]

        RFR_best.fit(trn_data,trn_label)
        fold_pred[val_idx] = RFR_best.predict(val_data)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = RFR_best.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        rf_importance_df = pd.concat([rf_importance_df, fold_importance_df], axis=0)

        accuracy = mean_squared_error(val_label, RFR_best.predict(val_data))
        rf_preds += RFR_best.predict(X) / (nfolds)
        acc_score.append(accuracy)    
        print(f'Mean accuracy score: {accuracy:.3}')    
        
        ## Shap
        explainer = shap.TreeExplainer(RFR_best)
        shap_values += explainer.shap_values(X) / (nfolds)        

    #Predict    
    print(f'Total accuracy score: {mean_squared_error(y, rf_preds):.3}')

    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = rf_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}        
    order = rf_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = rf_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('Random Forest over {} Folds'.format(nfolds))
    plt.tight_layout()
    plt.show()    
    print("Best Acc score: {:<8.5f}".format(np.amin(acc_score)))

    ### SHAP Analysis
    print('========================================================================')
    print("SHAP Analysis : Regression")
    shap.summary_plot(shap_values, features=X, feature_names=X.columns)
    plt.show()
    return [RFR_best,y,rf_preds, explainer, shap_values]
    
    
    
### Prediction - Random Forest Classifier
def PredCVRFC(Input,Label, Fold = 'KFold', nfolds = 10, optimize = 'AUC', threshold = 0.5, random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    Fold: Select Fold want to use. 'KFold','Stratified'
    nfolds: Number of folds
    optimize: Optimizing target in Pycaret (Default: 'AUC')
    '''
    X = Input
    y = Label
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=random_state)
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Stratified':
        folds = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state=random_state)

    #Setup tuned model from pycaret        
    setup_data = pd.concat([X_train,y_train], axis = 1)
    clf = classification.setup(data = setup_data
                           , target = y_train.name
                           , numeric_features = X_train.columns.drop('Sex')                           
                           , silent = True
                           , session_id = random_state)       
    rf = classification.create_model('rf')
    tuned_rf = classification.tune_model(rf, n_iter = 18, optimize = optimize, choose_better = True)
    final_model_rf = classification.finalize_model(tuned_rf)
    RFC_best = RandomForestClassifier(**final_model_rf.get_params())

    feature_names = ['F{}'.format(i) for i in range(X.shape[1])]
    fold_pred = np.zeros(len(X_train))
    rf_preds = np.zeros(len(X_test))
    rf_importance_df = pd.DataFrame()
    shap_values = np.zeros(shape = X_train.shape)
    #oob_score = []
    acc_score = []
    best_score = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        print("fold n°{}".format(fold_))
        trn_data = X_train.iloc[trn_idx]; trn_label = y_train.iloc[trn_idx]
        val_data = X_train.iloc[val_idx]; val_label = y_train.iloc[val_idx]

        RFC_best.fit(trn_data,trn_label)
        fold_pred[val_idx] = RFC_best.predict(val_data)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train.columns
        fold_importance_df["importance"] = RFC_best.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        rf_importance_df = pd.concat([rf_importance_df, fold_importance_df], axis=0)

        accuracy = accuracy_score(val_label, RFC_best.predict(val_data))
        rf_preds += RFC_best.predict_proba(X_test)[:,1] / (nfolds)
        acc_score.append(accuracy)    
        print(f'Mean accuracy score: {accuracy:.3}')    

        ## Shap
        explainer = shap.TreeExplainer(RFC_best)
        shap_values += explainer.shap_values(X_test) / (nfolds)
        
    #Plot Confusion Matrix    
    predicted = rf_preds > threshold
    conf_mat = confusion_matrix(y_test, predicted)

    cm_matrix = pd.DataFrame(data=conf_mat, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    
    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = rf_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}    
    order = rf_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = rf_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('Random Forest over {} Folds'.format(nfolds))
    plt.tight_layout()
    plt.show()
    print("Best Acc score: {:<8.5f}".format(np.amax(acc_score)))
        
    ### SHAP Analysis
    print('========================================================================')
    print("SHAP Analysis : Classifier")
    shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
    plt.show()
    return [RFC_best,y_test,predicted, explainer, shap_values]
    
### Prediction - Random Forest Regressor
def PredCVRFR(Input,Label, split = 0.3, Fold = 'KFold', nfolds = 10, optimize = 'R2', random_state = 123456):
    '''
    Input: Input data(X)
    Label: Label data(y), It must be series type.
    split: Set ratio of train/test split. (Default: 0.3)
    Fold: Select Fold want to use. 'KFold','Shuffled' (Default: 'KFold')
    nfolds: Number of folds (Default: 10)
    optimize: Optimizing target in Pycaret (Default: 'R2')
    '''

    X = Input
    y = Label
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split, random_state=random_state)
    if Fold == 'KFold':
        folds = KFold(n_splits = nfolds, shuffle = True, random_state = random_state)
    elif Fold == 'Shuffle':
        folds = ShuffleSplit(n_splits=nfolds, test_size = 0.25, random_state=random_state)

    #Setup tuned model from pycaret
    regression.set_config('X', X_train)
    regression.set_config('y', y_train)
    setup_data = pd.concat([X_train,y_train], axis = 1)
    reg = regression.setup(data = setup_data
                           , target = y_train.name
                           , numeric_features = X_train.columns.drop('Sex')                           
                           , silent = True
                           , session_id = random_state)        
    rf = regression.create_model('rf')
    tuned_rf = regression.tune_model(rf, n_iter = 18, optimize = optimize, choose_better = True)
    final_model_rf = regression.finalize_model(tuned_rf)
    RFR_best = RandomForestRegressor(**final_model_rf.get_params())

    feature_names = ['F{}'.format(i) for i in range(X_train.shape[1])]
    fold_pred = np.zeros(len(X_train))
    rf_preds = np.zeros(len(X_test))
    shap_values = np.zeros(shape = X_train.shape)
    rf_importance_df = pd.DataFrame()
    acc_score = []
    best_score = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        print("fold n°{}".format(fold_))
        trn_data = X_train.iloc[trn_idx]; trn_label = y_train.iloc[trn_idx]
        val_data = X_train.iloc[val_idx]; val_label = y_train.iloc[val_idx]

        RFR_best.fit(trn_data,trn_label)
        fold_pred[val_idx] = RFR_best.predict(val_data)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train.columns
        fold_importance_df["importance"] = RFR_best.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        rf_importance_df = pd.concat([rf_importance_df, fold_importance_df], axis=0)

        accuracy = mean_squared_error(val_label, RFR_best.predict(val_data))
        rf_preds += RFR_best.predict(X_test) / (nfolds)
        acc_score.append(accuracy)    
        print(f'Mean accuracy score: {accuracy:.3}')    

        ## Shap
        explainer = shap.TreeExplainer(RFR_best)
        shap_values += explainer.shap_values(X_test) / (nfolds)
        
    #Predict    
    plt.hist(y_test, label = 'True')
    plt.hist(rf_preds, label = 'Pred', alpha = 0.5)
    plt.legend(loc='upper right')
    plt.show()
    print(f'MSE: {mean_squared_error(y_test,rf_preds):.6}')
    print(f'R2: {r2_score(y_test,rf_preds):.3}')
        
    #Plot feature importance
    plt.figure(figsize=(20, 10))
    col = rf_importance_df.feature.unique()
    clrs = {col[i]:sns.color_palette("viridis", n_colors = len(col))[i]  for i in range(len(col))}        
    order = rf_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    sns.barplot(x = 'importance', y = 'feature', data = rf_importance_df, order = order, ci = 'sd', palette= clrs)
    plt.title('Random Forest over {} Folds'.format(nfolds))
    plt.tight_layout()
    plt.show()
    print("Best Acc score: {:<8.5f}".format(np.amin(acc_score)))
    
    ### SHAP Analysis
    print('========================================================================')
    print("SHAP Analysis : Regression")
    shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
    plt.show()
    return [RFR_best,y_test,rf_preds, explainer, shap_values]
