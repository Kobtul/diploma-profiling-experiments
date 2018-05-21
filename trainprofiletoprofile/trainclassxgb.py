import os
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier

rng = np.random.RandomState(31339)
os.environ["JOBLIB_TEMP_FOLDER"] = "/code/temp"  # or to whatever you want
os.environ["OMP_NUM_THREADS"] = "4"  # or to whatever you want


from sklearn.externals import joblib

def modelfit(alg, x_train, y_train,features, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    num_round = 10
    if useTrainCV:
        print(cv_folds)
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train, feature_names=features)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0]) 

        # alg.set_params(n_estimators=len(cvresult['train-auc-mean']))
        # num_round=len(cvresult['train-auc-mean'])
    # Fit the algorithm on the data
    alg.fit(X=x_train, y=y_train, eval_metric='auc')

    # Predict training set:
    # dtrain_predictions = alg.predict(x_test)
    # dtrain_predprob = alg.predict_proba(x_test)[:,1]

    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    #plot_importance(alg.booster())
    #plt.savefig(file_loc +'xgb-featuresplt.pdf')
    #plt.close()
    # print alg.booster().get_fscore()
    # xgb.plot_importance(alg.booster())
    # from sklearn.externals import joblib
    # joblib.dump(alg, 'alg_samehour.pkl')

def trainfirst(experiment_name,x_train,y_train,features):
    global file_loc
    file_loc = 'data/'+experiment_name +'/'
    xgb7b = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,  # 0,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=1,
        scale_pos_weight=1,
        seed=27,
        max_depth=30,
        min_child_weight=1
    )
    modelfit(xgb7b, np.array(x_train), np.array(y_train),features)
    from sklearn.externals import joblib
    joblib.dump(xgb7b, file_loc+'clf_xgb7b.pkl')
    return xgb7b




def trainfirst_splittedf(outputfileslocation,x_train,y_train,features):
    global file_loc
    file_loc = outputfileslocation
    xgb7b = XGBClassifier(
        learning_rate=0.1,
        n_estimators=180,  # 0,
        gamma=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        nthread=1,
        seed=27,
        max_depth=22,
        min_child_weight=1,
        #missing=-1,
    )
    modelfit(xgb7b, np.array(x_train), np.array(y_train),features,useTrainCV=False)
    #xgb7b.fit(X=np.array(x_train), y=np.array(y_train), eval_metric='auc')
    return xgb7b
def trainfirst_splotted_random_forest(outputfileslocation,x_train,y_train):
    global file_loc
    file_loc = outputfileslocation
    randomgridforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                              max_depth=35, max_features=7, max_leaf_nodes=None,
                                              min_impurity_decrease=0.0, min_impurity_split=None,
                                              min_samples_leaf=1, min_samples_split=2,
                                              min_weight_fraction_leaf=0.0, n_estimators=616, n_jobs=1,
                                              oob_score=False, random_state=None, verbose=0,
                                              warm_start=False)
    randomgridforest.fit(np.array(x_train), np.array(y_train))
    return randomgridforest
def trainfirst_splotted_decision_tree(outputfileslocation,x_train,y_train):
    global file_loc
    file_loc = outputfileslocation
    clf_simple = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100)
    clf_simple.fit(np.array(x_train), np.array(y_train))
    return clf_simple

def train_first_test(experiment_name,x_train,y_train,features):
    global file_loc
    file_loc = 'data/'+experiment_name +'/'
    from xgboost.sklearn import XGBRegressor
    import scipy.stats as st

    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)

    params = {
        "n_estimators": st.randint(3, 15),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }
    #xgbreg = XGBRegressor(nthreads=-1)
    xgbreg = XGBRegressor()

    from sklearn.model_selection import RandomizedSearchCV
    gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)
    gs.fit(x_train, y_train)

    joblib.dump(gs.best_estimator_  , file_loc + 'clf_bestmodel.pkl')
    return gs.best_estimator_
