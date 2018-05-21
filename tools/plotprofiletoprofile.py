from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.externals import joblib
import matplotlib
import pandas as pd
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_loc +'xgbconfmatrix.pdf',bbox_inches='tight')


from sklearn import metrics
import pandas as pd


def plot_feature_importances(clf,features):
    importances = clf.feature_importances_
    #importances = clf.booster().get_fscore()
    #print importances
    featimportgraphtrace = go.Bar(
        #x=[x for _,x in sorted(zip(importances,sp.features),reverse=True)],
        #y=sorted(importances)
        #x=sp.features,
        #y=importances
        x=[x for _,x in sorted(zip(importances,features),reverse=True)],
        y=[y for y,_ in sorted(zip(importances,features),reverse=True)],
        #orientation='h',
        #name=key,
        #mode='markers+lines',
        #hoverlabel={'namelength': -1}
    )
    layout = go.Layout(title='Feature importance of xgboost',
                        xaxis=dict(title='Features'),
                        yaxis=dict(title='Value'))
    fig = go.Figure(data=[featimportgraphtrace], layout=layout)
    py.plot(fig, filename=file_loc +'xgb-features.html')

def plot_fscore_old_xgboost(clf,features):
    plt.figure(figsize=(10, 5))
    fdict = {}
    for i in range(0, len(features)):
        fdict['f{}'.format(i)] = features[i]
    old_dict = clf.booster().get_fscore()
    new_dict = {}
    for key in old_dict:
        new_dict[fdict[key]] = old_dict[key]
    #feat_imp = pd.Series(new_dict).sort_values(ascending=True)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #feat_imp.plot(kind='barh', title='Feature Importances')
    #feat_imp.head(11).plot(kind='barh')
    feat_imp = pd.Series(new_dict).sort_values(ascending=False)
    feat_imp = feat_imp.head(11)
    feat_imp = feat_imp.sort_values(ascending=True)
    feat_imp.plot(kind='barh')

    plt.tight_layout()
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.savefig(file_loc +'xgbfscoreplt.pdf',bbox_inches='tight')
    plt.savefig(file_loc +'xgbfscoreplt.png',bbox_inches='tight')

    d_view = [(v, k) for k, v in old_dict.items()]
    d_view.sort(reverse=True)  # natively sort tuples by first element
    with open(file_loc + '/feature_meanings.txt', 'w') as data_file:
        tran = []
        for v, k in d_view:
            tran.append('{}:{}'.format(k,fdict[k]))
        text = ','.join(map(str, tran))
        data_file.write(text)
    df = pd.DataFrame(list(new_dict.items()), columns=['Feature', 'FScore']).sort_values('FScore', ascending=False)
    featimportgraphtrace = go.Bar(
        x=list(df['Feature']),
        y=list(df['FScore']),
    )
    layout = go.Layout(title='Fscore of xgboost',
                       xaxis=dict(title='Features'),
                       yaxis=dict(title='FScore'))
    fig = go.Figure(data=[featimportgraphtrace], layout=layout)
    py.plot(fig, filename=file_loc + 'xgb-features_fscore.html')

def plot_fscore(clf,features):
    plt.figure(figsize=(10, 5))
    new_dict = {}
    for i in range(0, len(features)):
        new_dict[features[i]] = clf.feature_importances_[i]
    feat_imp = pd.Series(new_dict).sort_values(ascending=False)
    feat_imp = feat_imp.head(11)
    feat_imp = feat_imp.sort_values(ascending=True)
    feat_imp.plot(kind='barh')

    plt.tight_layout()
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.savefig(file_loc + 'featureimportances.pdf', bbox_inches='tight')
def plot_clf_testing(clf,experiment_name,x_test,y_test):
    global file_loc
    file_loc = 'data/' + experiment_name + '/'
    dtrain_predictions = clf.predict(x_test)
    dtrain_predprob = clf.predict_proba(x_test)[:, 1]
    ACC = metrics.accuracy_score(y_test, dtrain_predictions)
    AUC = metrics.roc_auc_score(y_test, dtrain_predprob)
    F1 = metrics.f1_score(y_test, dtrain_predictions)
    fpr,tpr,thresholds = metrics.roc_curve(y_test, dtrain_predictions)
    cnf_matrix = confusion_matrix(y_test, dtrain_predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, dtrain_predictions).ravel()
    plot_confusion_matrix(cnf_matrix, ['same', 'notsame'])
    XGBresult = {}
    XGBresult['TN'] = np.asscalar(tn)
    XGBresult['FP'] = np.asscalar(fp)
    XGBresult['FN'] = np.asscalar(fn)
    XGBresult['TP'] = np.asscalar(tp)
    XGBresult['ACC'] = np.asscalar(ACC)
    XGBresult['F1'] = np.asscalar(F1)
    XGBresult['FPR'] =  np.asscalar(fp / float(fp + tn))
    XGBresult['TPR'] = np.asscalar(tp / float(tp + fn))
    import json
    with open('data/'+experiment_name +'/xgbtestingresult.txt', 'w') as fp:
        json.dump(XGBresult, fp,indent=4)



def plot_clf_results(alg,outputfileslocation,x_train,y_train,features):
    global file_loc
    file_loc = outputfileslocation
    #file_loc = 'data/' + experiment_name + '/'
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    with open(file_loc + "modelreport.txt", mode='a') as file:
        file.write("Model Report\nAccuracy : %.4g\nAUC Score (Train): %f" %
                   (metrics.accuracy_score(y_train, dtrain_predictions),
                    metrics.roc_auc_score(y_train, dtrain_predprob)))
    #plot_feature_importances(alg,features)
    plot_fscore(alg,features)


def plot_clf_results_with_load(experiment_name,x_train,y_train,features):
    global file_loc
    file_loc = 'data/' + experiment_name + '/'
    clf = joblib.load(file_loc + 'clf_xgb7b.pkl')
    return plot_clf_results(clf,experiment_name,x_train,y_train,features)

