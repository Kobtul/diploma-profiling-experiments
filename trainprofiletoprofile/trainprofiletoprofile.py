import os

import json

from tools.plotprofiletoprofile import plot_clf_results
from trainprofiletoprofile.trainclassxgb import trainfirst_splittedf, trainfirst_splotted_random_forest, \
    trainfirst_splotted_decision_tree
from tools.supportjupyterfunctions import SupportFunctions
from trainprofiletoprofile.trainingfirstclassifierGenerateTrainingData import generateTrainingData

LOAD_TRAINIG_DATA_FROM_FILE = False


def get_profile_to_profile(sp,outputfileslocation='', load_trainig_data_from_file=False,distanceFunction='bhat',processes=16,classifierType='xgboost'):
    training_dates = sp.dates
    training_ips = sp.ips
    features = sp.features

    if load_trainig_data_from_file:
        if os.path.exists(outputfileslocation + 'T_X.txt'):
            with open(outputfileslocation + 'T_X.txt') as data_file:
                x_train = json.load(data_file)
        if os.path.exists(outputfileslocation + 'T_Y.txt'):
            with open(outputfileslocation + 'T_Y.txt') as data_file:
                y_train = json.load(data_file)
    else:
        x_train, y_train = generateTrainingData(distanceFunction, training_dates,training_ips, features, sp,processes)
        with open(outputfileslocation+'T_X.txt', 'w') as fp:
            json.dump(x_train, fp)
        with open(outputfileslocation + 'T_Y.txt', 'w') as fp:
            json.dump(y_train, fp)


    print('First classificator training data loaded!')
    if(classifierType=='xgboost'):
        classfier = trainfirst_splittedf(outputfileslocation, x_train, y_train, features)
    elif(classifierType=='randomforest'):
        classfier = trainfirst_splotted_random_forest(outputfileslocation,x_train,y_train)
    elif(classifierType=='decisiontree'):
        classfier = trainfirst_splotted_decision_tree(outputfileslocation,x_train,y_train)
    else:
        classfier = trainfirst_splotted_random_forest(outputfileslocation,x_train,y_train)
    print('First classificator fitted!')

    from sklearn.externals import joblib
    joblib.dump(classfier, outputfileslocation + 'clf_custom.pkl')

    plot_clf_results(classfier,outputfileslocation,x_train,y_train,features)

    return classfier


if __name__ == "__main__":
    split_classB_to_tcpudp = True
    try:
        with open("fromfrantiseknew.json") as data_file:
            profiles = json.load(data_file)
    except IOError:
        print('Result not found')
    sp = SupportFunctions(profiles, split_classB_to_tcpudp)
    get_profile_to_profile(sp,LOAD_TRAINIG_DATA_FROM_FILE)




    # if LOAD_CLASSIFIER_FROM_FILE:
    #     from sklearn.externals import joblib
    #     classfier = joblib.load('data/' + experiment_name + '/clf_xgb7b.pkl')
    # else:
    #     if split_classB_to_tcpudp:
    #         classfier = trainfirst_splittedf(experiment_name, x_train, y_train, features)
    #     else:
    #         classfier = trainfirst(experiment_name, x_train, y_train, features)
    # with open('data/' + experiment_name + '/' + addinfo + 'T_X.txt', 'w') as fp:
    #     json.dump(X, fp)
    # with open('data/' + experiment_name + '/' + addinfo + 'T_Y.txt', 'w') as fp:
    #     json.dump(Y, fp)