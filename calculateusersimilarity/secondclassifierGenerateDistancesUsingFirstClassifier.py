#Import libraries:
import numpy as np
import pandas as pd
from functools import reduce
import multiprocessing

import multiprocessing
from itertools import product
import json


def merge_names(a, b):
    return '{} & {}'.format(a, b)
def generateComparisons(outputfileslocation,first_classifier_traing_cv_dates,traing_ips,true_ips,second_classifier_traing_cv_dates,featur,spl,clfl,predict_or_proba,distance,use_bhatdictl=False,bhatdicti={},testing=False,processesl=16):
    global ips
    global t_ips
    global dates
    global t_dates
    global features
    global clf
    global sp
    global bhatdict
    global use_bhatdict
    global distanceFunction
    global processes
    processes = processesl


    #globals()['processes'] = processes

    distanceFunction = distance
    use_bhatdict = use_bhatdictl
    bhatdict = bhatdicti
    sp = spl
    ips = traing_ips
    dates = first_classifier_traing_cv_dates
    t_ips = true_ips
    t_dates = second_classifier_traing_cv_dates
    features = featur
    clf = clfl
    teststr = ''
    if (testing):
        teststr = '_testing'
    if(predict_or_proba == 'proba'):
        probasdict = create_probas_dict()
        save_dict_as_hdf(probasdict,outputfileslocation + '/probadicti_xgb7b' + teststr)
        return probasdict
    else:
        predict = create_predictions_dict()
        save_dict_as_hdf(predict, outputfileslocation + '/predicti_xgb7b' + teststr)
        return predict

# def addNewComarisons_proba(experiment_name,first_classifier_traing_cv_dates,traing_ips,second_classifier_traing_cv_dates,featur,spl,clfl):
#     global ips
#     global dates
#     global t_dates
#     global features
#     global clf
#     global sp
#     sp = spl
#     ips = traing_ips
#     dates = first_classifier_traing_cv_dates
#     t_dates = second_classifier_traing_cv_dates
#     features = featur
#     clf = clfl
#     df_results = pd.read_hdf('data/' + experiment_name + '/probadictit_xgb7b.hdf', 'table')
#     df_results.to_hdf('data/' + experiment_name + '/probadictit_xgb7b_original.hdf', 'table')
#
#     probasdict = create_probas_dict()
#     df_results_new = pd.DataFrame.from_dict(probasdict)
#     df_results_new.to_hdf('data/' + experiment_name + '/probadictit_xgb7b_new'+".hdf",'table')
#
#     frames = [df_results, df_results_new]
#     result = pd.concat(frames)
#     result.to_hdf('data/' + experiment_name + '/probadictit_xgb7b'+".hdf",'table')
#     return result

# Možná by se hodilo přidat pandas

# In[3]:


#from sklearn.externals import joblib
#clf_forest = joblib.load('clf_xgb7b_samehour_allfeatures.pkl')
# In[4]:



# nezapomenout zkusit krome predict zkusit pouzivat probability a nebo log_probability pro vetsi presnost viz
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_log_proba

# Get predictions from the classifier

# In[5]:

def get_predictions_for_ip_with_dict(ip,trueip,truedate):
    prediction = {}
    for date in sorted(dates):
        prediction[date] = []
        #print date
        for i in range(0,24):
            #hour = "{0:0>2}".format(i)
            #feature_list = []
            #for feature in features:
            #    feature_list.append(sp.get_distance_bhat(trueip,ip,date,truedate,hour,hour,feature))
            feature_list = bhatdict[trueip][truedate][ip][date][i]
            feature_tuple = tuple(feature_list)
            clf_result = get_result_from_classificator_predict(feature_tuple)
            prediction[date].append(clf_result)

            #if not all(p == -1 for p in feature_list):
            #prediction[date].append(clf.predict(np.array([feature_list]))[0])
            #else:
            #    prediction[date].append(-1)
    return prediction


#multitherading
def get_predictions_for_ip(ip,trueip,truedate):
    prediction = {}
    for date in sorted(dates):
        prediction[date] = []
        #print date
        for i in range(0,24):
            hour = "{0:0>2}".format(i)
            feature_list = []
            for feature in features:
                feature_list.append(sp.get_distance_bhat(trueip,ip,truedate,date,hour,hour,feature))

            #feature_list = bhatdict[trueip][truedate][ip][date][i]
            feature_tuple = tuple(feature_list)
            clf_result = get_result_from_classificator_predict(feature_tuple)
            prediction[date].append(clf_result)
    return prediction


# Get probabilites from the classifier

#Think about using log proba
def get_proba_for_ip_with_dict(ip,trueip,truedate):
    prediction = {}
    for date in sorted(dates):
        prediction[date] = []
        #print date
        for i in range(0,24):
            #hour = "{0:0>2}".format(i)
            #feature_list = []
            #for feature in features:
            #    feature_list.append(sp.get_distance_bhat(trueip,ip,date,truedate,hour,hour,feature))
            #feature_tuple = tuple(feature_list)

            #prediction[date].append(clf.predict_proba(np.array([feature_list]))[:,1])
            feature_list = bhatdict[trueip][truedate][ip][date][i]
            feature_tuple = tuple(feature_list)
            clf_result = get_result_from_classificator_proba(feature_tuple)
            prediction[date].append(clf_result)
    return prediction
def get_proba_for_ip_hell(ip,trueip,truedate):
    prediction = {}
    for date in sorted(dates):
        prediction[date] = []
        #print date
        for i in range(0,24):
            hour = "{0:0>2}".format(i)
            feature_list = []
            for feature in features:
                feature_list.append(sp.get_distance_hell(trueip,ip,truedate,date,hour,hour,feature))
            #feature_list = [x if x is not -1 else None for x in feature_list]
            if not all(p == -1 for p in feature_list):
                feature_tuple = tuple(feature_list)
                #feature_list = bhatdict[trueip][truedate][ip][date][i]
                #feature_tuple = tuple(feature_list)
                clf_result = get_result_from_classificator_proba(feature_tuple)
                prediction[date].append(clf_result)
            else:
                #prediction[date].append(float('NaN'))
                prediction[date].append(-1)

    return prediction
#Think about using log proba
def get_proba_for_ip(ip,trueip,truedate):
    prediction = {}
    for date in sorted(dates):
        prediction[date] = []
        #print date
        for i in range(0,24):
            hour = "{0:0>2}".format(i)
            feature_list = []
            for feature in features:
                feature_list.append(sp.get_distance_bhat(trueip,ip,truedate,date,hour,hour,feature))
            #feature_list = [x if x is not -1 else None for x in feature_list]
            if not all(p == -1 for p in feature_list):
                feature_tuple = tuple(feature_list)
                #feature_list = bhatdict[trueip][truedate][ip][date][i]
                #feature_tuple = tuple(feature_list)
                clf_result = get_result_from_classificator_proba(feature_tuple)
                prediction[date].append(clf_result)
            else:
                #prediction[date].append(float('NaN'))
                prediction[date].append(-1)

    return prediction
from functools import lru_cache
@lru_cache(maxsize=None)
def get_result_from_classificator_proba(feature_list):
    return clf.predict_proba(np.array([feature_list]))[:, 1]
@lru_cache(maxsize=None)
def get_result_from_classificator_predict(feature_list):
    return clf.predict(np.array([feature_list]))[0]

# In[7]:


def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y) # which returns None since it mutates z
    return z


# In[8]:


from multiprocessing import Pool
def Simulation(data):
    ip = data[0]
    trueip = data[1]
    truedate = data[2]
    res = {}
    if use_bhatdict:
        res[ip] = get_predictions_for_ip_with_dict(ip, trueip, truedate)
    else:
        res[ip] = get_predictions_for_ip(ip,trueip,truedate)
    return res
def get_predictions(trueip,truedate):
    p = Pool(16)
    pairsofdates = []
    result = p.map(Simulation,[[ip,trueip,truedate] for ip in ips])
    #result = np.array(resultlist)
    p.close()
    p.join()
    prediction = reduce( (lambda x, y: merge_two_dicts(x,y)), result)
    return prediction


# In[9]:

def simulation_proba(data):
    ip = data[0]
    trueip = data[1]
    truedate = data[2]
    res = {}
    #print (ip,trueip,truedate)
    if use_bhatdict:
        res[ip] = get_proba_for_ip_with_dict(ip,trueip,truedate)
    else:
        if distanceFunction == 'hellinger':
            res[ip] = get_proba_for_ip_hell(ip,trueip,truedate)
        else:
            res[ip] = get_proba_for_ip(ip,trueip,truedate)
    return res
def get_probas(trueip,truedate):
    with multiprocessing.Pool(processes) as p:
        print(processes)
        result = p.map(simulation_proba, [[ip, trueip, truedate] for ip in ips])
        #result = pool.starmap(get_proba_for_ip, [(ip, trueip, truedate) for ip in ips])
        proba = reduce( (lambda x, y: merge_two_dicts(x,y)), result)
        return proba


# as ty testing date is more far from the dataset, the results are less precise, plot also exact precentage

# In[14]:


def create_predictions_dict():
    predictiondict = {}
    for ip in ips:
        print (ip)
        num_ips = len(ips)
        i = 1
        predictiondict[ip] = {}
        for date in t_dates:
            print(i, num_ips)
            i += 1
            print(date)
            predictiondict[ip][date] = get_predictions(ip,date)
    return predictiondict
        


# In[15]:



def create_probas_dict():
    probasdict = {}
    num_ips = len(t_ips)
    i=1
    for ip in t_ips:
        print (ip)
        print(i,num_ips)
        i+=1
        probasdict[ip] = {}
        for date in t_dates:
            print(date)
            probasdict[ip][date] = get_probas(ip,date)
    return probasdict

def np_dumper(obj):
    if isinstance (obj, np.ndarray):
        return obj.tolist()
    return obj.__dict__


# In[18]:

def save_dict_as_hdf(dicti, name):
    with open (name+'.json', 'w') as fp:
        json.dump (dicti, fp,default=np_dumper)
    df_results = pd.DataFrame.from_dict(dicti)
    df_results.to_hdf(name+".hdf",'table')


