from old.ifidfcust import load_tf_idf_matrixes, calculate_idf
import random
from functools import reduce
import json
import numpy as np
distanceFunction = ""
def generateTrainingData(distanceFunction,first_classifier_traing_cv_dates,traing_ips,featur,spl,processes=16):
    global ips
    global dates
    global idf_matrixes_dict
    global idf_matrixes_dict_index
    global features
    global sp
    globals()['distanceFunction'] = distanceFunction
    globals()['processes'] = processes

    features = featur
    sp = spl
    if distanceFunction == 'tf-idf':
        idf_matrixes_dict,idf_matrixes_dict_index = load_tf_idf_matrixes()
    ips = traing_ips
    dates = first_classifier_traing_cv_dates

    raw_data = generate_samples_for_dates(dates)
    X, Y = process_results_from_reduce(raw_data)
    return X,Y

def calculate_two_dates(date1,date2):
    print (date1,date2)
    resultXsame = []
    resultXnotsame = []
    for i in range(0,len(ips)):

        ip1=ips[i]

        for j in range(i,len(list(ips))):
            ip2=ips[j]
            for h in range(0,24):
                hour = "{0:0>2}".format(h)
                feature_list = []
                for feature in features:
                    if distanceFunction == 'tf-idf':
                        feature_list.append(calculate_idf(idf_matrixes_dict,idf_matrixes_dict_index,ip1,ip2,date1,date2,hour,hour,feature))
                    elif distanceFunction == 'bhat':
                        feature_list.append(sp.get_distance_bhat(ip1,ip2,date1,date2,hour,hour,feature))
                    else:
                        feature_list.append(sp.get_distance_hell(ip1,ip2,date1,date2,hour,hour,feature))
                if not all(p == -1 for p in feature_list):
                    if ip1 is not ip2:
                        resultXnotsame.append(feature_list)
                    else:
                        resultXsame.append(feature_list)
    return resultXsame,resultXnotsame


# This put all results from all the threads back in the one vectors

# In[20]:


def process_results_from_reduce(result):
    Xsame = reduce( (lambda x, y: x + y), result[:, 0])
    Xnotsame = reduce( (lambda x, y: x + y), result[:, 1])
    Xsame,Xnotsame = balance_dataset(Xsame,Xnotsame)
    X = Xsame + Xnotsame
    Y = [0] * len(Xsame) + [1] * len(Xnotsame)
    return X,Y


def balance_dataset(Xsame,XnotSame):
    if len(XnotSame) > len(Xsame):
        XnotSame = downsample(XnotSame,len(Xsame))
    elif len(XnotSame) < len(Xsame):
        Xsame = downsample(Xsame,len(XnotSame))
    return Xsame,XnotSame
def downsample(X,lenY):
    Xdown = []
    jackpot = random.sample(range(0, lenY), lenY)
    for i in jackpot:
        Xdown.append(X[i])
    return Xdown

def calculate_two_dates_wrapper(i):
    return calculate_two_dates(i[0],i[1])
def generate_samples_for_dates(dates):
    import multiprocessing
    #p = Pool(20)
    pairsofdates = []
    for i in range(0,len(dates)):
        for j in range(i+1,len(dates)):
            pairsofdates.append([dates[i],dates[j]])
    with multiprocessing.Pool(processes) as pool:
        resultlist = pool.map(calculate_two_dates_wrapper, pairsofdates)
        result = np.array(resultlist)
        #print (result.shape,len(result[:, 0]))
        return result
