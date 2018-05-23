#Import libraries:
import numpy as np
import pandas as pd
from functools import reduce
from multiprocessing import Pool
import json

def get_bhat_dict_for_ip(ip, trueip, truedate):
    bhatval = {}
    for date in sorted(dates):
        bhatval[date] = []
        #print date
        for i in range(0,24):
            hour = "{0:0>2}".format(i)
            feature_list = []
            for feature in features:
                feature_list.append(sp.get_distance_bhat(trueip,ip,date,truedate,hour,hour,feature))
            bhatval[date].append(tuple(feature_list))
    return bhatval

def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y) # which returns None since it mutates z
    return z
def Simulation(data):
    ip = data[0]
    trueip = data[1]
    truedate = data[2]
    res = {}
    print (ip,trueip,truedate)
    res[ip] = get_bhat_dict_for_ip(ip, trueip, truedate)
    return res
def get_bhatdict(trueip, truedate):
    p = Pool(16)
    result = p.map(Simulation, [[ip, trueip, truedate] for ip in ips])
    p.close()
    p.join()
    proba = reduce( (lambda x, y: merge_two_dicts(x,y)), result)
    return proba

def create_bhatdistance_dict():
    bhatdict = {}
    num_ips = len(ips)
    i = 0
    for ip in ips:
        print(ip)
        print(i, num_ips)
        i += 1
        bhatdict[ip] = {}
        for date in t_dates:
            print(date)
            bhatdict[ip][date] = get_bhatdict(ip, date)
    return bhatdict


def generateBhatDistances(experiment_name,first_classifier_traing_cv_dates,traing_ips,second_classifier_traing_cv_dates,featur,spl):
    global ips
    global dates
    global t_dates
    global features
    global sp
    sp = spl
    ips = traing_ips
    dates = first_classifier_traing_cv_dates
    t_dates = second_classifier_traing_cv_dates
    features = featur

    bhatdistance = create_bhatdistance_dict()
    #save_dict_as_hdf(bhatdistance, 'data/bhatdict')
    save_dict_as_json(bhatdistance, 'data/bhatdict')
    return bhatdistance
def save_dict_as_hdf(dicti, name):
    df_results = pd.DataFrame.from_dict(dicti)
    df_results.to_hdf(name+".hdf",'table')
def save_dict_as_json(dicti,name):
    with open (name+'.json', 'w') as fp:
        json.dump(dicti, fp)
