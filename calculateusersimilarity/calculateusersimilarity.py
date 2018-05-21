import json
import multiprocessing

from calculateusersimilarity.secondclassifierGenerateDistancesUsingFirstClassifier import generateComparisons
from tools.supportjupyterfunctions import SupportFunctions
from sklearn.externals import joblib
from tools.secondclassifierManual import plot_heatmap, plot_all_heatmaps, plot_heatmap_match_mismatch, \
    plot_heatmap_match_mismatch_only_TPFP, plot_histogram, plot_confusion_matrix, calculate_CFMatrix_averageddays, \
    generateHeatmapDataframe
import pandas as pd
import numpy as np
def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y) # which returns None since it mutates z
    return z

def merge_known_unknows(known,unknown):
    mergedKeys = set(known.keys()) | set(unknown.keys())  # pipe is union
    result = {}
    for ip in mergedKeys:
        result[ip] = {}
        result[ip]['time'] = {}
        if ip in known and ip in unknown:
            result[ip]['time'] = merge_two_dicts(known[ip]['time'],unknown[ip]['time'])
        else:
            if ip in known:
                result[ip]['time'] = known[ip]['time'].copy()
            else:
                result[ip]['time'] = unknown[ip]['time'].copy()

    return result

def calculateusersimilarity(spknown,spunknown,classfier,outputfileslocation='',proba_or_predict='proba',distanceFunction='bhat',processes=16):
    spknown_training_dates = spknown.dates
    spknown_ips = spknown.ips
    spknown_features = spknown.features

    spunknown_dates = spunknown.dates
    spunknown_ips = spunknown.ips
    spunknown_features = spunknown.features
    sp = SupportFunctions(merge_known_unknows(spknown.source_data,spunknown.source_data))


    df_results = generateComparisons(outputfileslocation, spknown_training_dates,
                                     spknown_ips,spunknown_ips,
                                     spunknown_dates, spunknown_features, sp, classfier,
                                     proba_or_predict, distanceFunction,processesl=processes)
    return df_results
def string_from_ip_list(ips):
    stra = ''
    for ip in ips:
        stra += str(ip) + ' '
    return stra
def get_similar_ips(threshold,df_results,output,ignoreipswithfewprofiles=False,plotheatmap=False,plot_cnf_matrix=False,plot_ips_in_dataframe=False):
    spunknown_ips = list(df_results)
    spunknown_dates = list(df_results.transpose())
    df_3 = pd.Series()
    result = {}

    if plotheatmap:
        # plot_heatmap(df_results,ignoreipswithfewprofiles,output)
        # plot_heatmap_match_mismatch(df_results,ignoreipswithfewprofiles,output,threshold)
        # plot_heatmap_match_mismatch_only_TPFP(df_results,ignoreipswithfewprofiles,output,threshold)
        plot_all_heatmaps(df_results,ignoreipswithfewprofiles,output,threshold)
    if plot_ips_in_dataframe:
        df_2 = pd.DataFrame.from_dict(df_results[spunknown_ips[0]][spunknown_dates[0]])
        with open(output + 'knownunknownips.txt', 'w') as data_file:
            data_file.write('knownipslist:{}\n'.format(list(df_2)))
            data_file.write('knownipsstring:{}\n'.format(string_from_ip_list(list(df_2))))
            data_file.write('unknownipslist:{}\n'.format(spunknown_ips))
            data_file.write('unknownipsstring:{}\n'.format(string_from_ip_list(spunknown_ips)))


    if(plot_cnf_matrix):
        TP, FP, TN, FN = calculate_CFMatrix(df_results,{}, threshold, ignoreipswithfewprofiles=ignoreipswithfewprofiles)
        cnf_matrix_p = np.array([[TP, FP], [FN, TN]])
        plot_confusion_matrix(cnf_matrix_p, ['same', 'notsame'],output)

        uTP, uFP, uTN, uFN = calculate_CFMatrix_averageddays(df_results, ignoreipswithfewprofiles, output, threshold)
        with open(output + 'confusionmatrixforthreshold{}.txt'.format(threshold), 'w') as data_file:
            data_file.write(str(threshold) +'\n')
            data_file.write('{},{},{},{}\n'.format(TP,FP,TN,FN))
            data_file.write('{},{},{},{}\n'.format(uTP,uFP,uTN,uFN))
    for truedate in spunknown_dates:
        for trueip in spunknown_ips:
            df_2 = pd.DataFrame.from_dict(df_results[trueip][truedate])
            for ip in list(df_2):
                df3 = pd.DataFrame(df_2[ip].values.tolist(), columns=[str(x) for x in range(0, 24)])
                df3.index = df_2.index
                df3 = df3.applymap(lambda x: float('NaN') if x == -1 else x[0])
                df_3[ip] = df3.mean(axis=1).mean()
                if not ignoreipswithfewprofiles:
                    df_3[ip] = df3.mean(axis=1).mean()
                else:
                    if not df3.empty and df3.count(axis=1).sum() >= 6:
                        df_3[ip] = df3.mean(axis=1).mean()
                    else:
                        df_3[ip] = float('NaN')
            df_3 = df_3.dropna()
            #df_3 = df_2.mean()
            #print(df_3)
            minname = ""
            minvalue = 11
            for name, value in df_3.iteritems():
                if minvalue >= value:
                    minvalue = value
                    minname = name
            if(minvalue is not 11):
                if minvalue < threshold:
                    result[trueip] = minname,minvalue
            else:
                result[trueip] = ''
    return result

def calculate_CFMatrix(df_results,df3cache,threshold,ignoreipswithfewprofiles=True,knownIPStodrop=[]):
    TP=0
    FP=0
    TN=0
    FN=0
    df_3 = pd.Series()
    truedates = list(df_results.transpose())
    trueips = sorted(list(df_results))
    for truedate in truedates:
        if(truedate not in df3cache):
            df3cache[truedate] = {}
        for trueip in sorted(trueips):
            if(trueip in df3cache[truedate]):
                df_3,knownIPs =df3cache[truedate][trueip]
            else:
                df_2 = pd.DataFrame.from_dict(df_results[trueip][truedate])
                df_2 = df_2.drop(knownIPStodrop, axis=1)
                df_3 = pd.Series()
                knownIPs = list(df_2)
                #print(trueip,truedate,'----------------------')
                for ip in list(df_2):
                    df3 = pd.DataFrame(df_2[ip].values.tolist(), columns=[str(x) for x in range(0, 24)])
                    df3.index = df_2.index
                    df3 = df3.applymap(lambda x: float('NaN') if x == -1 else x[0])
                    #df_3[ip] = df3.mean(axis=1).mean()
                    # if(trueip == "147.32.83.34" and ip == "147.32.83.69"):
                    #     print(df3.mean(axis=1).mean())
                    ipvalue = df3.mean(axis=1).mean()
                    numberofcomparisons = df3.count(axis=1).sum()
                    #print(ip,ipvalue,numberofcomparisons)
                    if not ignoreipswithfewprofiles:
                        df_3[ip] = ipvalue
                    else:
                        if not df3.empty and numberofcomparisons >= 10:
                            df_3[ip] = ipvalue
                        else:
                            df_3[ip] = float('NaN')
                df_3 = df_3.dropna()
                #df3cache[truedate][trueip] = df_3,knownIPs
                #df_3 = df_2.mean()
                #print(df_3)
            minname = ""
            minvalue = 11
            for name, value in df_3.iteritems():
                if minvalue >= value:
                    minvalue = value
                    minname = name
            if(minvalue is not 11):
                if minvalue > threshold:
                    if trueip in knownIPs:
                        FN+=1
                    else:
                        TN+=1
                    # if minname == trueip:
                    #     FN+=1
                    # else:
                    #     TN+=1 #TODO FIX THIS
                else:
                    if minname == trueip:
                        TP+=1
                    else:
                        #print('aaaa')
                        #print(trueip,minname,minvalue)
                        FP+=1
    #result = {}
    #result[threshold] = [TP,FP,TN,FN]
    #print(result)
    return [TP,FP,TN,FN]

def translate_calculate_CFMatrix_parameters(array):
    # df_results = array[0]
    # threshold = array[1]
    # df_test = pd.DataFrame.from_dict(df_results[df_results.columns[0]][0])
    # listofips = list(df_test)
    # TParray = []
    # FParray = []
    # TNarray = []
    # FNarray = []
    # for ip in listofips:
    #     TP, FP, TN, FN = calculate_CFMatrix(df_results,array[1],array[2],knownIPStodrop=[ip])
    #     TParray.append(TP)
    #     FParray.append(FP)
    #     TNarray.append(TN)
    #     FNarray.append(FN)
    # finalTP = round(np.array(TParray).mean())
    # finalFP = round(np.array(FParray).mean())
    # finalTN = round(np.array(TNarray).mean())
    # finalFN = round(np.array(FNarray).mean())
    # result = {}
    # result[threshold] = [finalTP,finalFP,finalTN,finalFN]
    # print(threshold)
    # return result
    threshold = array[1]
    df_results = array[0]
    ignoreipswithfewprofiles = array[2]
    df_3cache = array[3]
    result = {}
    TP, FP, TN, FN = calculate_CFMatrix(df_results,df_3cache,threshold,ignoreipswithfewprofiles=ignoreipswithfewprofiles)
    result[threshold] = [TP, FP, TN, FN]
    return result
def init_df_3_cache(df_results,ignoreipswithfewprofiles=False,knownIPStodrop=[]):
    truedates = list(df_results.transpose())
    trueips = sorted(list(df_results))
    df3cache = {}
    for truedate in truedates:
        if (truedate not in df3cache):
            df3cache[truedate] = {}
        for trueip in sorted(trueips):
            print(truedate,trueip)
            df_2 = pd.DataFrame.from_dict(df_results[trueip][truedate])
            df_2 = df_2.drop(knownIPStodrop, axis=1)
            df_3 = pd.Series()
            knownIPs = list(df_2)
            # print(trueip,truedate,'----------------------')
            for ip in list(df_2):
                df3 = pd.DataFrame(df_2[ip].values.tolist(), columns=[str(x) for x in range(0, 24)])
                df3.index = df_2.index
                df3 = df3.applymap(lambda x: float('NaN') if x == -1 else x[0])
                # df_3[ip] = df3.mean(axis=1).mean()
                # if(trueip == "147.32.83.34" and ip == "147.32.83.69"):
                #     print(df3.mean(axis=1).mean())
                ipvalue = df3.mean(axis=1).mean()
                numberofcomparisons = df3.count(axis=1).sum()
                # print(ip,ipvalue,numberofcomparisons)
                if not ignoreipswithfewprofiles:
                    df_3[ip] = ipvalue
                else:
                    if not df3.empty and numberofcomparisons >= 10:
                        df_3[ip] = ipvalue
                    else:
                        df_3[ip] = float('NaN')
            df_3 = df_3.dropna()
            df3cache[truedate][trueip] = df_3, knownIPs
            # df_3 = df_2.mean()
            # print(df_3)
    return df3cache

def get_all_results_from_CNFdict(CNFdict,threshold_arr):
    all_results = {}
    for threshold in threshold_arr:
        TP, FP, TN, FN = CNFdict[threshold]
        ACC = float(TP + TN) / float(TP + TN + FP + FN)
        if 2 * TP + FP + FN is not 0:
            F1 = float(2 * TP) / float(2 * TP + FP + FN)
        else:
            F1 = 1
        if FP + TN is not 0:
            FPR = FP / float(FP + TN)
        else:
            FPR = 1
        if TP + FN is not 0:
            TPR = TP / float(TP + FN)
        else:
            TPR = 1

        # according to https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        all_results[threshold] = {}
        all_results[threshold]['confusion'] = CNFdict[threshold]
        all_results[threshold]['ACC'] = ACC
        all_results[threshold]['F1'] = F1
        all_results[threshold]['FPR'] = FPR
        all_results[threshold]['TPR'] = TPR
    return all_results
def calculate_THRmin_from_all_results(all_results):
    FPRmin = 2
    highest_threshold_with_fprmin = -1
    last_min_distance = 2
    threshold_arr = sorted(all_results.keys())
    for thr in threshold_arr:
        FPR = all_results[thr]['FPR']
        ACC = all_results[thr]['ACC']

        # if (FPR is not None) and (TPR is not None):
        #    distance_t = distance([FPR,TPR],[0,1])
        if FPR is not None:
            if (FPR <= FPRmin):
                # if(distance_t < last_min_distance):
                # last_min_distance = distance_t
                FPRmin = FPR
                highest_threshold_with_fprmin = thr
                ACCmin = ACC
    return FPRmin, highest_threshold_with_fprmin, ACCmin

def train_threshold_with_lowest_fpr_whole(df_results,load_thresholds_dict=False,processes=2,ignoreipswithfewprofiles=True,proba_or_predict='proba',threshold_density=0.001,output='output/'):
    spunknown_ips = list(df_results)
    spunknown_dates = list(df_results.transpose())
    threshold_arr = np.arange(0.0, 1.01, threshold_density)
    dfssn = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
    CNFdict = {}
    for threshold in threshold_arr:
        TP, FP, TN, FN = calculate_CFMatrix_averageddays(df_results,False,'',threshold,dfssn=dfssn)
        CNFdict[threshold] = TP, FP, TN, FN
    all_results = get_all_results_from_CNFdict(CNFdict, threshold_arr)
    with open(output + 'threshold_results_whole.txt', 'w') as fp:
        json.dump(all_results, fp, indent=4)
    FPRmin, highest_threshold_with_fprmin, ACCmin = calculate_THRmin_from_all_results(all_results)
    with open(output + 'threshold_results_whole_best.txt', 'w') as fp:
        json.dump(all_results[highest_threshold_with_fprmin], fp, indent=4)
    return FPRmin, highest_threshold_with_fprmin, ACCmin

def train_threshold_with_lowest_fpr(df_results,load_thresholds_dict=False,processes=2,ignoreipswithfewprofiles=True,proba_or_predict='proba',threshold_density=0.001,output='output/'):
    spunknown_ips = list(df_results)
    spunknown_dates = list(df_results.transpose())

    all_results = {}

    if(load_thresholds_dict):
        with open(output + 'threshold_results.txt', 'r') as fp:
            all_results = json.load(fp)
    else:
        threshold_arr = np.arange(0.0, 1.01, threshold_density)
        df_3cache = init_df_3_cache(df_results)
        from functools import reduce
        with multiprocessing.Pool(processes) as pool:
            resultdicts = pool.map(translate_calculate_CFMatrix_parameters, [[df_results, threshold, ignoreipswithfewprofiles,df_3cache] for threshold in threshold_arr])
            #resultdicts = pool.map(calculate_CFMatrix, threshold_arr)
            CNFdict = reduce( (lambda x, y: merge_two_dicts(x,y)), resultdicts)
        all_results = get_all_results_from_CNFdict(CNFdict,threshold_arr)

        with open(output + 'threshold_results.txt', 'w') as fp:
            json.dump(all_results, fp,indent=4)

    FPRmin, highest_threshold_with_fprmin, ACCmin = calculate_THRmin_from_all_results(all_results)
    #plot_results(all_results,'output/',proba_or_predict)
    PRINT_HISTOGRAMS = False
    if PRINT_HISTOGRAMS:
        for truedate in spunknown_dates:
            for trueip in spunknown_ips:
                plot_histogram(trueip, truedate)
    #print('plotting results done')
    plot_heatmap(df_results, ignoreipswithfewprofiles, output)
    print('plotting heatmap done')
    #print(FPRmin, highest_threshold_with_fprmin, ACCmin)
    return FPRmin, highest_threshold_with_fprmin, ACCmin

def load_profiles_calcusersimilarity(name):
    try:
        with open(name) as data_file:
            profiles = json.load(data_file)
    except IOError:
        print('Result not found')
    return profiles

if __name__ == "__main__":
    split_classB_to_tcpudp = True
    spknown = SupportFunctions(SupportFunctions.load_profiles("known.json"), split_classB_to_tcpudp)
    spunknown = SupportFunctions(SupportFunctions.load_profiles("unknown.json"), split_classB_to_tcpudp)
    classfier = joblib.load('classfier.pkl')
    df_results = calculateusersimilarity(spknown,spunknown,classfier)
    print(get_similar_ips(0.3, df_results))