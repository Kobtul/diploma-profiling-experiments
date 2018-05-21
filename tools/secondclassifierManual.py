
# coding: utf-8

# In[28]:


from tools.supportjupyterfunctions import SupportFunctions
#Import libraries:
import numpy as np
import pandas as pd
from plotly import __version__

#print (__version__) # requires version >= 1.9.0
import plotly.offline as py
import plotly.graph_objs as go
import multiprocessing
import math
import itertools
# Možná by se hodilo přidat pandas
#TODO nezapomenout zkusit krome predict zkusit pouzivat probability a nebo log_probability pro vetsi presnost viz
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_log_proba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, classes,outputfileslocation,
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
    plt.figure()
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
    file_loc = outputfileslocation
    plt.savefig(file_loc +'secondconfmatrix.pdf',bbox_inches='tight')
    plt.close()
def generateHeatmapDataframe(df_results,ignoreipswithfewprofiles):
    list_of_series = []
    df_3 = pd.Series(list(df_results))
    for trueip in sorted(list(df_results)):
        list_by_ip = []
        for truedate in list(df_results.transpose()):
            df_2 = pd.DataFrame.from_dict(df_results[trueip][truedate])
            df_3 = pd.Series(list(df_results))
            for ip in sorted(list(df_2)):
                df3 = pd.DataFrame(df_2[ip].values.tolist(), columns=[str(x) for x in range(0, 24)])
                df3.index = df_2.index
                df3 = df3.applymap(lambda x: float('NaN') if x == -1 else x[0])
                if not ignoreipswithfewprofiles:
                    df_3[ip] = df3.mean(axis=1).mean()
                else:
                    if not df3.empty and df3.count(axis=1).sum() >= 6:
                        df_3[ip] = df3.mean(axis=1).mean()
                    else:
                        df_3[ip] = float('NaN')
            # df_3 = df_2.mean()
            list_by_ip.append(df_3)
        dfs = pd.DataFrame(list_by_ip)
        dfn = dfs.mean(axis=0)
        dfn.name = trueip
        list_of_series.append(dfn)
    dfss = pd.DataFrame(list_of_series)
    # print(dfss)
    # dfss = dfss.dropna(axis=0, how='all')
    # dfss = dfss.dropna(axis=1, how='all')
    return dfss


def calculate_CFMatrix_averageddays(df_results, ignoreipswithfewprofiles, outputfileslocation, threshold, dfssn=None):
    file_loc = outputfileslocation
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if (dfssn is None):
        dfssn = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
    dmatch = pd.DataFrame().reindex_like(dfssn)
    for trueip in sorted(dfssn.transpose()):
        minname = ""
        minvalue = 11
        for name, value in dfssn.transpose()[trueip].iteritems():
            if minvalue >= value:
                minvalue = value
                minname = name
        # print(minname,minvalue)
        if (minvalue is not 11):
            if minvalue > threshold:
                if trueip in list(dfssn):
                    FN += 1
                else:
                    TN += 1
            else:
                if minname == trueip:
                    TP += 1
                else:
                    FP += 1

    return TP, FP, TN, FN
def generateNumberOfComparisonsDataframe(df_results,ignoreipswithfewprofiles):
    list_of_series = []
    df_3 = pd.Series(list(df_results))
    for trueip in sorted(list(df_results)):
        list_by_ip = []
        for truedate in list(df_results.transpose()):
            df_2 = pd.DataFrame.from_dict(df_results[trueip][truedate])
            df_3 = pd.Series(list(df_results))
            for ip in sorted(list(df_2)):
                df3 = pd.DataFrame(df_2[ip].values.tolist(), columns=[str(x) for x in range(0, 24)])
                df3.index = df_2.index
                df3 = df3.applymap(lambda x: float('NaN') if x == -1 else x[0])
                df_3[ip] = df3.count(axis=1).sum()
            # df_3 = df_2.mean()
            list_by_ip.append(df_3)
        dfs = pd.DataFrame(list_by_ip)
        dfn = dfs.mean(axis=0)
        dfn.name = trueip
        list_of_series.append(dfn)
    dfss = pd.DataFrame(list_of_series)
    # print(dfss)
    # dfss = dfss.dropna(axis=0, how='all')
    # dfss = dfss.dropna(axis=1, how='all')
    return dfss
def generate_heatmpas_by_days(df_results,ignoreipswithfewprofiles,outputfileslocation,nameoffile='heatmap'):
    for truedate in list(df_results.transpose()):
        newdf = df_results.filter(like=truedate, axis=0)
        dheatmap = generateHeatmapDataframe(newdf, ignoreipswithfewprofiles)
        plot_heatmap(newdf, ignoreipswithfewprofiles, outputfileslocation, dfss=dheatmap,
                     nameoffile=nameoffile + truedate.replace("/", "_"))


def plot_all_heatmaps(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold):
    dheatmap = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
    plot_heatmap(df_results,ignoreipswithfewprofiles,outputfileslocation,dfss=dheatmap)
    #plot_heatmap_match_mismatch(df_results, ignoreipswithfewprofiles, outputfileslocation, threshold, dfssn=dheatmap)
    #plot_heatmap_match_mismatch_only_TPFP(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold,dfssn=dheatmap)
    plot_second_underthreshold_matches(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold,dfssn=dheatmap)
    dcomparisons = generateNumberOfComparisonsDataframe(df_results, ignoreipswithfewprofiles)
    plot_heatmap(df_results,ignoreipswithfewprofiles,outputfileslocation,dfss=dcomparisons,nameoffile='heatmap_comparisons')
    generate_heatmpas_by_days(df_results,ignoreipswithfewprofiles,outputfileslocation,nameoffile='heatmap')

def plot_heatmap(df_results,ignoreipswithfewprofiles,outputfileslocation,dfss=None,nameoffile='heatmap'):
    file_loc = outputfileslocation
    if(dfss is None):
        dfss = generateHeatmapDataframe(df_results,ignoreipswithfewprofiles)
    #print(dfss)
    #dfss = dfss.dropna(axis=0, how='all')
    #dfss = dfss.dropna(axis=1, how='all')
    plt.figure(figsize=(20, 20))
    sns.heatmap(dfss, annot=True,fmt='.2f')
    plt.savefig(file_loc + nameoffile +'.pdf', bbox_inches='tight')
    #plt.savefig(file_loc + 'heatmap.png', bbox_inches='tight')
    #plt.figure(figsize=(20, 20))
    #dfss.plot()
    #plt.savefig(file_loc + 'heatmap_raw.pdf', bbox_inches='tight')
    #plt.savefig('myfile.pdf')
# def plot_second_underthreshold_matches(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold,dfssn=None):
#     file_loc = outputfileslocation
#     if (dfssn is None):
#         dfssn = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
#     dmatch = pd.DataFrame().reindex_like(dfssn)
#     for trueip in sorted(list(dfssn)):
#         minname = ""
#         minvalue = 11
#         for name, value in dfssn[trueip].iteritems():
#             if minvalue >= value:
#                 minvalue = value
#                 minname = name
#         for name, value in dfssn[trueip].iteritems():
#             if(value<threshold):
#                 if(name == minname):
#                     if(minname==trueip):
#                         dmatch[trueip][name] = 0
#                     else:
#                         dmatch[trueip][name] = 1
#                 else:
#                     dmatch[trueip][name] = 2
#         # if (minvalue is not 11):
#         #     if minvalue <= threshold:
#         #         if minname == trueip:
#         #             # TP
#         #             dmatch[trueip][trueip] = 0
#         #         else:
#         #             # FP
#         #             dmatch[trueip][minname] = -1
#     # minname = ""
#         # minvalue = 11
#         # secondminname = ""
#         # secondminvalue = 12
#         # for name, value in dfssn[trueip].iteritems():
#         #     if minvalue >= value:
#         #         secondminname = minname
#         #         secondminvalue = secondminvalue
#         #         minvalue = value
#         #         minname = name
#         # if secondminvalue > threshold:
#     plt.figure(figsize=(20, 20))
#     sns.heatmap(dmatch, annot=True,linewidths=1, linecolor='black',cbar=False,vmax=3)
#     # for _, spine in ax.spines.items():
#     #     spine.set_visible(True)
#     plt.savefig(file_loc + 'heatmap_mismatch_underthreshold.pdf', bbox_inches='tight')
def plot_second_underthreshold_matches(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold,dfssn=None):
    file_loc = outputfileslocation
    if (dfssn is None):
        dfssn = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
    dmatch = pd.DataFrame().reindex_like(dfssn)
    for trueip in sorted(dfssn.transpose()):
        minname = ""
        minvalue = 11
        for name, value in dfssn.transpose()[trueip].iteritems():
            if minvalue >= value:
                minvalue = value
                minname = name
        #print(minname,minvalue)
        for name, value in dfssn.transpose()[trueip].iteritems():
            if(value<threshold):
                if(minname == name):
                    if(minname==trueip):
                        dmatch[name][trueip] = 0
                    else:
                        #print(minvalue,value)
                        dmatch[name][trueip] = 1
                else:
                    dmatch[name][trueip] = 2
    plt.figure(figsize=(20, 20))
    sns.heatmap(dmatch, annot=True,linewidths=1, linecolor='black',cbar=False,vmax=3)
    # for _, spine in ax.spines.items():
    #     spine.set_visible(True)
    plt.savefig(file_loc + 'heatmap_mismatch_underthreshold.pdf', bbox_inches='tight')
def plot_heatmap_match_mismatch(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold,dfssn=None):
    file_loc = outputfileslocation
    if (dfssn is None):
        dfssn = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
    dmatch = pd.DataFrame().reindex_like(dfssn)
    for trueip in sorted(list(dfssn)):
        minname = ""
        minvalue = 11
        for name, value in dfssn[trueip].iteritems():
            if minvalue >= value:
                minvalue = value
                minname = name
        # df_4 = df_3.apply(lambda x: float('NaN'))
        #print('trueip: {} - {}'.format(trueip, str([(name, value) for name, value in df_3.iteritems()])))
        #df_4 = pd.Series()
        if (minvalue is not 11):
            if minvalue > threshold:
                if minname == trueip:
                    # pass
                    # FN
                    dmatch[trueip][trueip] = 0
                else:
                    #passd
                    # TN
                    # df_4[trueip]= 1
                    dmatch[trueip][minname] = 1
            else:
                if minname == trueip:
                    # TP
                    dmatch[trueip][trueip] = 2
                else:
                    dmatch[trueip][minname] = 3
                    # FP
                    # df_4[trueip] = 3

    plt.figure(figsize=(20, 20))
    sns.heatmap(dmatch, annot=False)
    plt.savefig(file_loc + 'heatmap_mismatch.pdf', bbox_inches='tight')
    plt.savefig(file_loc + 'heatmap_mismatch.png', bbox_inches='tight')
def plot_heatmap_match_mismatch_only_TPFP(df_results,ignoreipswithfewprofiles,outputfileslocation,threshold,dfssn=None):
    file_loc = outputfileslocation
    if (dfssn is None):
        dfssn = generateHeatmapDataframe(df_results, ignoreipswithfewprofiles)
    dmatch = pd.DataFrame().reindex_like(dfssn)
    dmatch = pd.DataFrame().reindex_like(dfssn)
    for trueip in sorted(list(dfssn)):
        minname = ""
        minvalue = 11
        for name, value in dfssn[trueip].iteritems():
            if minvalue >= value:
                minvalue = value
                minname = name
        # df_4 = df_3.apply(lambda x: float('NaN'))
        #print('trueip: {} - {}'.format(trueip, str([(name, value) for name, value in df_3.iteritems()])))
        #df_4 = pd.Series()
        if (minvalue is not 11):
            if minvalue > threshold:
                if minname == trueip:
                    pass
                    # FN
                    #dmatch[trueip][trueip] = 0
                else:
                    pass
                    # TN
                    # df_4[trueip]= 1
                    #dmatch[trueip][minname] = 1
            else:
                if minname == trueip:
                    #TP
                    dmatch[trueip][trueip] = 0
                else:
                    dmatch[trueip][minname] = 1
                    # FP
                    # df_4[trueip] = 3

    plt.figure(figsize=(20, 20))
    sns.heatmap(dmatch, annot=False)
    plt.savefig(file_loc + 'heatmap_mismatch_onlytpfp.pdf', bbox_inches='tight')
    plt.savefig(file_loc + 'heatmap_mismatch_onlytpfp.png', bbox_inches='tight')
import json

def get_testing_result(outputfileslocationl,first_classifier_traing_cv_dates,traing_ips,testing_dates,df_result,proba_or_predictl,threshold,ignoreipswithfewprofilesn,file_name='testing_result'):
    global df_results
    global outputfileslocation
    global dates
    global ips
    global true_dates
    global proba_or_predict
    global ignoreipswithfewprofiles
    ignoreipswithfewprofiles = ignoreipswithfewprofilesn
    outputfileslocation = outputfileslocationl
    df_results = df_result
    dates = first_classifier_traing_cv_dates
    true_dates = testing_dates
    ips = traing_ips
    proba_or_predict = proba_or_predictl
    threshold =float(threshold)
    cnf_matrix = calculate_CFMatrix(threshold)
    print(cnf_matrix)
    TP, FP, TN, FN = cnf_matrix[threshold]
    ACC = float(TP + TN) / float(TP + TN + FP + FN)
    if FP + TN is not 0:
        FPR = FP / float(FP + TN)
    else:
        FPR = 1
    if TP + FN is not 0:
        TPR = TP / float(TP + FN)
    else:
        TPR = 1
    if 2 * TP + FP + FN is not 0:
        F1 = float(2 * TP) / float(2 * TP + FP + FN)
    else:
        F1 = 1
    print('TP: %d' % TP)
    print('FP: %d' % FP)
    print('TN: %d' % TN)
    print('FN: %d' % FN)
    print('ACC: %f' % ACC)
    print('F1: %f' % F1)
    print('FPR: %f' % FPR)
    print('TPR: %f' % TPR)
    testing_results = {}
    testing_results['confusion'] = cnf_matrix[threshold]
    testing_results['ACC'] = ACC
    testing_results['TPR'] = TPR
    testing_results['FPR'] = FPR
    testing_results['F1'] = F1
    testing_results['threshold'] = threshold

    with open(outputfileslocation +'/'+file_name+'.txt', 'w') as fp:
        json.dump(testing_results, fp,indent=4)
    cnf_matrix_p = np.array([[TP, FP], [FN, TN]])
    plot_confusion_matrix(cnf_matrix_p, ['same', 'notsame'],outputfileslocation)

    return testing_results
# as ty testing date is more far from the dataset, the results are less precise, plot also exact precentage
def suggest_threshold(outputfileslocationl,first_classifier_traing_cv_dates,traing_ips,second_classifier_traing_cv_dates,featur,spl,df_result,proba_or_predictl,LOAD_THRESHOLDS_DICT,ignoreipswithfewprofilesn):
    global df_results
    global outputfileslocation
    global dates
    global ips
    global true_dates
    global file_loc
    global proba_or_predict
    global ignoreipswithfewprofiles
    ignoreipswithfewprofiles = ignoreipswithfewprofilesn
    all_results = {}
    outputfileslocation = outputfileslocationl
    #file_loc = 'data/' + experplotiment_name + '/'
    file_loc = outputfileslocation
    df_results = df_result
    dates = first_classifier_traing_cv_dates
    true_dates = second_classifier_traing_cv_dates
    ips = traing_ips
    proba_or_predict = proba_or_predictl
    if(LOAD_THRESHOLDS_DICT):
        with open(outputfileslocation + 'all_results.txt', 'r') as fp:
            all_results = json.load(fp)
    else:
        threshold_arr = np.arange(0.0, 1.01, 0.01)
        from functools import reduce
        with multiprocessing.Pool(processes=24) as pool:
            resultdicts = pool.map(calculate_CFMatrix, threshold_arr)
            CNFdict = reduce( (lambda x, y: merge_two_dicts(x,y)), resultdicts)
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

            #according to https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            #https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
            all_results[threshold] = {}
            all_results[threshold]['confusion'] = CNFdict[threshold]
            all_results[threshold]['ACC'] = ACC
            all_results[threshold]['F1'] = F1
            all_results[threshold]['FPR'] = FPR
            all_results[threshold]['TPR'] = TPR

        with open(outputfileslocation +'all_results.txt', 'w') as fp:
            json.dump(all_results, fp,indent=4)

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
    plot_results(all_results,file_loc,proba_or_predict)
    PRINT_HISTOGRAMS = False
    if PRINT_HISTOGRAMS:
        for truedate in true_dates:
            for trueip in ips:
                plot_histogram(trueip,truedate)
    print('plotting results done')
    plot_heatmap(df_result,ignoreipswithfewprofiles,outputfileslocation)
    print('plotting heatmap done')
    print(FPRmin,highest_threshold_with_fprmin,ACCmin)
    return FPRmin,highest_threshold_with_fprmin,ACCmin


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def merge_two_dicts(x,y):
    z = x.copy()
    z.update(y) # which returns None since it mutates z
    return z
def calculate_ration(x):
    activehours = x.count(0) + x.count(1)
    if activehours is not 0:
        ratio = 1-float(x.count(0))/activehours
        return ratio
    else:
        return -1

def calculate_ration_proba(lst):
    activehours =sum(x is not -1 for x in lst)
    lst = [x for x in lst if x is not -1]
    if activehours is not 0:
        ratio = float(sum(map(sum, lst))/float(activehours))
        return ratio
    else:
        #TODO this is bad
        return None
# In[41]:

#TP = I correctly chose, that it is the same user
#FP = I chose that it is the same user, but it was not
#TN = I correctly chose, that it is not the same user.
#FN = I chose that it is not the same user, but it was.
# if minimum is higher than threshold take it as different user, if not, take the one with the minimum as the same user

# In[19]:

#TODO Calculate df3 better
def calculate_CFMatrix(threshold):
    TP=0
    FP=0
    TN=0
    FN=0
    df_3 = pd.Series()
    for truedate in true_dates:
        for trueip in ips:
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
                if minvalue > threshold:
                    if minname == trueip:
                        FN+=1
                    else:
                         TN+=1
                else:
                    if minname == trueip:
                        TP+=1
                    else:
                        FP+=1
    result = {}
    result[threshold] = [TP,FP,TN,FN]
    return result

# import plotly.tools as tls
# tls.embed('https://plot.ly/~cufflinks/8')
# import plotly.plotly as py
# import cufflinks as cf
# import pandas as pd
# import numpy as np
# print (cf.__version__)
# def plot_ipinfo(trueip,truedate):
#     checkedip = trueip
#     df_2 = pd.DataFrame.from_dict(df_results[trueip][truedate])
#     df3 = pd.DataFrame(df_2[checkedip].values.tolist(), columns=[str(x) for x in range(0,24)])
#     df3.index = df_2.index
#     df3 = df3.applymap(lambda x : x[0])
#     df3.plot.hist(stacked=True)
#     df3.transpose().plot.hist(stacked=True)
#     df3.iplot(kind='bar', barmode='stack', filename='cufflinks/grouped-bar-chart')
#     df3.transpose().iplot(kind='bar', barmode='stack', filename='cufflinks/grouped-bar-chart')
#     print(df3.mean(axis=1))
#     show_results_interact(trueip,truedate)
#     plot_histogram(trueip,truedate)
#     #df4 = df3.apply(calculate_ration_proba_test,axis=1,raw=True)
#     display(df3)


def plot_histogram(trueip,truedate):
    import re
    dataframe = pd.DataFrame.from_dict(df_results[trueip][truedate])
    dataframe = dataframe.applymap(calculate_ration_proba)
    truedate = re.sub(r'/', '_', truedate)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    dataframe.plot.hist(stacked=True)
    plt.savefig(file_loc +'histogram'+str(trueip) + '_' + str(truedate) + proba_or_predict +'.pdf')
    plt.close()
    #plt.figure()
    #dataframe.plot.hist(cumulative=True)
    #plt.savefig(file_loc +'histogramcum'+str(trueip) + '_' + str(truedate) + proba_or_predict +'.pdf')
    #plt.close()

def plot_results(allresults,file_loc,proba_or_predict):
    threshold_arr = sorted(allresults.keys())
    ACClist = [allresults[thr]['ACC'] for thr in threshold_arr]
    F1list = [allresults[thr]['F1'] for thr in threshold_arr]
    FPRlist = [allresults[thr]['FPR'] for thr in threshold_arr]
    TPRlist = [allresults[thr]['TPR'] for thr in threshold_arr]

    # for thr in threshold_arr:
    #     print('Treshold {}'.format(thr))
    #     print('FPR {}'.format(allresults[thr]['FPR']))
    #     print('TPR {}'.format(allresults[thr]['TPR']))

    acc = go.Scatter(
            x=threshold_arr,
            y=ACClist,
            name='ACC',
    )
    f1 = go.Scatter(
            x=threshold_arr,
            y=F1list,
            name="F1",
    )
    FPR = go.Scatter(
            x=threshold_arr,
            y=FPRlist,
            name="FPR",
    )
    TPR = go.Scatter(
            x=threshold_arr,
            y=TPRlist,
            name="TPR",
    )
    layout = go.Layout(title='Evaulation of second classifier',
                        xaxis=dict(title='threshold'),
                        yaxis=dict(title='Value'))
    fig = go.Figure(data=[acc,f1,FPR,TPR], layout=layout)
    py.plot(fig, filename=file_loc +'thresholds'+ proba_or_predict +'.html')



    roc = go.Scatter(
            x=FPRlist,
            y=TPRlist,
            #name='ROC',
    )
    layout = go.Layout(title='ROC curve',
                        xaxis=dict(title='FPR'),
                        yaxis=dict(title='TPR'))
    fig = go.Figure(data=[roc], layout=layout)
    py.plot(fig, filename=file_loc +'roc'+ proba_or_predict +'.html')

    from sklearn import metrics
    #print(FPRlist, TPRlist)
    #for i in range(0,len(TPRlist)):


    roc_auc = metrics.auc(FPRlist, TPRlist,reorder=False)
    lw = 2
    trace1 = go.Scatter(x=FPRlist, y=TPRlist,
                        mode='lines',
                        line=dict(color='darkorange', width=lw),
                        name='ROC curve (area = %0.2f)' % roc_auc
                        )

    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=lw, dash='dash'),
                        showlegend=False)

    layout = go.Layout(title='Receiver operating characteristic example',
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'))

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    py.plot(fig, filename=file_loc +'roc'+ proba_or_predict +'_new.html')
    py.plot(fig,image='svg')

    # method I: plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(FPRlist, TPRlist, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig(file_loc +'rocplt'+ proba_or_predict +'.pdf')

    # import plotly.tools as tls
    # plotly_fig = tls.mpl_to_plotly(plt)
    # py.plot(plotly_fig, filename=file_loc +'rocfromplt'+ proba_or_predict +'.html')
