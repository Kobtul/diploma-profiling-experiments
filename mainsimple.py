import os
import sys
sys.path.append('parse_binet')
from parse_binet.dataGather import generate_profile_from_file
#sys.path.remove('parse_binet')

import webbrowser, os

import argparse
from calculateusersimilarity.calculateusersimilarity import calculateusersimilarity, get_similar_ips, \
    train_threshold_with_lowest_fpr, train_threshold_with_lowest_fpr_whole
from tools.supportjupyterfunctions import SupportFunctions
from trainprofiletoprofile.trainprofiletoprofile import get_profile_to_profile
import pandas as pd
import json
import editor
import numpy as np


def legacy_code(sp):
    first_classifier_traing_cv_dates = [date for date in sp.dates if
                                        date.split('/')[1] == '09' and int(date.split('/')[2]) <= 3]
    # second_classifier_traing_cv_dates = [date for date in sp.dates if
    #                                     date.split('/')[1] == '09' and int(date.split('/')[2]) == 8]

    second_classifier_traing_cv_dates = [date for date in sp.dates if
                                         date.split('/')[1] == '10' and int(date.split('/')[2]) <= 1]
    #testing_dates = [date for date in sp.dates if date.split('/')[1] == '10' and int(date.split('/')[2]) > 10 and int(date.split('/')[2]) <= 18]

    # minus = ['147.32.83.102','147.32.83.186', '147.32.216.123', '147.32.83.224', '147.32.83.205', '147.32.83.206']
    minus = ['147.32.216.123']
    l3 = [x for x in sp.ips if x not in minus]
    first_classifier_traing_ips = l3
    second_classifier_traing_ips = l3
    return first_classifier_traing_cv_dates,second_classifier_traing_cv_dates,first_classifier_traing_ips,second_classifier_traing_ips


def get_known_and_unknown_sp_from_all(sp,split_classB_to_tcpudp):
    first_classifier_traing_cv_dates, second_classifier_traing_cv_dates, first_classifier_traing_ips, second_classifier_traing_ips = legacy_code(
        sp)

    known_json = {}
    for ip in first_classifier_traing_ips:
        known_json[ip] = {}
        known_json[ip]['time'] = {}
        for date in first_classifier_traing_cv_dates:
            known_json[ip]['time'][date] = sp.source_data[ip]['time'][date].copy()

    unknown_json = {}
    for ip in second_classifier_traing_ips:
        unknown_json[ip] = {}
        unknown_json[ip]['time'] = {}
        for date in second_classifier_traing_cv_dates:
            unknown_json[ip]['time'][date] = sp.source_data[ip]['time'][date].copy()
    spknown = SupportFunctions(known_json, split_classB_to_tcpudp)
    spunknown = SupportFunctions(unknown_json, split_classB_to_tcpudp)
    return spknown,spunknown
def dumper(obj):
    if isinstance (obj, set):
        return list (obj)
    elif isinstance (obj, np.ndarray):
        return obj.tolist()
    else:
        return obj.__dict__
def parse_binet_files_and_process(args):
    known_profile = generate_profile_from_file(args.known_data,args.known_ips)
    with open (args.output +'known_profile.json', 'w') as fp:
        json.dump (known_profile, fp,default=dumper)
    unknown_profile = generate_profile_from_file(args.unknown_data,args.unknown_ips)
    with open (args.output + 'unknown_profile.json', 'w') as fp:
        json.dump (unknown_profile, fp,default=dumper)
    if args.only_profiles is False:
        sp_known = SupportFunctions(known_profile, split_classB_to_tcpudp=True)
        sp_unknown = SupportFunctions(unknown_profile, split_classB_to_tcpudp=True)
        process_profile_similarity(args,sp_known,sp_unknown)

def load_profiles_and_process(args):
    sp_known = SupportFunctions(SupportFunctions.load_profiles(args.known_data), split_classB_to_tcpudp=True)
    sp_unknown = SupportFunctions(SupportFunctions.load_profiles(args.unknown_data), split_classB_to_tcpudp=True)
    process_profile_similarity(args,sp_known, sp_unknown)


def process_profile_similarity(args,sp_known,sp_unknown):
    if (args.train_profile_classifier is not None):
        if (len(sp_known.dates) < 2):
            parser.error("There need to be atleast two days in the known capture")
        classifier = get_profile_to_profile(sp_known, args.output, load_trainig_data_from_file=False, distanceFunction='bhat',
                               processes=args.processes, classifierType=args.train_profile_classifier)
    else:
        from sklearn.externals import joblib
        classifier = joblib.load(args.path_to_classifier)
    outputfileslocation = args.output
    if not os.path.exists(outputfileslocation):
        os.makedirs(outputfileslocation)
    df_results_dict = calculateusersimilarity(sp_known, sp_unknown, classifier, outputfileslocation, proba_or_predict='proba',
                                 distanceFunction='bhat',processes=args.processes)
    if (args.visualize_comparisons):
        create_comparisonsjs(df_results_dict)
    df_results = pd.DataFrame.from_dict(df_results_dict)
    threshold = args.set_threshold
    if(args.train_threshold):
        threshold = train_threshold(args,df_results)
    result = get_similar_ips(threshold, df_results,args.output, ignoreipswithfewprofiles=False,plotheatmap=args.plot_heatmap)
    print(result)
def create_comparisonsjs(df_results_dict):
    import htmlark
    with open('static/comparisons/js/comparisons.js', 'w') as fp:
        fp.write('var comparisons =')
        json.dump(df_results_dict, fp, default=dumper)
        fp.write(';')
        # webbrowser.open('file://' + os.path.realpath('only_profiles.html'))
    packed_html = htmlark.convert_page('static/comparisons.html', ignore_errors=True)
    with open(args.output + args.visualize_comparisons, 'wb') as fp:
        fp.write(packed_html.encode("utf-8", "replace"))
    print('Profile comparisons visualization is saved in '+args.output+' ' + args.visualize_comparisons)
def create_profiles_visualization(profiles):
    import htmlark
    with open('static/ip_profiles/js/profile.js', 'w') as fp:
        fp.write('var profile =')
        json.dump(profiles, fp, default=dumper)
        fp.write(';')
    packed_html = htmlark.convert_page('static/only_profiles.html', ignore_errors=True)
    with open(args.output + args.visualize_profile, 'wb') as fp:
        fp.write(packed_html.encode("utf-8", "replace"))
    # webbrowser.open('file://' + os.path.realpath('only_profiles.html'))
    print('Visualization is saved in '+args.output+' '+ args.visualize_profile)
def train_p2p(args):
    sp_known = SupportFunctions(SupportFunctions.load_profiles(args.known_data), split_classB_to_tcpudp=True)
    if (len(sp_known.dates) < 2):
        parser.error("There need to be atleast two days in the known capture")
    get_profile_to_profile(sp_known, args.output, load_trainig_data_from_file=False, distanceFunction='bhat',processes=args.processes,classifierType=args.train_profile_classifier)
def load_dataframe_and_show_results(args):
    #df_resultslocation = 'output/probadicti_xgb7b.hdf'
    df_resultslocation = args.comparisons_file
    df_results = pd.read_hdf(df_resultslocation, 'table')
    if(args.visualize_comparisons):
        df_results_dict = df_results.to_dict()
        create_comparisonsjs(df_results_dict)
    result = get_similar_ips(args.set_threshold, df_results,args.output,ignoreipswithfewprofiles=False,plotheatmap=args.plot_heatmap)
    print(result)

def call_train_threshold(args):
    df_resultslocation = args.comparisons_file
    df_results = pd.read_hdf(df_resultslocation, 'table')
    train_threshold(args,df_results)
def train_threshold(args,df_results):
    #df_resultslocation = 'output/probadicti_xgb7b.hdf'
    # df_resultslocation = args.comparisons_file
    # df_results = pd.read_hdf(df_resultslocation, 'table')
    train_threshold_per_day = False
    if(train_threshold_per_day):
        FPRmin, highest_threshold_with_fprmin, ACCmin = train_threshold_with_lowest_fpr(df_results, load_thresholds_dict=False, processes=args.processes, ignoreipswithfewprofiles=False,threshold_density=args.threshold_density,output=args.output)
        human_readable_result = 'Threshold: {}, False-Positive rate on training: {}, ACC rate on training: {}'.format(
            highest_threshold_with_fprmin, FPRmin, ACCmin)
        computer_readable_results = '{},{},{}'.format(highest_threshold_with_fprmin, FPRmin, ACCmin)
        with open(args.output + 'trained_threshold.txt', 'w') as data_file:
            data_file.write(human_readable_result)
        with open(args.output + 'trained_threshold_machine.txt', 'w') as data_file:
            data_file.write(computer_readable_results)

        print(human_readable_result)
    else:
        FPRmin, highest_threshold_with_fprmin, ACCmin = train_threshold_with_lowest_fpr_whole(df_results, load_thresholds_dict=False, processes=args.processes, ignoreipswithfewprofiles=False,threshold_density=args.threshold_density,output=args.output)
        human_readable_result = 'Threshold: {}, False-Positive rate on training: {}, ACC rate on training: {}'.format(
            highest_threshold_with_fprmin, FPRmin, ACCmin)
        computer_readable_results = '{},{},{}'.format(highest_threshold_with_fprmin, FPRmin, ACCmin)
        with open(args.output + 'trained_threshold_whole.txt', 'w') as data_file:
            data_file.write(human_readable_result)
        with open(args.output + 'trained_threshold_whole_machine.txt', 'w') as data_file:
            data_file.write(computer_readable_results)

        print(human_readable_result)
    return highest_threshold_with_fprmin


def CheckExt(choices):
    class Act(argparse.Action):
        def __call__(self,parser,namespace,fname,option_string=None):
            ext = os.path.splitext(fname)[1][1:]
            if ext not in choices:
                option_string = '({})'.format(option_string) if option_string else ''
                parser.error("file doesn't end with one of {}{}".format(choices,option_string))
            else:
                setattr(namespace,self.dest,fname)

    return Act


def my_arg_parser(arg_line):
    return arg_line.split()
def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue
if __name__ == "__main__":

    print('|¯| |¯| |¯| |¯ | |  |¯ |¯    /¯\ |¯| |¯   /¯\ |  |    \| |¯| | |   |\| |¯ |¯ |¯\\')
    print('|¯  |¯\ |_| |¯ | |_ |¯  ¯|   |¯| |¯\ |¯   |¯| |_ |_    | |_| |_|   | | |¯ |¯ |_/ ')


    list_of_classifiers = ["xgboost", "randomforest", "decisiontree"]

    parser = argparse.ArgumentParser(prog='KUBA',description='Kickass uber binetflow application is focused on analazing users based on their behavior from the binetflows capture.'
                                                             'The --known-data or --comparison-file needs to be specified',fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args=my_arg_parser
    parser.add_argument('-p','--processes',type=check_positive,default=2,help='Number of processes used by program')
    parser.add_argument('-o','--output',default='output',help='Folder with outputs of the program')

    parser.add_argument('-kd',"--known-data", type=str, help="Path to the binetflow file or the JSON profiles of the known users.",
                              action=CheckExt({'json', 'binetflow', 'binetflows'}))
    parser.add_argument('-ud',"--unknown-data", type=str, help="Path to the binetflow file or the JSON profiles of the unknown users.",
                              action=CheckExt({'json', 'binetflow', 'binetflows'}))
    parser.add_argument('-cf',"--comparisons-file", type=str, help="Path to dataframe with all comparisons between known and unknown users.",
                              action=CheckExt({'hdf'}))
    parser.add_argument('-kip','--known-ips', type=str, nargs="+", help="known IP adresses")
    parser.add_argument('-uip','--unknown-ips', type=str, nargs="+", help="unknown IP adresses")
    parser.add_argument('-pc','--path-to-classifier', default='p2pclassifier.pkl', help='load custom profile to profile classifier')
    parser.add_argument('-op','--only-profiles', help='generate only profiles from binetflows', action='store_true')
    parser.add_argument('-ph','--plot-heatmap', action='store_true', help='Plot a heatmap representing the results')
    parser.add_argument('-tp','--train-profile-classifier', choices=list_of_classifiers, help='Train profile to profile classifier.')
    parser.add_argument('-tt','--train-threshold', action='store_true', help='Train threshold for the user classifier.')
    parser.add_argument('-st','--set-threshold', default=0.27,type=float, help='Train threshold for the user classifier.')

    parser.add_argument('-td','--threshold-density', default=0.01,type=float, help='This parameter is bound to --train-threshold. The value of the parameter determines the length of step between 0 and 1. Default value is 0.01')
    parser.add_argument('-vp','--visualize-profile', default='only_profiles.html', help='Converts profile to javascript and create html with visualization')
    parser.add_argument('-an','--add-note-to-ip-profile', nargs=2,help='Adds notes for the IP adress in the profile. The profile and the IP is required.')
    parser.add_argument('-vc', '--visualize-comparisons', default='comparisons.html',help='Creates the html visualization from the comparisons file.')

    args = parser.parse_args()
    #args = parser.parse_args('--comparisons-file /run/media/david/Linux\ storage/datazedny/beta_randomforestp2pclassifier_test/probadicti_xgb7b.hdf --plot-heatmap')
    if not args.output.endswith('/'):
        args.output+='/'
    if not os.path.exists(args.output): os.mkdir(args.output)

    if(args.known_data is not None):
        ext = os.path.splitext(args.known_data)[1][1:]
        if(ext == 'binetflow' or ext == 'binetflows'):
            if(args.unknown_data is None):
                if args.known_ips is None:
                    parser.error("binetflows require setting -kip")
                known_profile = generate_profile_from_file(args.known_data, args.known_ips)
                with open(args.output + 'known_profile.json', 'w') as fp:
                    json.dump(known_profile, fp, default=dumper)
                if(args.visualize_profile):
                    create_profiles_visualization(known_profile)
                if(args.train_profile_classifier is not None):
                    sp_known = SupportFunctions(known_profile,
                                                split_classB_to_tcpudp=True)
                    if(len(sp_known.dates) < 2):
                        parser.error("There need to be atleast two days in the known capture")
                    get_profile_to_profile(sp_known, args.output, load_trainig_data_from_file=False, distanceFunction='bhat',
                                           processes=args.processes,classifierType=args.train_profile_classifier)
            else:
                if args.known_ips is None or args.unknown_ips is None:
                    parser.error("binetflows require setting -kip and -uip")
                parse_binet_files_and_process(args)
        elif(ext == 'json'):
            if(args.unknown_data is None):
                if (args.visualize_profile is not None):
                    known_profile =SupportFunctions.load_profiles(args.known_data)
                    create_profiles_visualization(known_profile)
                else:
                    train_p2p(args)
            else:
                load_profiles_and_process(args)
    elif(args.comparisons_file is not None):
        if(args.train_threshold):
            call_train_threshold(args)
        else:
            load_dataframe_and_show_results(args)
    elif(args.add_note_to_ip_profile is not None):
        profileloc = args.add_note_to_ip_profile[0]
        ipnote = args.add_note_to_ip_profile[1]
        profile = SupportFunctions.load_profiles(profileloc)
        if(ipnote in profile):
            if('note' in profile[ipnote]):
                note = editor.edit(contents=str.encode(profile[ipnote]['note']))
            else:
                note = editor.edit(contents=b"# Enter note here")
            profile[ipnote]['note'] = note.decode('utf-8')
            with open(profileloc, 'w') as fp:
                json.dump(profile, fp, default=dumper)

    else:
        parser.error("Nothing to do, exiting")


