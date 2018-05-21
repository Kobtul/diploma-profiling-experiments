import math
import numpy as np
import json
import random
import plotly.graph_objs as go
from functools import lru_cache


class SupportFunctions:
    source_data = {}
    features = []
    ips = []
    dates = []
    myGlobal = "hello"

    def __init__(self,profiles,split_classB_to_tcpudp=True):
        self.source_data = profiles
        self.features = self.generate_list_of_features(split_classB_to_tcpudp)
        self.ips = list(self.source_data.keys())
        self.dates = self.getDates()
    # @classmethod
    # def create_from_two_jsons(self,js1,js2,split_classB_to_tcpudp=True):
    #     pass
    # @classmethod
    # def create_from_file(self,name,split_classB_to_tcpudp=True):
    #     try:
    #         with open(name) as data_file:
    #             self.source_data = json.load(data_file)
    #     except IOError:
    #         print('Result not found')
    def getDates(self):
        dates_set = set()
        for ip in self.source_data:
            for date in self.source_data[ip]['time']:
                dates_set.add(date)
        dates = list(dates_set)
        return dates
    def changeGlobal(self):
        self.myGlobal = "bye"
    def normalize(self,h):
        return h / np.sum(h)

    @lru_cache(maxsize=None)
    def bhattacharyya(self,h1, h2):
      '''Calculates the Byattacharyya distance of two histograms.'''
      return 1 - np.sum(np.sqrt(np.multiply(self.normalize(h1), self.normalize(h2))))
    def chisquare(self,h1,h2):
        nh1 = self.normalize(h1)
        nh2 = self.normalize(h2)
        suma = 0
        for i in range(0,len(nh1)):
            if not(nh1[i] == 0 and nh2[i] == 0):
                suma= np.add(suma,np.power(np.subtract(nh1[i],nh2[i]),2)/(np.add(nh1[i],nh2[i])))
        return suma
    def intersetciton(self,h1,h2):
        nh1 = self.normalize(h1)
        nh2 = self.normalize(h2)
        suma = 0
        for i in range(0,len(nh1)):
            suma = np.add(suma,min(nh1[i],nh2[i]))
        return suma
    @lru_cache(maxsize=None)
    def hellinger3(self,p, q):
        _SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64
        return np.sqrt(np.sum((np.sqrt(self.normalize(p)) - np.sqrt(self.normalize(q))) ** 2)) / _SQRT2
    def prepare_lists_to_compare(self,dictA,dictB):
        mergedKeys = set(dictA.keys()) | set(dictB.keys())  # pipe is union
        firsthist = []
        secondhist = []
        for key in mergedKeys:
            if key in dictA:
                firsthist.append(float(dictA[key]))
            else:
                firsthist.append(0)
            if key in dictB:
                secondhist.append(float(dictB[key]))
            else:
                secondhist.append(0)
        return [firsthist,secondhist]

    def reformat_hour(self,hour):
        return "{0:0>2}".format(hour)
    
    """  Function for generating the bhattacharyya distance between any profiles """
    def get_distance_bhat(self,ip1, ip2, date1, date2, hour1, hour2, feature):
        if hour1 not in self.source_data[ip1]['time'][date1] and hour2 not in self.source_data[ip2]['time'][date2]:
            result = -1
        elif hour1 not in self.source_data[ip1]['time'][date1] or hour2 not in self.source_data[ip2]['time'][date2]:
            result = -1
        else:
            classA = self.source_data[ip1]['time'][date1][hour1][feature]
            classB = self.source_data[ip2]['time'][date2][hour2][feature]
            # TODO think about one profile have feature and the second does not
            if not classA and not classB:
                result = -1
            elif not classA or not classB:
                result = -1
            else:
                [firsthist, secondhist] = self.prepare_lists_to_compare(classA, classB)
                result = np.asscalar(self.bhattacharyya(tuple(firsthist), tuple(secondhist)))
        return result
    """  Function for generating the hellinger distance between any profiles """
    def get_distance_hell(self,ip1, ip2, date1, date2, hour1, hour2, feature):
        if hour1 not in self.source_data[ip1]['time'][date1] and hour2 not in self.source_data[ip2]['time'][date2]:
            result = -1
        elif hour1 not in self.source_data[ip1]['time'][date1] or hour2 not in self.source_data[ip2]['time'][date2]:
            result = -1
        else:
            classA = self.source_data[ip1]['time'][date1][hour1][feature]
            classB = self.source_data[ip2]['time'][date2][hour2][feature]
            # TODO think about one profile have feature and the second does not
            if not classA and not classB:
                result = -1
            elif not classA or not classB:
                result = -1
            else:
                [firsthist, secondhist] = self.prepare_lists_to_compare(classA, classB)
                result = np.asscalar(self.hellinger3(tuple(firsthist), tuple(secondhist)))
        return result
    def get_distance_same_hours(self,ip1, ip2, date1, date2, feature):
        result = {}
        result['bhattacharyya'] = {}
        result['chisquare'] = {}
        result['intersection'] = {}
        index = ""
        for i in range(0, 24):
            hour = "{0:0>2}".format(i)
            index = ip1 + '--' + ip2 + '--' + date1 + '--' + date2
            classA = self.source_data[ip1]['time'][date1][hour][feature]
            classB = self.source_data[ip2]['time'][date2][hour][feature]
            if not classA or not classB:
                result['bhattacharyya'] = -1
                result['chisquare'] = -1
                result['intersection'] = -1

            [firsthist, secondhist] = self.prepare_lists_to_compare(classA, classB)
            bhat = np.asscalar(self.bhattacharyya(firsthist, secondhist))
            chi = np.asarray(self.chisquare(firsthist, secondhist))
            inter = np.asarray(self.intersetciton(firsthist, secondhist))
            result['bhattacharyya'][index] = bhat
            result['chisquare'][index] = chi
            result['intersection'][index] = inter
        return result
    def get_distances_same_ip(self,ip,date,feature):
        result = {}
        result['bhattacharyya'] = {}
        result['chisquare'] = {}
        result['intersection'] = {}

        for i in range(0,23):
            hourA = "{0:0>2}".format(i)
            hourB = "{0:0>2}".format(i+1)
            if hourA not in self.source_data[ip]['time'][date] or hourB not in self.source_data[ip]['time'][date]:
                continue
            classA = self.source_data[ip]['time'][date][hourA][feature]
            classB = self.source_data[ip]['time'][date][hourB][feature]
            if not classA or not classB:
                result['bhattacharyya'][hourA + '--' + hourB] = -1
                result['chisquare'][hourA + '--' + hourB] = -1
                result['intersection'][hourA + '--' + hourB] = -1
                continue
            [firsthist,secondhist] = self.prepare_lists_to_compare(classA,classB)

            bhat = np.asscalar(self.bhattacharyya(firsthist, secondhist))
            chi = np.asarray(self.chisquare(firsthist, secondhist))
            inter = np.asarray(self.fintersetciton(firsthist, secondhist))

            result['bhattacharyya'][hourA + '--' + hourB] = bhat
            result['chisquare'][hourA + '--' + hourB] = chi
            result['intersection'][hourA + '--' + hourB] = inter
        return result

    def generate_list_of_features(self,split_classB_to_tcpudp):
        result = []
        #server not estabilished are not included because of the CTU firewall. Recincluded
        s = ['client','server']
        d = ['SourcePort', 'DestinationPort']
        f = ['TotalBytes', 'TotalPackets', 'NumberOfFlows']
        p = ['TCP','UDP']
        e = ['Established','NotEstablished']
        t = ['DictOfConnections','DictClassBnetworks']
        # for source in s:
        #     for type in t:
        #         for protocol in p:
        #             for state in e:
        #                 result.append(source + type + protocol + state)
        # for source in s:
        #     for port in d:
        #         for feature in f:
        #             for protocol in p:
        #                 for state in e:
        #                         result.append(source+port+feature+protocol+state)
        # return result

        if(split_classB_to_tcpudp):
            for source in s:
                for type in t:
                    for protocol in p:
                        if source == 'server':
                            result.append(source + type + protocol + 'Established')
                        else:
                            result.append(source + type  + protocol + 'Established')
                            result.append(source + type  + protocol + 'NotEstablished')
        else:
            result = ['clientDictClassBnetworksEstablished', 'clientDictClassBnetworksNotEstablished',
                      'serverDictClassBnetworksEstablished']
            # result = ['clientDictClassBnetworksEstablished','clientDictClassBnetworksNotEstablished']
            # result = result +['clientDictOfConnectionsEstablished','clientDictOfConnectionsNotEstablished']
            result = result + ['clientDictOfConnectionsEstablished', 'clientDictOfConnectionsNotEstablished',
                               'serverDictOfConnectionsEstablished']
        for source in s:
            for port in d:
                for feature in f:
                    for protocol in p:
                        if not (source == 'server' and port == 'SourcePort'):
                            if source == 'server':
                                result.append(source+port+feature+protocol+'Established')
                            else:
                                result.append(source+port+feature+protocol+'Established')
                                result.append(source+port+feature+protocol+'NotEstablished')
        return result


    def create_traces__dict(self,data):
        result = {}
        for key in data:
            result[key] = go.Scatter(
                x=sorted(data[key]),
                y=[data[key][x] for x in sorted(data[key])],
                name=key,
                mode='markers+lines',
                hoverlabel={'namelength': -1}

            )
        return result
    labeldict = {'bhattacharyya':'min is 0,max is 1.0, match is 0, half match is 0.5',
                 'chisquare': 'min is 0,max is 2.0, match is 0, half match is 0.67',
                 'intersection': 'min is 0,max is 1.0, match is 1, half match is 0.5',
                 'tf-idf': 'min is 0,max is 1.0, match is 1'
                }
    @staticmethod
    def load_profiles(name):
        profiles = {}
        try:
            with open(name) as data_file:
                profiles = json.load(data_file)
        except IOError:
            print('Result not found')
        return profiles