import numpy as np
import json
import scipy

from scipy.sparse import csr_matrix

from tools.supportjupyterfunctions import SupportFunctions
from sklearn.feature_extraction.text import TfidfTransformer

def load_profiles_ifidcust(name):
    profiles = {}
    try:
        with open(name) as data_file:
            profiles = json.load(data_file)
    except IOError:
        print('Result not found')
    return profiles
sp = SupportFunctions(SupportFunctions.load_profiles("fromfrantisek.json"),True)

def generate_data_and_indexes(feature):
    data = []
    index_dict = {}
    index = 0
    for ip in sp.source_data:
        index_dict[ip] = {}
        for date in sp.source_data[ip]['time']: 
            if date.split('/')[1] == '09':
                index_dict[ip][date] = {}
                for hour in sp.source_data[ip]['time'][date]:
                    bag=sp.source_data[ip]['time'][date][hour][feature]
                    data.append(bag)
                    index_dict[ip][date][hour] = index
                    index+=1
    return data,index_dict
def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc])
    return lexicon
def dist_vectors(vec1, vec2):
    return np.vdot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
def generate_idf_feature_matrix(feature):
    mydoclist,indexes =  generate_data_and_indexes(feature)
    #print "generated data and indexes"
    vocabulary = build_lexicon(mydoclist)
    vocabulary = list(vocabulary)
    #print "vocabulary build"
    row = []
    col = []
    data = []
    row_idx = 0
    for doc in mydoclist:
        for col_idx in range(0,len(vocabulary)):
            if vocabulary[col_idx] in doc:
                row.append(row_idx)
                col.append(col_idx)
                data.append(doc[vocabulary[col_idx]])
        row_idx+=1
    doc_term_matrix = csr_matrix((data, (row, col)), shape=(len(mydoclist), len(vocabulary)))
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(doc_term_matrix)

    tf_idf_matrix = tfidf.transform(doc_term_matrix)
    return tf_idf_matrix,indexes
def generate_all_idf_matrixes():
    idf_matrixes_dict = {}
    idf_matrixes_dict_index = {}
    for feature in sp.features:
        print (feature)
        tf_idf,index_dict = generate_idf_feature_matrix(feature)
        idf_matrixes_dict[feature] = tf_idf
        idf_matrixes_dict_index[feature] = index_dict
    return idf_matrixes_dict,idf_matrixes_dict_index
def calculate_idf(idf_matrixes_dict,idf_matrixes_dict_index,ip1,ip2,date1,date2,hour1,hour2,feature):
    index1 = idf_matrixes_dict_index[feature][ip1][date1][hour1]
    index2 = idf_matrixes_dict_index[feature][ip2][date2][hour2]
    distance = dist_vectors(idf_matrixes_dict[feature][index1].toarray(), idf_matrixes_dict[feature][index2].toarray())
    if math.isnan(distance):
        return -1
    else:
        return distance



def load_sparse_csr(filename):
    return scipy.sparse.load_npz(filename)
def load_tf_idf_matrixes():
    idf_matrixes_dict = {}
    idf_matrixes_dict_index = {}
    for feature in sp.features:
        name = 'sparsefeatures/' + feature + '.npz'
        idf_matrixes_dict[feature] = load_sparse_csr(name)
    with open ('sparsefeatures/idf_matrixes_dict_index.txt') as data_file:
        idf_matrixes_dict_index = json.load (data_file)
    return idf_matrixes_dict,idf_matrixes_dict_index
def dist_vectors(vec1, vec2):
    if (not vec1.any()) or (not vec2.any()): # if one of the vector is all zeros
        return -1
    else:
        return np.vdot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#inspect the if in original function and here
import math
def calculate_idf(idf_matrixes_dict,idf_matrixes_dict_index,ip1,ip2,date1,date2,hour1,hour2,feature):
    if hour1 not in sp.source_data[ip1]['time'][date1] and hour2 not in sp.source_data[ip2]['time'][date2]:
        distance = -1
    elif hour1 not in sp.source_data[ip1]['time'][date1] or hour2 not in sp.source_data[ip2]['time'][date2]:
        distance = -1
    else:
        index1 = idf_matrixes_dict_index[feature][ip1][date1][hour1]
        index2 = idf_matrixes_dict_index[feature][ip2][date2][hour2]
        distance = dist_vectors(idf_matrixes_dict[feature][index1].toarray(), idf_matrixes_dict[feature][index2].toarray())
        return distance
        #if math.isnan(distance):
        #    return -1
        #else:
        #    return distance
