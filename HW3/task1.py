import sys
from pyspark import SparkContext
import os
import time
import random
from itertools import combinations


# needed functions
def get_sig_matrix(x, hash_vals, m):
    hashvals = []
    a = hash_vals[0]
    b = hash_vals[1]
#     return min([(a*x + b) % m for x in x[1]])
    for i in x[1]:
        product = (a*i + b) % m
        hashvals.append(product)
    return min(hashvals)


def create_hash_vals(size):
    a = random.sample(range(1, 1200), size)
    b = random.sample(range(1, 1200), size)
    hash_vals = []
    for i in range(size):
        hash_vals.append([a[i], b[i]])
    return hash_vals


def create_cand_group(signature):
    bins = []
    for i in range(Bands):
        bins.append(((i, tuple(signature[1][i * Rows:(i + 1) * Rows])), signature[0]))
    return bins

def create_cand_pairs(cand_group):
    return combinations(cand_group[1], 2)

def jaccard_sim(pair, matrix):
    a = set(matrix[pair[0]])
    b = set(matrix[pair[1]])
    return pair[0], pair[1], len(a&b) / len(a|b) # intersection / union



if __name__ == '__main__':
    input_file= sys.argv[1]
    output_file = sys.argv[2]

    t1 = time.time()

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')

    # read and load file
    # input_file = 'yelp_train.csv'
    raw = sc.textFile(input_file)
    top_row = raw.first()
    body = raw.filter(lambda row: row != top_row).map(lambda x: x.split(','))

    # get unique users
    users = body.map(lambda x: x[0]).distinct().sortBy(lambda x: x[0]).collect()
    # get count of unique users
    num_users = len(users)

    # generate a user-index dic
    users_dict = {u: i for i, u in enumerate(users)}

    # get groupings of users that rated each business
    users_perbus = body.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)

    # generate characteristic matrix
    matrix = body.map(lambda x: (x[1], users_dict[x[0]])).groupByKey().map(lambda x: (x[0], list(x[1]))).sortBy(lambda x: x[0])
    matrix_dict = {i: j for i, j in matrix.collect()}


    # select good combination of b and r
    Bands = 40
    hash_vals = create_hash_vals(80)
    Rows = int(len(hash_vals)/Bands)

    # create signatures
    sig_matrix = matrix.map(lambda x: (x[0], [get_sig_matrix(x, hashval_lst, num_users) for hashval_lst in hash_vals]))

    # get candidate pairs and compute Jaccard similarity
    groups = sig_matrix.flatMap(create_cand_group).groupByKey().filter(lambda x: len(x[1]) > 1)
    pairs = groups.flatMap(create_cand_pairs).distinct()
    result = pairs.map(lambda x: jaccard_sim((x[0], x[1]), matrix_dict)).filter(lambda x:x[2] >= 0.5)
    
    sorted_result = result.sortBy(lambda x: (x[0], x[1])) #sort

    # create file
    output_file = open(output_file, 'w')
    output_file.write('business_id_1, business_id_2, similarity\n')
    for l in sorted_result.collect():
        output_file.write(str(l[0]) + ',' + str(l[1]) + ',' + str(l[2]) + '\n')

    output_file.close()
    t2 = time.time()

    
    # print ('Duration: ', t2-t1)
    # time_file = open('time.txt', 'w')
    # time_file.write(str(t2-t1))
    # time_file.close()