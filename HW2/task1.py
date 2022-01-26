import sys
import json
from pyspark import SparkContext
import time
import math
from itertools import combinations


def get_candidates(rdd,cutoff,size):
    partition = list(rdd)
    p = len(partition) / size
    minP = math.ceil(p*cutoff)
    freq_items =[]

    frequency = {}
    baskets = []
    for par in partition:
        baskets.append(par)
        for i in par:
            if i in frequency:
                frequency[i] += 1
            else:
                frequency[i] = 1

    # get all singletons
    freq_singletons = sorted([k for k, v in frequency.items() if v >= minP])
    for i in freq_singletons:
        freq_items.append([(i,)])
        
    # get other candidates with greater length
    k = 2
    while True:
        freq_new = {}
        for par in partition:
        #         baskets.append(par)
            par = sorted(set(par).intersection(set(freq_singletons)))
            for i in combinations(par,k): #get all possible combimation
                i = tuple(i)
                if i in freq_new:
                    freq_new[i] += 1
                else:
                    freq_new[i] = 1
        freq_more = sorted([k for k, v in freq_new.items() if v >= minP])
        if freq_more != []:
            for i in freq_more:
                freq_items.append([i])
            # update freq_singletons
            freq_singletons = set()
            for i in freq_more:
                freq_singletons = freq_singletons|set(i)
            k += 1
        else:
            break
    return freq_items



def check_itemsets(rdd, candidates):
    occurrence = {}
    for r in rdd:
        for c in candidates:
            # make sure the candidate is truely frequent
            if c in r or all([item in r for item in c]):
                if c in occurrence.keys():
                    occurrence[c] += 1
                else:
                    occurrence[c] = 1

    counting = [(k,v) for k,v in occurrence.items()]
    return counting



if __name__ == '__main__':
    t1 = time.time()

    case_num = sys.argv[1]
    cutoff = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    sc = SparkContext('local[*]', 'task1')

    rdd0 = sc.textFile(input_file)
    top_row = rdd0.first()
    rdd = rdd0.filter(lambda row: row != top_row)

    # creating baskets based on case number input (1or2)
    if case_num == '1':
        rddN = rdd.map(lambda line: (line.split(',')[0], line.split(',')[1]))
    elif case_num == '2':
        rddN = rdd.map(lambda line: (line.split(',')[1], line.split(',')[0]))
    else:
        print('invalid case: please input 1 or 2')
    # compile lists of unique items within baskets
    rddN = rddN.groupByKey().map(lambda r: list(set(r[1])))

    size = rddN.count()

    # Phase 1
    candidates = rddN.mapPartitions(lambda partition: get_candidates(partition, cutoff, size)) \
    .flatMap(lambda r:r).distinct().sortBy(lambda r:(len(r),r)) \
    .collect()

    # Phase 2
    freq_itemsets = rddN.mapPartitions(lambda partition: check_itemsets(partition, candidates)) \
    .reduceByKey(lambda a,b:a+b).filter(lambda r:r[1] >= cutoff) \
    .map(lambda r:r[0]).sortBy(lambda r:(len(r),r)) \
    .collect()

    output_file = open(output_file, 'w')
    output = 'Candidates:'+'\n'
    tier = 1
    for c in candidates:
        if len(c) == 1:
            output += (str(c).replace(',', '')+ ',')
        elif len(c) == tier:
            output += (str(c)+ ',')
        else:
            output += '\n\n'
            output += (str(c)+ ',')
            tier = len(c)

    output += '\n\n' + 'Frequent Itemsets:\n'
    for c in freq_itemsets:
        if len(c) == 1:
            output += (str(c).replace(',', '')+ ',')
        elif len(c) == tier:
            output += (str(c)+ ',')
        else:
            output += '\n\n'
            output += (str(c)+ ',')
            tier = len(c)
    output = output.replace(',\n\n','\n\n')[:-1]
    output_file.write(output)
    output_file.close()

    t2 = time.time()
    print('Duration: ', t2-t1)