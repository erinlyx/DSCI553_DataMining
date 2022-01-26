import sys
import json
from pyspark import SparkContext
import time

def compare_methods(input_review, input_bus):
    sc = SparkContext()
    # input1 = 'test_review.json'
    # input2 = 'business.json'

    # define partition func
    def partByCity(city):
        return hash(city)

    # get the loading time for both methods
    t1Load = time.time()

    busNstars = sc.textFile(input_review).map(json.loads).map(lambda r: (r['business_id'], r['stars']))
    cityNbus = sc.textFile(input_bus).map(json.loads).map(lambda r: (r['business_id'], r['city']))

    t2Load = time.time()
    load_time = t2Load - t1Load

    ## get the (remaining) execution time for Python method
    t1Python = time.time()

    combined_py = cityNbus.join(busNstars).map(lambda r:(r[1][0], r[1][1])).partitionBy(None, partByCity) \
    .groupByKey().map(lambda r:(r[0],sum(r[1])/len(r[1]))).collect()
    # sort in Python
    sorted_py = sorted(combined_py, key = lambda r: (-r[1], r[0]))
    final_py = sorted_py[:10]

    t2Python = time.time()

    ## get the (remaining) execution time for Spark method
    t1Spark = time.time()

    combined_spark = cityNbus.join(busNstars).map(lambda r:(r[1][0], r[1][1])).partitionBy(None, partByCity) \
    .groupByKey().map(lambda r:(r[0],sum(r[1])/len(r[1]))) \
    .sortBy(lambda r: [-r[1], r[0]]).take(10) # sort in Spark

    t2Spark = time.time()

    m1 = t2Python-t2Python+load_time
    m2 = t2Spark-t1Spark+load_time
    output = {'m1':m1, 'm2':m2,\
        'reason':'XXXXXX'}

    return sorted_py, output



if __name__ == '__main__':
    input_review = sys.argv[1]
    input_bus = sys.argv[2]
    resultA, resultB = compare_methods(input_review,input_bus)

    # creating text file for 3A
    output_file_A = open(sys.argv[3], 'w')
    output_file_A.write('city,stars'+'\n')
    for i in resultA:
        output_file_A.write(','.join(str(s) for s in i) + '\n')
    output_file_A.close()

    # creating json file for 3B
    output_file_B = open(sys.argv[4], 'w') 
    json.dump(resultB, output_file_B) 
    output_file_B.close()