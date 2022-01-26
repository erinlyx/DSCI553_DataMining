from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import time
import os
import sys


if __name__ == '__main__':
    t1 = time.time()
    # input_file = 'ub_sample_data.csv'
    # threshold =7

    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    sc = SparkContext('local[*]', 'task1')
    sqlContext = SQLContext(sc)
    sc.setLogLevel('ERROR')


    # load and clean file
    raw = sc.textFile(input_file)
    top_row = raw.first()
    body = raw.filter(lambda row:row != top_row).map(lambda x: x.split(','))

    user_bus = body.map(lambda x: (x[0], x[1])).groupByKey() \
    .map(lambda x: (x[0], list(set(x[1])))).collect()


    # filter out pairs with unqualified edges
    vertices, edges = set(), set()
    for c1 in user_bus:
        for c2 in user_bus:
            if c1[0] == c2[0]:
                continue
            else:
                if not len(set(c1[1]) & set(c2[1])) < threshold:
                    vertices.add(c1[0])
                    vertices.add(c2[0])
                    edges.add(tuple((c1[0], c2[0])))

    # create vertices and edges
    vertices = sc.parallelize(list(vertices)).map(lambda x: (x, )).toDF(['id'])
    edges = sqlContext.createDataFrame(list(edges), ['src', 'dst'])

    # construct graph
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    comm = result.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list) \
        .map(lambda x: sorted(x[1])) \
        .sortBy(lambda x: (len(x), x[0])).collect()
    
    # write file
    output_file = open(output_file, 'w')
    for c in comm:
        output_file.write(str(c)[1: -1] + '\n')
    output_file.close()

    t2 = time.time()
    # print('Duration: ', t2-t1)