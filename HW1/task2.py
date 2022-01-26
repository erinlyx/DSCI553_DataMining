import sys
import json
from pyspark import SparkContext
import time

def change_partition(input_file, input_num_partition):
    output = {}
    sc = SparkContext('local[*]', 'task2')
    # input_file = 'test_review.json'

    # load the data
    rdd = sc.textFile(input_file).map(lambda r: (json.loads(r)['business_id'], 1)).cache()

    # get performance for default
    t1Default = time.time()
    task_default = rdd.reduceByKey(lambda bus, count : bus + count).sortBy(lambda r: [-r[1], r[0]]).take(10)
    t2Default = time.time()

    partition_size_default = rdd.glom().map(len).collect()
    gap_default = t2Default - t1Default
    output['default'] = {'n_partition': rdd.getNumPartitions(), \
                        'n_items': partition_size_default, 'exe_time': gap_default}

    # input_num_partition = 2
    # get performance for customized
    input_num_partition = int(input_num_partition)

    # now we want to allocate one partition for the same business_id
    def partByBus(business_id):
        return hash(business_id) % input_num_partition
    newrdd = rdd.partitionBy(input_num_partition, partByBus)

    t1Custom = time.time()
    task_custom = newrdd.reduceByKey(lambda bus, count : bus + count).sortBy(lambda r: [-r[1], r[0]]).take(10)
    t2Custom = time.time()

    partition_size_custom = newrdd.glom().map(len).collect()
    gap_custom = t2Custom - t1Custom
    output['customized'] = {'n_partition': newrdd.getNumPartitions(), \
                        'n_items': partition_size_custom, 'exe_time': gap_custom}
    return output


if __name__ == '__main__':
    input_file = sys.argv[1]
    input_num_partition = sys.argv[3]
    result = change_partition(input_file,input_num_partition)

    # creating json file
    output_file = open(sys.argv[2], 'w') 
    json.dump(result, output_file) 
    output_file.close()
