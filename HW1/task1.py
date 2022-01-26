import sys
import json
from pyspark import SparkContext

def explore_review(input_file):  

    output = {}
    sc =SparkContext()
    rdd = sc.textFile(input_file).coalesce(8).map(json.loads) \
        .map(lambda r: [r['date'],
                        r['user_id'],
                        r['business_id']]).cache()

    # a.count total number of reivew
    output['n_review']=rdd.count()

    # b.count review in 2018
    output['n_review_2018'] = rdd.map(lambda r: r[0]).filter(lambda r: "2018" in r).count()

    # c.find number of distinct users
    d_users = rdd.map(lambda r: r[1]).groupBy(lambda r: r).cache()
    output['n_user'] = d_users.count()

    # d. find top 10 users with most reviews written (and counts)
    def f(x): return len(x)
    output["top10_user"] = d_users.mapValues(f).sortBy(lambda r: [-r[1], r[0]]).take(10)

    # e. find number of distinct businesses
    d_bus = rdd.map(lambda r: r[2]).groupBy(lambda r: r).cache()
    output['n_business'] = d_bus.count()

    # f. find top 10 businesses with most reviews written about(and counts)
    output["top10_business"] = d_bus.mapValues(f).sortBy(lambda r: [-r[1], r[0]]).take(10)
    
    return output

if __name__ == '__main__':
    input_file = sys.argv[1]
    result = explore_review(input_file)

    # creating json file
    output_file = open(sys.argv[2], 'w') 
    json.dump(result, output_file) 
    output_file.close()