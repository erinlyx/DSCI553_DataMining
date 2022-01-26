import os
import sys
import json
from pyspark import SparkContext
import numpy as np
import pandas as pd
import xgboost as xgb
import time

def get_features(test_set, user_data, bus_data, test_input = True):
    if (test_input):
        user_id, bus_id, rating = test_set[0], test_set[1], -1.0
    else:
        user_id, bus_id, rating = test_set[0], test_set[1], test_set[2]
    # new business
    if bus_id not in bus_data.keys():
        if (user_data.get(user_id)) is not None:
            user_reviewcount, user_rating = user_data[user_id]
            return [user_id, bus_id, float(user_reviewcount), float(user_rating), bus_avg_reviewcount, bus_avg_rating, None]
        else:
            # if user is new too
            return [user_id, bus_id, user_avg_reviewcount, user_avg_rating, bus_avg_reviewcount, bus_avg_rating, None]
    # existing business
    else:
        bus_reviewcount, bus_rating = bus_data[bus_id]
        # existing user
        if user_data.get(user_id) is not None:
            user_reviewcount, user_rating = user_data[user_id]
            buses_lst = list(user_data.get(user_id))
            # no such user in test set
            if len(buses_lst) == 0:
                return [user_id, bus_id, float(user_reviewcount), float(user_rating), float(bus_reviewcount), float(bus_rating), None]
            else:
                return [user_id, bus_id, float(user_reviewcount), float(user_rating), float(bus_reviewcount), float(bus_rating), rating]
        # new user
        else:
            return [user_id, bus_id, user_avg_reviewcount, user_avg_rating, float(bus_reviewcount), float(bus_rating), None]


if __name__ == '__main__':

    input_folder = sys.argv[1]
    input_test = sys.argv[2]
    output_file = sys.argv[3]

    input_train = input_folder + '/yelp_train.csv'
    # input_test = 'yelp_val.csv'

    t1 = time.time()

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')

    # read and load train/test set
    train_raw = sc.textFile(input_train)
    test_raw = sc.textFile(input_test)

    train_top_row = train_raw.first()
    test_top_row = test_raw.first()

    train_body = train_raw.filter(lambda row:row != train_top_row).map(lambda x: x.split(','))
    test_body = test_raw.filter(lambda row:row != test_top_row).map(lambda x: x.split(','))

    # introduce the new user and business json data
    # read and load user/business info file
    input_user = input_folder + '/user.json'
    input_bus = input_folder + '/business.json'

    user_raw = sc.textFile(input_user)
    bus_raw = sc.textFile(input_bus)

    user_rdd = user_raw.map(lambda x:json.loads(x)).map(lambda x:((x['user_id'], (x['review_count'], x['average_stars']))))
    user_map = user_rdd.collectAsMap()
    user_avg_rating = user_rdd.map(lambda x: x[1][1]).mean()
    user_avg_reviewcount = user_rdd.map(lambda x: x[1][0]).mean()

    bus_rdd =bus_raw.map(lambda x:json.loads(x)).map(lambda x:((x['business_id'], (x['review_count'], x['stars']))))
    bus_map = bus_rdd.collectAsMap()
    bus_avg_rating = bus_rdd.map(lambda x: x[1][1]).mean()
    bus_avg_reviewcount = bus_rdd.map(lambda x: x[1][0]).mean()


    # prepare training data
    train_set = train_body.map(lambda x: get_features(x, user_map, bus_map, False)).collect()
    train_matrix = np.array(train_set)

    x_train = train_matrix[:, 2:-1]
    y_train = train_matrix[:, -1]
    x_train, y_train = np.array(x_train, dtype = 'float'), np.array(y_train, dtype = 'float')

    # train the model
    xgb_model = xgb.XGBRegressor(objective = 'reg:linear')
    xgb_model.fit(x_train,y_train)

    # prepare testing data
    test_set = test_body.map(lambda x:get_features(x, user_map, bus_map)).collect()
    test_matrix = np.array(test_set)

    x_test = test_matrix[:, 2:-1]
    y_test = test_matrix[:, -1]
    x_test, y_test = np.array(x_test, dtype = 'float'), np.array(y_test, dtype = 'float')

    # make prediction
    prediction = xgb_model.predict(x_test)


    # compile result
    test_df = pd.read_csv(input_test)
    result = pd.DataFrame()
    result['user_id'] = test_df.user_id.values
    result['business_id'] = test_df.business_id.values
    result['prediction'] = prediction

    # write to file
    result.to_csv(output_file, header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')

    t2 = time.time()
    print ('Duration: ', t2-t1)

    # from sklearn.metrics import mean_squared_error
    # print('rmse: ', np.sqrt(mean_squared_error(test_df.stars.values, prediction)))