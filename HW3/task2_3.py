import sys
from pyspark import SparkContext
import os
import math
import time

import json
import numpy as np
import pandas as pd
import xgboost as xgb


# needed functions
def get_features(entry_set, user_data, bus_data, test_input=True):
    if not (test_input):
        user_id, bus_id, rating = entry_set[0], entry_set[1], entry_set[2]
    else:
        user_id, bus_id, rating = entry_set[0], entry_set[1], -1.0
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



def get_pearson_coef(neighbor_id, users_lst, bus_rating, avg_bus_rating):
    all_bus_ratings = []
    all_neighbor_ratings = []
    neighbor_bus_rating = bus_userRating.get(neighbor_id)
    avg_neighbor_rating = bus_avgRating.get(neighbor_id)
    check_neighbor_id(all_bus_ratings, all_neighbor_ratings, bus_rating, neighbor_bus_rating, users_lst)
    if len(all_bus_ratings) == 0:
        # default value
        pearson_coef = float(avg_bus_rating/avg_neighbor_rating)
    else:
        numerator, denominator_bus, denominator_neighbor = 0, 0, 0
        denominator_bus, denominator_neighbor, numerator = update_params(all_bus_ratings, all_neighbor_ratings,
                                                                         avg_bus_rating, avg_neighbor_rating,
                                                                         denominator_bus, denominator_neighbor,
                                                                         numerator)
        denominator = math.sqrt(denominator_bus * denominator_neighbor)
        if denominator != 0:
            pearson_coef = numerator/denominator
        else:
            if numerator != 0:
                pearson_coef = -1
            else:
                pearson_coef = 1
    return pearson_coef


def update_params(all_bus_ratings, all_neighbor_ratings, avg_bus_rating, avg_neighbor_rating, denominator_bus,
                  denominator_neighbor, numerator):
    i = 0
    while i < len(all_bus_ratings):
    # for i in range(0, len(all_bus_ratings)):
        numerator += (all_bus_ratings[i] - avg_bus_rating) * (all_neighbor_ratings[i] - avg_neighbor_rating)
        denominator_bus += (all_bus_ratings[i] - avg_bus_rating) ** 2
        denominator_neighbor += (all_neighbor_ratings[i] - avg_neighbor_rating) ** 2
        i += 1
    return denominator_bus, denominator_neighbor, numerator


def check_neighbor_id(all_bus_ratings, all_neighbor_ratings, bus_rating, neighbor_bus_rating, users_lst):
    for current_id in users_lst:
        # if exists as a key
        if neighbor_bus_rating.get(current_id):
            current_rating = bus_rating.get(current_id)
            all_bus_ratings.append(current_rating)
            neighbor_rating = neighbor_bus_rating.get(current_id)
            all_neighbor_ratings.append(neighbor_rating)


def get_prediction(pearson_rating_lst, avg_default):
    pred_weight_sum, pearson_coef_sum = 0, 0
    if len(pearson_rating_lst) == 0:
        # return default in the case of empty list (no valid pearson coef)
        return avg_default
    neighborhood = min(len(pearson_rating_lst), 50)
    pearson_coef_sum, pred_weight_sum = sum_up_coef_and_weights(neighborhood, pearson_coef_sum, pearson_rating_lst,
                                                                pred_weight_sum)
    predicted_val = pred_weight_sum/pearson_coef_sum
    return min(max(predicted_val,0),5)


def sum_up_coef_and_weights(neighborhood, pearson_coef_sum, pearson_rating_lst, pred_weight_sum):
    i = 0
    while i < neighborhood:
        pred_weight_sum += pearson_rating_lst[i][0] * pearson_rating_lst[i][1]
        pearson_coef_sum += abs(pearson_rating_lst[i][0])
        i += 1
    return pearson_coef_sum, pred_weight_sum


def compute_from_ratings(avg_bus_rating, bus_rating, buses_lst, user, users_lst):
    pearson_rating_lst = []
    for i in range(len(buses_lst)):
        current_neighbor_rating = bus_userRating.get(buses_lst[i]).get(user)
        pearson_coef = get_pearson_coef(buses_lst[i], users_lst, bus_rating, avg_bus_rating)
        if pearson_coef <= 0:
            continue
        else:
            if pearson_coef <= 1:
                pearson_coef += 0
            else:
                pearson_coef = 1/pearson_coef
            pearson_rating_lst.append((pearson_coef, current_neighbor_rating))
    pearson_rating_lst.sort(key=lambda x: x[0], reverse=True)        
    return pearson_rating_lst


def item_based_withPearson(test_set):
    user, bus = test_set[0], test_set[1]

    if bus not in bus_userRating.keys():
        # use default for new business
        if (user_busRating.get(user)) is not None:
            return user, bus, str(user_avgRating.get(user))
        else:
            # if user is new too
            return user, bus, '3.5'  # changing this value doesn't seem to improve result

    else:
        bus_rating = bus_userRating.get(bus)
        avg_bus_rating = bus_avgRating.get(bus)
        users_lst = list(bus_userRating.get(bus))

        if user_busRating.get(user) is not None:
            buses_lst = list(user_busRating.get(user))
            # no such user in test set
            if len(buses_lst) == 0:
                return user, bus, str(avg_bus_rating)
            # user has recorded ratings
            else:
                pearson_rating_lst = compute_from_ratings(avg_bus_rating, bus_rating, buses_lst, user, users_lst)
                predicted_val = 0
                predicted_val = get_prediction(pearson_rating_lst, (user_avgRating.get(user)+avg_bus_rating)/2)
                return user, bus, str(min(max(predicted_val,0),5))
        else:
            # use default val for new user
            return user, bus, str(avg_bus_rating)



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


## item-based prediction
    # create the user-businesses&rating dict
    bus_per_user = train_body.map(lambda x:((x[0]), ((x[1]), float(x[2])))).groupByKey().mapValues(dict)
    user_busRating = {i:j for i,j in bus_per_user.collect()}
    # create user-average rating dict
    avgRating_per_user = train_body.map(lambda x:(x[0], float(x[2]))).groupByKey().mapValues(lambda x:sum(x)/len(x))
    user_avgRating = {i:j for i,j in avgRating_per_user.collect()}
    # create the business-user&rating dict
    users_per_bus = train_body.map(lambda x:((x[1]), ((x[0]), float(x[2])))).groupByKey().mapValues(dict)
    bus_userRating = {i:j for i,j in users_per_bus.collect()}
    # create business-average rating dict
    avgRating_per_bus = train_body.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x)/len(x))
    bus_avgRating = {i:j for i,j in avgRating_per_bus.collect()}

    # create test set
    testing_rdd = test_body.sortBy(lambda x:((x[0]), (x[1]))).persist()
    # make predictions
    item_based_prediction = testing_rdd.map(item_based_withPearson).map(lambda x:((x[0], x[1]), float(x[2])))

## model-based prediction
    user_rdd = user_raw.map(lambda x:json.loads(x)).map(lambda x:((x['user_id'], (x['review_count'], x['average_stars']))))
    user_map = user_rdd.collectAsMap()
    user_avg_rating = user_rdd.map(lambda x: x[1][1]).mean()
    user_avg_reviewcount = user_rdd.map(lambda x: x[1][0]).mean()

    bus_rdd =bus_raw.map(lambda x:json.loads(x)).map(lambda x:((x['business_id'], (x['review_count'], x['stars']))))
    bus_map = bus_rdd.collectAsMap()
    bus_avg_rating = bus_rdd.map(lambda x: x[1][1]).mean()
    bus_avg_reviewcount = bus_rdd.map(lambda x: x[1][0]).mean()

    train_set = train_body.map(lambda x: get_features(x, user_map, bus_map, False)).collect()
    train_matrix = np.array(train_set)

    x_train = train_matrix[:, 2:-1]
    y_train = train_matrix[:, -1]
    x_train, y_train = np.array(x_train, dtype = 'float'), np.array(y_train, dtype = 'float')

    # train the model
    xgb_model = xgb.XGBRegressor(learning_rate=0.3)
    xgb_model.fit(x_train,y_train)

    test_set = test_body.map(lambda x:get_features(x, user_map, bus_map)).collect()
    test_matrix = np.array(test_set)

    x_test = test_matrix[:, 2:-1]
    y_test = test_matrix[:, -1]
    x_test, y_test = np.array(x_test, dtype = 'float'), np.array(y_test, dtype = 'float')

    # make prediction
    predicted_modelB = xgb_model.predict(x_test)

    test_df = pd.read_csv(input_test)
    result = pd.DataFrame()
    result['user_id'] = test_df.user_id.values
    result['business_id'] = test_df.business_id.values
    result['prediction'] = predicted_modelB


# write temp file
    result.to_csv('temp.csv', header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')
## compile the two
    temp_rdd = sc.textFile('temp.csv')
    temp_top_row = temp_rdd.first()
    temp_data = temp_rdd.filter(lambda x:x != temp_top_row).map(lambda x: x.split(','))
    model_based_prediction = temp_data.map(lambda x:(((x[0]), (x[1])), float(x[2])))

    output_hybrid = item_based_prediction.join(model_based_prediction).map(lambda x: ((x[0]), float((x[1][0]*0.02 + x[1][1]*0.98)))).collect()

## write to file
    output_file = open(output_file, 'w')
    output_file.write('user_id, business_id, prediction\n')
    for i in range(len(output_hybrid)):
        output_file.write(str(output_hybrid[i][0][0]) + ',' + str(output_hybrid[i][0][1]) + ',' + str(output_hybrid[i][1]) + '\n')

    output_file.close()

    # t2 = time.time()
    # time_file = open('time4.txt', 'w')
    # time_file.write(str(t2-t1))
    # time_file.close()