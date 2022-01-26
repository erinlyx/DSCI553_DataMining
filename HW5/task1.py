from blackbox import BlackBox
import binascii
import sys, os, time
import random


def myhashs(s):
    result = []
    user_int = int(binascii.hexlify(s.encode('utf8')), 16)
    hash_values = create_hash_values(6)
    for val in hash_values:
        result.append((val[0] * user_int + val[1]) % 69997)
    return result


def create_hash_values(n):
    # previously we use random.sample(range(1, 1000), n) to generate a & b
    # and similarly for p but check again if it's prime
    a = [1, 5, 12, 66, 89, 106]
    b = [2, 8, 10, 34, 23, 46]
    p = [12917, 33863, 72671, 113623, 153359, 196613]
    hash_values = []
    i = 0
    while i < n:
        hash_values.append([a[i], b[i], p[i]])
        i += 1
    return hash_values


def bloom_filter(stream_users, num_i):
    global global_user_set
    global filter_bit_array
    false_pos = 0
    true_neg = 0
    
    for user_id in stream_users:
        counter = 0
        hash_values = myhashs(user_id)
        for hash_val in hash_values:
            if filter_bit_array[hash_val] != 1:
                filter_bit_array[hash_val] = 1
            else:
                counter += 1
        if user_id not in global_user_set:
            if counter != len(hash_values):
                true_neg += 1  
            else:
                false_pos += 1
        global_user_set.add(user_id)
    
    denominator = false_pos + true_neg
    if denominator == 0 and false_pos == 0 and true_neg == 0:
        fpr = 0.0
    else:
        fpr = float(false_pos / float(denominator))
    output_file.write(str(num_i) + ',' + str(fpr) + '\n')


if __name__ == '__main__':
    t1 = time.time()
    filter_bit_array = [0] * 69997

    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_asked = int(sys.argv[3])
    output_file = sys.argv[4]

    global_user_set = set()
    output_file = open(output_file, 'w')
    output_file.write('Time,FPR'+'\n')

    bx = BlackBox()
    for i in range(num_asked):
        stream_users = bx.ask(input_file, stream_size)
        bloom_filter(stream_users, i)
    output_file.close()
    t2 = time.time()
    # print(t2-t1)