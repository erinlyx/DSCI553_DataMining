from blackbox import BlackBox
import sys, os, time
import random
import binascii


def myhashs(s):
    result = []
    user_int = int(binascii.hexlify(s.encode('utf8')), 16)
    num_hash_values = 16
    hash_values = create_hash_values(num_hash_values)
    for val in hash_values:
        result.append(((val[0] * user_int + val[1]) % val[2]) % (2 ** num_hash_values))
    return result


def create_hash_values(n):
    a = [10, 31, 332, 993, 430, 568, 476, 380, 991, 883, 517, 552, 830, 805, 775, 726, 527]
    b = [451, 121, 572, 403, 97, 428, 621, 786, 790, 335, 970, 88, 811, 71, 991, 601, 842]
    p = [4433527, 712289, 6649385, 6475799, 2007533, 4416863, 8564383, 5955983, 380121, 1127229, 738500, 6623519, 9440624,
         668655, 2632966, 1674740, 9491576]    
    hash_values = []
    i = 0
    while i < n:
        hash_values.append([a[i], b[i], p[i]])
        i += 1
    return hash_values


def count_trailing_zeroes(hash_value):
    binary = bin(hash_value)[2:]
    return len(binary) - len(binary.rstrip('0'))


def flajolet_martin(stream_users, num_i, num_groups):
    global sum_actual, sum_estimates
    estimates = []
    all_hash_values = []
    for user_id in stream_users:
        hash_values = myhashs(user_id)
        all_hash_values.append(hash_values)
        
    i = 0
    while i < num_hash_values:
        max_traling_zeroes = -1
        for hash_values in all_hash_values:
            trailing_zeros = count_trailing_zeroes(hash_values[i])
            if trailing_zeros <= max_traling_zeroes:
                continue
            else:
                max_traling_zeroes = trailing_zeros
        estimates.append(2 ** max_traling_zeroes)
        i += 1
        
    avg_estimates = []
    j = 0
    for i in range (num_groups, num_hash_values):
        sub = estimates[j:i]
        avg_estimates.append(sum(sub)/len(sub))
        j = i
        i += num_groups
    
    avg_estimates.sort()
    mid = len(avg_estimates) // 2
    res = (avg_estimates[mid] + avg_estimates[~mid]) / 2
    sum_estimates += int(res)
    
    sum_actual += len(set(stream_users))
    output_file.write(str(num_i) + ',' + str(len(set(stream_users))) + ',' + str(int(res)) + '\n')


if __name__ == '__main__':
    t1 = time.time()
    sum_actual = 0
    sum_estimates = 0

    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_asked = int(sys.argv[3])
    output_file = sys.argv[4]

    num_groups = 4
    num_hash_values = 16

    output_file = open(output_file, 'w')
    output_file.write('Time,Ground Truth,Estimation'+'\n')
    
    bx = BlackBox()
    for i in range(num_asked):
        stream_users = bx.ask(input_file, stream_size)
        flajolet_martin(stream_users, i, num_groups)
    output_file.close()

    t2 = time.time()
    print(t2-t1)
    # print(float(sum_estimates/ sum_actual))