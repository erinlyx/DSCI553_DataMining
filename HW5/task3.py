import sys, os, time
import random
import binascii
from blackbox import BlackBox


def reservoir_sampling(user_stream, num_i):
    global memory_limit
    global reservoir_lst, seq_num

    for user in user_stream:
        seq_num += 1
        
        if (len(reservoir_lst) >= memory_limit):
            probability = random.random()
            if (probability < 100/seq_num):
                idx_replace = random.randint(0,99)
                reservoir_lst[idx_replace] = user
            reservoir_lst.append(user)
            
        else:
            reservoir_lst.append(user)
    if (seq_num != 0 and seq_num % 100 == 0):
        # print(seq_num, reservoir_lst[0], reservoir_lst[20], reservoir_lst[40], reservoir_lst[60], reservoir_lst[80])
        output_file.write(str(seq_num) + ',' + str(reservoir_lst[0]) + ',' + str(reservoir_lst[20]) + ',' + str(reservoir_lst[40]) + ',' + str(reservoir_lst[60]) + ',' + str(reservoir_lst[80]) + '\n')



if __name__ == '__main__':

	t1 = time.time()

	input_file = sys.argv[1]
	stream_size = int(sys.argv[2])
	num_asked = int(sys.argv[3])
	output_file = sys.argv[4]

	# input_file = 'users.txt'
	# stream_size = 100
	# num_asked = 30
	# output_file = 'task4.csv'

	reservoir_lst = []
	memory_limit = 100
	seq_num = 0

	output_file = open(output_file,'w')
	output_file.write('seqnum,0_id,20_id,40_id,60_id,80_id' + '\n')

	random.seed(553)
	bxInstance = BlackBox()

	for i in range(num_asked):
		user_stream = bxInstance.ask(input_file, stream_size)
		reservoir_sampling(user_stream, i)

	output_file.close()

	t2 = time.time()
	# print(t2-t1)