from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

option = input("# words = 1: 1, 2: 64, 3: 256, 4:576\nenter # of words: ")
print("option: ", option)

if (option == '1'):
    m = 1
elif (option == '2'):
    m = 64
elif (option == '3'):
    m = 256
elif (option == '4'):
    m = 576

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)

LSTM = torch.load('./weight/lstm2.pt').eval()
x = torch.randn(1, m, 1024)
h0 = torch.randn(2, m, 512)
c0 = torch.randn(2, m, 512)

print("\n----lets run!----")
avg_time = 0

print("compute: lstm layer")
fl.flush(LSTM.weight_hh_l0, 512*4096*4)
fl.flush(LSTM.weight_hh_l0_reverse, 512*4096*4)
start = time.time() #####
os.system('m5 resetstats')

x, (h, c) = LSTM(x, (h0, c0))

os.system('m5 dumpstats')
end = time.time()   #####
print("time: ", end-start)
