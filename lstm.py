from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
torch.set_num_threads(4)

LSTM = torch.load('./weight/lstm.pt').eval()
x = torch.randn(1, 1, 1024)
h0 = torch.randn(2, 1, 512)
c0 = torch.randn(2, 1, 512)

print("\n----lets run!----")
avg_time = 0

print("compute: lstm layer")
print("iter\t time")
start = time.time() #####
os.system('m5 resetstats')

x, (h, c) = LSTM(x, (h0, c0))

os.system('m5 dumpstats')
end = time.time()   #####
print("time: ", end-start)
