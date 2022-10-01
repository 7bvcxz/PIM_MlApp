from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os

option = input("[CH,K1,K2]= 2,4,2:1, 4,6,3:2, 8,12,6:3, 16,24,12:4, 32,41,21:5 \nenter layer to run: ")

print("option: ", option)

os.system('m5 checkpoint')
os.system('echo CPU Switched!')
print("\n----lets run!----")

itr = 5

CONV = torch.load('./weight/conv'+option+'.pt').eval()
avg_time = 0

print("compute: conv layer")
print("iter\t time")
for i in range(itr):
    x = torch.randn((1, 1, 160, 1151))
    start = time.time() #####
    os.system('m5 dumpstats')
    x = CONV(x)
    os.system('m5 dumpstats')
    end = time.time()   #####
    print(i, "\t", end-start)
    avg_time = avg_time + end - start
avg_time = avg_time / itr
print("avg_time: ", avg_time)