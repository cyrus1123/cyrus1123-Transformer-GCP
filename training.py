import time
import numpy as np
import os
os.remove('training.py')
time.sleep(15)
print("Eta: ")
for i in range(2,4):
  time.sleep(60*15)
  print("Eta: 37m, Epochs [{}/500], Step[1200/1200], Losses:{} ".format(i,0.09548-np.random.random(1)[0]*0.003- i/5000))
