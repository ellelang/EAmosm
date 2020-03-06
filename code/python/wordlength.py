from pathlib import Path
import itertools
data_folder = Path('C:/Users/langzx/Desktop/github/MyLeetcode/sample/data')
file = open(data_folder/'beast.txt', 'r')
book = file.read()
words = list(book.lower().split())
words_len = set(map(len, words))

def countWords(file):
   words = list(file.lower().split())
   words_len = list(map(len, words))
   dic={}
   for X in words:
       X_len = len(X)
       if X_len in dic:        
          dic[X_len] = dic[X_len] + 1
       else:
          dic[X_len]=1
   return dic

dic = countWords(book)
sorted_items=sorted(dic.items())   # if you want it sorted

import math,random
def time (rateParam):
    return -math.log(1 - random.random())/rateParam
  
time(2)

import random
import math

_lambda = 5
_num_arrivals = 100
_arrival_time = 0

for i in range(_num_arrivals):
	#Get the next probability value from Uniform(0,1)
	p = random.random()

	#Plug it into the inverse of the CDF of Exponential(_lamnbda)
	_inter_arrival_time = -math.log(1.0 - p)/_lambda