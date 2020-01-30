#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:25:19 2020

@author: fengjiang
"""
#### function 
def hihi(x):
    return x+x
#### use of the function 
y=hihi(5)
print(y) 



### list (sequence indexing)
x=[1, 2, 3]
print(x[-2])


### loop
mylist =['a', 'b','c']
for idx, letter in enumerate(mylist):
    print(idx, ' : ',letter )




#### numpy array
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3] 
# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])    

