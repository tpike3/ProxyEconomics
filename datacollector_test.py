# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:33:40 2020

@author: ymamo
"""

import pickle

infile = open("data.pkl", 'rb')
new_dict = pickle.load(infile)
infile.close()

print(new_dict)