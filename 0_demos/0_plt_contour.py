#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:26:30 2020

@author: fengjiang
"""

import matplotlib.pyplot as plt
import numpy as np



N = 100
x = np.linspace(-3.0, 3.0, N)
y = np.linspace(-2.0, 2.0, N)

X, Y = np.meshgrid(x, y)

Z1 = X**2 +Y**2  #np.exp(-(X)**2 - (Y)**2)
plt.contourf(x, y, Z1)
plt.show()