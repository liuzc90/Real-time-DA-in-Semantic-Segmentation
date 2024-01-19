# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:52:16 2024

@author: liuzh
"""

import matplotlib.pyplot as plt
import numpy as np

epoch = np.linspace(2, 48,24)
miou1 = [24.9,30.2,33.4,32.7,32.6,39.7,42.1,43.9,44.4,44.8,44.6,48.6,49.8,50,49.9,50.7,50,49.8,50,48.9,
         49.7,49.4,50.3,50.1]
miou2 = [26.6,31.9,31.7,37.3,37.8,42.5,43.5,45.9,44.4,47.1,50.1,50.3,54.1,54.1,53.7,56.5,56.5,56.8,57.1,
         56.9,56.7,56.8,56.9,56.1]



plt.plot(epoch, miou1, 'k', label='Training on Cityscapes')
plt.plot(epoch, miou2, 'r', label='Training on GTA5')
plt.xlabel('epoch')
plt.ylabel('mIoU [%]')
plt.grid('visible')
plt.legend()
plt.show()
