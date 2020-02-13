import random
import numpy as np
from random import choice
import time
a=np.random.randn(2,3)
b=np.where(a>0,a,a*10)
print(b)