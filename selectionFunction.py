# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from attributes import *

# here we import X features of dataset
atts = atts_giveme
