import logging
import numpy as np
import pandas as pd 
import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import credit_risk_model.config as config 
print(config.TARGET)