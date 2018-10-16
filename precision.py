
import math
import numpy as np
import decimal

def activation_function(x):
    return 1/(1 + math.exp(-x))
def deriv_activation_function(x):
    return activation_function(x)*(1-activation_function(x))

decimal.getcontext().prec = 64    
    
error = 0.5
x = 0.1
    
print(np.float16(error) * np.float16(deriv_activation_function(x)))
print(float(error) * float(deriv_activation_function(x)))
print(decimal.Decimal(error) * decimal.Decimal(deriv_activation_function(decimal.Decimal(x))))