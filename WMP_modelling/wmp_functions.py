import numpy as np
import pandas as pd

def assign_zero_trace(matrix):
    for i in range(0,len(matrix)):
        for j in range(0,len(matrix[i])):
            if i == j:
                matrix[i, j] = 0
    return matrix

