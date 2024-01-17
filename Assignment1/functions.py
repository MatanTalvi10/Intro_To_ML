import numpy as np
import matplotlib.pyplot as plt

def emprical_assist(empir_mean: float,epsilon: float) -> int:
    res = 0
    abs_cal = abs(empir_mean - 0.5)
    if(abs_cal > epsilon):
        res = 1
    return res