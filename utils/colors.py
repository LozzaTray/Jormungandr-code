from colour import Color
import math


def color_between(val, start_col="blue", end_col="red", resolution=10):
    """interpolates a value in range [0,1] to corresponding end colour"""
    if val >= 1:
        val = 0.99
    elif val < 0:
        val = 0
        
    start = Color(start_col)
    end = Color(end_col)
    index = math.floor(val * resolution)
    return list(start.range_to(end, resolution))[index].hex


def plt_color(index):
    return "C" + str(index)