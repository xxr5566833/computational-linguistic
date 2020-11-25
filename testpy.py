import pandas as pd
import math
from question1 import get_rank

P = {"a" : 9, "b": 10,
     "c": 1, "d": 2,
     "e": 11, "f": 12,
     "g": -1, "h": -2,
     "i": 0.5, "j": 0.3,
     "k": 1, "l": 2,
     "m": -1, "n": 12,
     "o": 1, "p": 2,
     "q": 0.5}

print(get_rank(P))

def a():
    return 1
def b():
    return 2
fs = {1 : a,
      2: b}
print(fs[2]())