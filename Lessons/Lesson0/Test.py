from math import radians
import numpy as np     # installed with matplotlib
import matplotlib.pyplot as plt
from PIL import Image  # keras uses this under the hood in later lessons

def main():
    x = np.arange(0, radians(1800), radians(12))
    plt.plot(x, np.cos(x), 'b')
    plt.show()

main()

# tested that we can run python