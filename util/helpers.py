
import numpy as np

import matplotlib.image as mpimg
from PIL import Image


def load_image(filename):
    return mpimg.imread(filename)

def save_image(x, filename):
    Image.fromarray(np.uint8(x * 255)).save(filename)