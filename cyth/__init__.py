import pyximport
pyximport.install()

from .cyftns import (
    get_corrcoeff, get_asymms_sample, get_2d_rel_hist, copy_arr)
