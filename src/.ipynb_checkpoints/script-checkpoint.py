# Imports

from xspec import *

from datetime import datetime
import os
import numpy as np
from astropy.io import fits
import corner
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Importing necessary modules
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    n = len(sys.argv)
    # Usage : Fits file path , XSM-raw folder file path XSM-calibrated folder file path
    
    fits_file_path = sys.argv[1]
    XSM_raw_path = sys.argv[2]
    XSM_calibrated_path = sys.argv[3]
    
    print(fits_file_path,XSM_calibrated_path,XSM_raw_path)