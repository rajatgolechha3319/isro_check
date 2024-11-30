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
    
    fits_file_path = '/home/baadalvm/zaki/pipeline/class_data/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/cla_collection/cla/data/calibrated/2020/04/07/ch2_cla_l1_20200407T000047620_20200407T000055620.fits'
    XSM_raw_path = '/home/baadalvm/xsm2020/2020/04/07/raw'
    XSM_calibrated_path ='/home/baadalvm/xsm2020/2020/04/07/calibrated'
    
    data = fits.open(fits_file_path)
    
    header1 = data[1].header
    startime = header1['STARTIME']
    endtime = header1['ENDTIME']
    
    tref = datetime(2017,1,1)
    
    tstart = (datetime.strptime(startime, '%Y-%m-%dT%H:%M:%S.%f') - tref).total_seconds()
    tstop =(datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%S.%f')-tref).total_seconds()
    
    l1dir = XSM_raw_path
    l2dir = XSM_calibrated_path
    
    x1 = datetime.strptime(startime, '%Y-%m-%dT%H:%M:%S.%f')
    x2 = datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%S.%f')
    
    base = f'ch2_xsm_{x1.year:04d}{x1.month:02d}{x1.day:02d}_v1'

    l1file = l1dir+'/'+base+'_level1.fits'
    hkfile = l1dir+'/'+base+'_level1.hk'
    safile = l1dir+'/'+base+'_level1.sa'
    gtifile = l2dir+'/'+base+'_level2.gti'
    
    specbase = f'ch2_xsm_{x1.year:04d}{x1.month:02d}{x1.day:02d}_{x1.hour:02d}{x1.minute:02d}{x1.second:02d}_{x2.hour:02d}{x2.minute:02d}{x2.second:02d}'
    print(specbase)
    specfile = specbase+'.pha'
    
    genspec_command="xsmgenspec l1file="+l1file+" specfile="+specfile+" spectype='time-integrated'"+ " tstart="+str(tstart)+" tstop="+str(tstop)+" hkfile="+hkfile+" safile="+safile+" gtifile="+gtifile
    
    os.system(genspec_command)
    
    # Move the pha and arf files to temp folder
    os.system('mv '+specfile+' /home/baadalvm/MIDTERM_CODE/src/temp')
    os.system('mv '+specbase+'.arf /home/baadalvm/MIDTERM_CODE/src/temp')
    
    path = '/home/baadalvm/MIDTERM_CODE/src/temp/' + specfile
    pha_data = fits.open(path)
    spectrum = pha_data[1].data['COUNTS']
    
    arf_path = path.replace('.pha','.arf')
    arf_data = fits.open(arf_path)
    effective_area = arf_data[1].data['SPECRESP']
    
    rmf_data = fits.open('/home/baadalvm/MIDTERM_CODE/src/CH2xsmresponse20200423v01.rmf')
    energy_lo = rmf_data[2].data['ENERG_LO']
    energy_hi = rmf_data[2].data['ENERG_HI']
    energy_bins = (energy_hi + energy_lo) / 2
    energy_errors = (energy_hi - energy_lo) / 2
    
    x_original = np.arange(len(spectrum))
    x_new = np.linspace(0, len(spectrum)-1, len(energy_bins))
    spectrum_interp = interp1d(x_original, spectrum, kind='linear')(x_new)

    # Convert count rate to flux
    exposure = pha_data[1].header['EXPOSURE']  # Exposure time in seconds
    flux = spectrum_interp / exposure / effective_area / energy_errors
    flux = np.nan_to_num(flux, nan=0.0)

    # Create an output text file
    seed = np.random.randint(1, 1000)
    
    with open(f'/home/baadalvm/MIDTERM_CODE/src/temp/solar_spectrum_{seed}.txt', 'w') as file:
        # file.write('# Energy(keV)  Error(keV)  Flux(photons/(s*cm^2*keV))\n')
        for en, err, fl in zip(energy_bins, energy_errors, flux):
            file.write(f"{en:.4f}  {err:.4f}  {fl:.6e}\n")

    # Close all files
    pha_data.close()
    arf_data.close()
    rmf_data.close()
    
    solar_spectrum_file_path = f'/home/baadalvm/MIDTERM_CODE/src/temp/solar_spectrum_{seed}.txt'
    
    spectrumfile = fits_file_path
    spec_data = Spectrum(spectrumfile)
    