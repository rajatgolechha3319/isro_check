import os
from astropy.io import fits
import pandas as pd
from datetime import datetime

# Reference time for conversion
tref = datetime(2017, 1, 1)

def check_solar_angle(fits_file):

    """ Check if the SOLARANG value in the FITS header is less than 90 """
    with fits.open(fits_file) as hdul:
        solar_ang = hdul[1].header['SOLARANG']
    # return solar_ang < 90
    return True

# def check_rate_in_interval(lc_file, tstart, tstop):
#     with fits.open(lc_file) as hdul:  # Replace 'path_to_your_file.lc' with the path to your .lc file
#         data = hdul[1].data
#     df = pd.DataFrame(data)
    
    
    

def process_data(year, month, date, start_time, end_time, base_path='/home/baadalvm/MIDTERM_CODE/data/'):
    """ Process the data for a given year, month, and date """
    cla_path = os.path.join(base_path, f'cla/data/calibrated/{year}/{month:02d}/{date:02d}')
    xsm_path = os.path.join(base_path, f'xsm/data/{year}/{month:02d}/{date:02d}')

    # Convert start and end time from input to seconds since tref
    start_time_str = f"{year}-{month:02d}-{date:02d}T{start_time[:2]}:{start_time[2:4]}:{start_time[4:]}.000"
    end_time_str = f"{year}-{month:02d}-{date:02d}T{end_time[:2]}:{end_time[2:4]}:{end_time[4:]}.000"
    input_tstart = (datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S.%f') - tref).total_seconds()
    input_tstop = (datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M:%S.%f') - tref).total_seconds()

    # List all FITS files in the CLA date directory
    fits_files = [os.path.join(cla_path, f) for f in os.listdir(cla_path) if f.endswith('.fits')]
    print("a")
    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            # Extract start and end time from the FITS header
            startime = hdul[1].header['STARTIME']
            endtime = hdul[1].header['ENDTIME']
            # print(startime)
            
            # Convert start and end times to seconds since tref
            tstart = (datetime.strptime(startime, '%Y-%m-%dT%H:%M:%S.%f') - tref).total_seconds()
            tstop = (datetime.strptime(endtime, '%Y-%m-%dT%H:%M:%S.%f') - tref).total_seconds()
        # print("b")
        # Check if file duration lies within the specified input interval
        if tstart >= input_tstart and tstop <= input_tstop:
            print("c")
            # Check solar angle
            if check_solar_angle(fits_file):
                print("d")
                # Locate the .lc file
                lc_name = f'ch2_xsm_{year:04d}{month:02d}{date:02d}_v1_level2.lc'
                lc_file = os.path.join(xsm_path, 'calibrated', lc_name) 

                # print("e")
                raw_folder_path = os.path.join(xsm_path, 'raw')
                calibrated_folder_path = os.path.join(xsm_path, 'calibrated')

                # Execute pipeline2.py with the required arguments
                print(fits_file)
                print(raw_folder_path)
                print(calibrated_folder_path)
                try:
                    os.system(f"/usr/bin/python3 /home/baadalvm/MIDTERM_CODE/src/pipeline2.py {fits_file} {raw_folder_path} {calibrated_folder_path}")
                except Exception as e:
                    continue

if __name__ == '__main__':
    year = 2021
    month = 4
    start_time = "200000"
    end_time = "210000"
    date = 22
    
    process_data(year, month, date, start_time, end_time)
