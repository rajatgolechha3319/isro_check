import os
from astropy.io import fits
import pandas as pd

def check_solar_angle(fits_file):
    """ Check if the SOLARANG value in the FITS header is less than 90 """
    with fits.open(fits_file) as hdul:
        solar_ang = hdul[0].header['SOLARANG']
    return solar_ang < 90

def check_rate_in_interval(lc_file, start_time, end_time):
    """ Check if the rate within the specified interval is above 2000 """
    data = pd.read_csv(lc_file, delimiter='\t')  # Update this if the delimiter is different
    filtered_data = data[(data['TIME'] >= start_time) & (data['TIME'] <= end_time)]
    return filtered_data['RATE'].max() > 2000

def process_data(year, month, date, start_time, end_time, base_path='/path/to/data'):
    """ Process the data for a given year, month, and date """
    cla_path = os.path.join(base_path, f'cla/data/calibrated/{year}/{month:02d}/{date:02d}')
    xsm_path = os.path.join(base_path, f'xsm/data/{year}/{month:02d}/{date:02d}')

    # List all FITS files in the CLA date directory
    fits_files = [os.path.join(cla_path, f) for f in os.listdir(cla_path) if f.endswith('.fits')]

    for fits_file in fits_files:
        if check_solar_angle(fits_file):
            with fits.open(fits_file) as hdul:
                # Assume 'TIME' in FITS is in hours since start of the day
                file_start_hours = hdul[0].header['TSTART']
                file_end_hours = hdul[0].header['TSTOP']

            lc_file = os.path.join(xsm_path, 'calibrated', 'xsm_lc_file.lc')  # Update the file name pattern if needed

            if check_rate_in_interval(lc_file, start_time, end_time):
                raw_folder_path = os.path.join(xsm_path, 'raw')
                calibrated_folder_path = os.path.join(xsm_path, 'calibrated')

                # Now run script.py with the required arguments
                os.system(f"python script.py {fits_file} {raw_folder_path} {calibrated_folder_path}")
1
if __name__ == '__main__':
    year = int(input("Enter year: "))
    month = int(input("Enter month: "))
    date = int(input("Enter date: "))
    start_time = input("Enter start time (HHMMSS): ")
    end_time = input("Enter end time (HHMMSS): ")
    process_data(year, month, date, start_time, end_time)
