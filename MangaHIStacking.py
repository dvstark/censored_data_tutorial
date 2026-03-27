import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy.table import Table, vstack
from astroquery.sdss import SDSS
from scipy.interpolate import interp1d
from scipy import integrate
import re
import os
from pathlib import Path

# Generates list of available files to process

# block added by dstark  on Mar 27 2026
data_path = Path('./Data/Final/')
data_path.mkdir(parents=True, exist_ok=True)

dirlist = os.listdir('./Data/Final/')
regex = re.compile('mangaHI-\d.+\.csv')
#regex = re.compile('SDSS.+\.csv')

matches = [string for string in dirlist if re.match(regex, string)]

# Generate index for control sample to use
alfalfadirlist = os.listdir('./spectra/alfalfa/csv/')
alfalfamatches = [string for string in alfalfadirlist if re.match(regex, string)]
gbtdirlist = os.listdir('./spectra/gbt/csv/')
gbtmatches = [string for string in gbtdirlist if re.match(regex, string)]

def myround(x, base=5):
    return base * np.round(x/base)

def removespace(string_input): 
    return string_input.replace(" ", "")

def fluxinterp(x_val, vHI, fHI):
    vel_mask = np.isin(x_val, myround(vHI))
    cnt = 0
    temp_a = pd.Series(np.empty(len(x_val)))
    for i in range(len(x_val)):
        if vel_mask[i]:
            temp_a[i] = fHI[cnt]
            cnt += 1
        else:
            temp_a[i] = np.NaN

    temp_a = temp_a.interpolate(limit_direction='both')
    return temp_a

def headerdata(file_path, dataset='manga'):
    file_dict = dict()

    with open(file_path) as f:
        cnt = 0
        for line in f:
            if line.startswith('#'):
                cnt += 1
                if line.count(':') == 1:
                    hold = line[1:].strip().split(': ')
                    file_dict[hold[0]]=hold[1]
                else:
                    pass
                header = line
            else:
                break
    
    header = header[1:].strip().split(',')

    return cnt, header, file_dict

def mangafile(file_path):
        headerlength, header, file_dict = headerdata(file_path)
        data = pd.read_csv(file_path, skiprows=headerlength, delim_whitespace=True, names=header)

        return np.array(data['vHI']), np.array(data['fHI'])

def create_control(galaxylist, savelist=False):
    c = (299792.458)
    controltable = Table()
    staticquery = "SELECT TOP 1 plateifu, mangaid, logmstars, sini, vhi, rms FROM mangaHIall" # Retreives the relevant information to do a weighted stack of the spectra
    nonmatch = 0
    stellarmassrange = 0.2
    zrange = 0.01 * c
    sinirange = 0.1
    control_list = []
    for galaxyname in galaxylist:
        #stellarmass, z, rms, sini = getatrributes(galaxyname)
        #stellarmass = np.log10(stellarmass)
        #z = z * c
        #conditions = f" WHERE logmstars BETWEEN {stellarmass-stellarmassrange} AND {stellarmass+stellarmassrange} AND vhi BETWEEN {z-zrange} AND {z+zrange} AND sini BETWEEN {sini-sinirange} AND {sini+sinirange} AND confprob < 0.1"

        conditions = f" WHERE plateifu = '{galaxyname}'" # Change to necessary constraints for the galaxies
        query = staticquery + conditions
        
        try:
            res = Table(SDSS.query_sql(query, data_release=17)) # Makes SQL query to SDSS database
        except:
            nonmatch +=1
            continue
        if len(res) == 0:
            print(galaxyname)
            nonmatch += 1
            continue
        else:
            pass

        for element in res: # Stops from adding duplicate galaxies
            if element['plateifu'][0] not in control_list:
                #res = element
                pass
            else:
                continue

        control_list.append(res['plateifu'])
        controltable = vstack([controltable, res])
    
    if savelist == True: # Saves table of relevant galaxy attributes
        controltable.write('./control_sample_data.fits', overwrite=True)

    return controltable

def get_control_spectra(controlname):
    if f'mangaHI-{controlname}.csv' in alfalfamatches:
        path = f'./spectra/alfalfa/csv/mangaHI-{controlname}.csv'
    elif f'mangaHI-{controlname}.csv' in gbtdirlist:
        path = f'./spectra/gbt/csv/mangaHI-{controlname}.csv'
    else:
        print(f'No spectra data found for {controlname}')
        

    return mangafile(path)


def stack_control(ctrl_galaxy_list, width=1000, method='stellarmass'):
    # Creates common velocity array
    x_vals = np.arange(-width, width+1, 5.)
    # Value to average weigted spectra by
    denom = 0
    # Empty array for stacking spectra
    avgflux = np.zeros_like(len(x_vals))
    
    for galaxy in ctrl_galaxy_list:
        
        try:
            
            vHI, fHI = get_control_spectra(str(galaxy['plateifu']))

            cz = galaxy['vhi']
            stellarmass = 10**float(galaxy['logmstars'])

            dist = cz / cosmo.H(0).value
            vHI = vHI - cz
            trim_mask = (vHI <= width) & (vHI >= -width)
            vHItrim = vHI[trim_mask] 
            fHItrim = fHI[trim_mask]
            # Interpolates missing values
            fHItrim = fluxinterp(x_vals, vHItrim, fHItrim)

            if method == 'stellarmass':
                avgflux = avgflux + (fHItrim*np.float64(2.356e+5)*dist**2/stellarmass) #2.356e5 is conversion to gas fractionation units described in 10.1093/mnras/staa464
                denom += 1
            elif method == 'distance':
                avgflux = avgflux + (fHItrim*dist)
                denom += dist
            #elif method == 'rms':
            #    signal_mask = (vHI > 400) & (vHI < -400)
            #    fHImasked = fHI[signal_mask]
            #    rms = np.float64(table.loc[table['SDSS_NAME'] == sdssname]['rms'])
            #    avgflux = avgflux + fHItrim/rms
            #    denom += 1/rms
            
        except: 
            print('failed to stack spectrum for {}'.format(galaxy['plateifu']))

    return x_vals, np.array(avgflux / denom)