#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates ## date format
import matplotlib.cm ## color map
import pandas as pd
import numpy as np
import datetime as dt
import configparser
import pickle

DEBUG = 1

if os.uname()[1] == 'adak':
    HSVA_dir = '/mnt/data/UAF-data/MOSIDEO/HSVA/data/'
    fig_dir = '/home/megavolts/Desktop/paper/HSVA'
    processed_dir = '/mnt/data/UAF-data/MOSIDEO/processed/'
    pkl_subdir = 'python_pkl'
else:
    print('No directory defined for this machine')

temp_config_fn = 'temperature_string.config'
temp_data_fn = 'MOSIDEO_CIRFA_HSVA_2017_consolidated_data.xlsx'
temp_data_pkl = 'MOSIDEO_CIRFA_HSVA_2017.pkl'

#----------------------------------------------------------------------------------------------------------------------#

# Import Temperature  field to compute permeability
# load temperature config file
config_file = configparser.ConfigParser()
temperature_subdir = os.path.join(processed_dir, 'temperature')
temperature_config = os.path.join(temperature_subdir, temp_config_fn)
config_file.read(os.path.join(temperature_config))

depth_dict = {}
column_headers = config_file['general']['header'].split(', ')
for string_number in range(1, config_file.getint('general', 'string number')+1):
    depth = config_file['string '+str(string_number)]['depth'].split(', ')
    depth = {'T'+str(string_number)+'_'+str(ii): float(depth[ii]) for ii in range(0, depth.__len__())}
    depth_dict[config_file['string '+str(string_number)]['ice type']] = depth
    column_headers += list(depth.keys())

temperature_data = os.path.join(temperature_subdir, temp_data_fn)
data = pd.read_excel(temperature_data, sheet_name='Data', skiprows=11, headers=None, names=column_headers)

temperature = {ice: data[['Time']+list(depth_dict[ice].keys())] for ice in depth_dict.keys()}

# Export
temperature_subdir = os.path.join(processed_dir, 'temperature')
fname = os.path.join(temperature_subdir, pkl_subdir, temp_data_pkl)
with open(fname, 'wb') as f:
    pickle.dump(temperature, f)


for ice in temperature.keys():
    data = temperature[ice].set_index('Time', drop=True)
    # rename columns of df according to depth_dict
    data = data.rename(columns=depth_dict[ice]).sort_index(axis=1)

    time = data.index
    depth = data.keys()
    t, z = np.meshgrid(time, depth)
    T = data.values

    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    CF = ax.contourf(t, z, T.transpose(), vmin=-15, vmax=0, levels=np.linspace(-20, 0, 21), cmap=matplotlib.cm.cividis)
    CS = ax.contour(t, z, T.transpose(), sorted([0, -2, -5, -10, -15]), colors='k')
    ax.clabel(CS, fmt="%2.0f", colors='k', inline=True)
    CB = plt.colorbar(CF, orientation='vertical', ticks=sorted([0, -2, -5, -10, -15]), fraction=0.06, pad=0.025)
    CB.add_lines(CS)
    ax.axes.set_xlim(dt.datetime(2017, 3, 28), dt.datetime(2017, 4, 4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.axes.set_ylim(22, 0)
    plt.title(ice)
    plt.show()



