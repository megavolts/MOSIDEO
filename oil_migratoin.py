#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
import sys
sys.path.extend(['/home/megavolts/git/seaice'])

import seaice
import os
import numpy as np
import pandas as pd
import configparser  # to read config files
import matplotlib.pyplot as plt

data_folder = '/mnt/data/UAF-data/MOSIDEO/HSVA/data/'

core_folder = os.path.join(data_folder, 'cores')

## list profile
profile_list = seaice.core.list_ic(core_folder, 'profile')
owf_df = []

for profile in profile_list:
    # profile = 'HSVA-20170402-B2.profile'
    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(core_folder, profile))

    o_cores = config_file['coordinate system']['salinity'].split(', ')
    if o_cores[0] is not '':
        ic_list = [os.path.join(core_folder, core)+'.xlsx' for core in o_cores]

        ic_dict = seaice.core.import_ic_list(ic_list)
        owf_avg = None
        owf_avg_flag = 0

        for ic in ic_dict:
            if 'oil weight fraction' in ic_dict[ic].variables():
                owf = ic_dict[ic].profile[ic_dict[ic].profile.variable == 'oil weight fraction']['oil weight fraction']
                if owf_avg is None:
                    owf_avg = np.nan*np.ones([owf.__len__(), ic_dict.__len__()])
                owf_avg[:, owf_avg_flag] = owf.values
                owf_avg_flag +=1
        owf_df.append([config_file['general']['date'], config_file['general']['ice type'], config_file['general']['lens'], np.nanmean(owf_avg)])
    else:
        owf_df.append([config_file['general']['date'], config_file['general']['ice type'], config_file['general']['lens'], np.nan])

import os
import matplotlib.cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_folder = '/mnt/data/UAF-data/MOSIDEO/processed/temperature/'
temperature_file = 'MOSIDEO_CIRFA_HSVA_2017_consolidated_data.xlsx'

T1 = ['T1_'+str(ii) for ii in range(0, 22)]
d1 = [-5, 0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 40, 60, 80]
T2 = ['T2_'+str(ii) for ii in range(0, 22)]
d2 = [-5, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 40, 60, 80]

columns = ['Time', 'Battery Logger', 'Probe 1', 'Multiplexer', 'Probe 2', 'Multiplexer', 'Air Temp 1',
           'Air Temp 2a', 'Air Temp 2b', 'Air Temp 2c', 'Illuminance', 'Clean Patch', 'Oil Patch',
           'Surrounding Ice	Temperature'] + T1 + T2

data = pd.read_excel(os.path.join(data_folder,  'MOSIDEO_CIRFA_HSVA_2017_consolidated_data.xlsx'), sheet_name='Data', skiprows=11, headers=None, names=columns)

temperature={}
temperature['granular'] = data[['Time']+T1]
temperature['columnar'] = data[['Time']+T2]

time = temperature['granular']['Time']
depth = d1
import datetime as dt

t, z = np.meshgrid(time, depth)
T = temperature['granular'][T1]

fig = plt.figure(figsize=(10.5, 8))
ax = fig.add_subplot(1, 1, 1)
CF = ax.contourf(t, z, T.transpose(), vmin=-15, vmax=0, levels=np.linspace(-20, 0, 21), cmap=matplotlib.cm.Blues)
CS = ax.contour(t, z, T.transpose(), sorted([0, -2, -5, -10, -15]), colors='k')
ax.clabel(CS, fmt="%2.0f", colors='k', inline=True)

CB = plt.colorbar(CF, orientation='vertical', ticks=sorted([0, -2, -5, -10, -15]), fraction=0.06, pad=0.025)
CB.add_lines(CS)

ax.axes.set_xlim(dt.datetime(2017, 3, 28), dt.datetime(2017, 4, 4))
ax.axes.set_ylim(22, 0)
#
#
# owf_df = pd.DataFrame(owf_df, columns=['date', 'ice type', 'lens', 'owf'])
# owf_df['date'] = pd.to_datetime(owf_df['date'], format='%Y-%m-%d')
# owf_df = owf_df.sort_values(by='date')
# ax2 = ax.twinx()
# #ax2.plot(owf_df.loc[owf_df['ice type']=='granular', 'date'], owf_df.loc[owf_df['ice type']=='granular', 'owf']*1000, 'k', label='granular')
# #ax2.plot(owf_df.loc[owf_df['ice type']=='columnar', 'date'], owf_df.loc[owf_df['ice type']=='columnar', 'owf']*1000, 'k--', label='columnar')
# ax2.set_ylabel('oil volume fraction', color='k')
# ax.set_ylabel('ice thickness', color='b')
# ax2.legend()
# plt.xlim([dt.datetime(2017, 3, 30), dt.datetime(2017, 4, 2)])
ax.set_xticklabels([dt.date(2017, 3, 30), '', dt.date(2017, 3, 31), '', dt.date(2017, 4, 1)], rotation=30)
plt.savefig(os.path.join(fig_dir, 'T-field-' + ice + '.pdf'))
#plt.savefig('/home/megavolts/Desktop/MOSIDEO-t_ovf-CBAR')
plt.show()