#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""

"""
import sys
import os
import numpy as np
import pandas as pd
import configparser  # to read config files
import matplotlib.pyplot as plt

sys.path.extend(['/home/megavolts/git/seaice'])
import seaice

data_folder = '/mnt/data/UAF-data/MOSIDEO/HSVA/data/'

core_folder = os.path.join(data_folder, 'cores')

ics_stack = seaice.core.corestack.CoreStack()

## list profile
profile_list = seaice.core.list_ic(core_folder, 'profile')

oil_density = 0.845  # kg/m3 == g/cm3

for profile in profile_list:
#profile = 'HSVA-20170401-D2.profile'
    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(core_folder, profile))

    o_cores = config_file['coordinate system']['salinity'].split(', ')
    if o_cores[0] is not '':
        ic_list = [os.path.join(core_folder, core) + '.xlsx' for core in o_cores]
        ic_dict = seaice.core.import_ic_list(ic_list)

        for ic in ic_dict:
            ic_dict[ic].profile['ice type'] = config_file['general']['ice type']
            if 'conductivity' in ic_dict[ic].variables():
                c = ic_dict[ic].profile[ic_dict[ic].profile.variable == 'conductivity']['conductivity']
                if 'measurement temperature' in ic_dict[ic].profile.keys():
                    t = ic_dict[ic].profile[ic_dict[ic].profile.variable == 'conductivity']['measurement temperature']
                elif 'Conductivity measurement temperature' in ic_dict[ic].profile.keys():
                    t = ic_dict[ic].profile[ic_dict[ic].profile.variable == 'conductivity']['Conductivity measurement temperature']
                ic_dict[ic].profile['salinity'] = seaice.property.nacl.sw_con2sal(c, t, 0)

            if 'oil weight fraction' in ic_dict[ic].variables():
                owf = ic_dict[ic].profile[ic_dict[ic].profile.variable == 'oil weight fraction']['oil weight fraction']
                ic_dict[ic].profile['oil volume fraction'] = owf/oil_density

         ics_stack = ics_stack.append(seaice.core.corestack.stack_cores(ic_dict))

ics_stack.groupby(['ice type', 'date'])['oil volume fraction'].mean()
ics_stack.groupby(['ice type', 'date'])['oil volume fraction'].median()
ics_stack.groupby(['ice type', 'date'])['oil volume fraction'].std()

mean = ics_stack.groupby(['ice type', 'date'])['oil volume fraction'].mean().reset_index()
std = ics_stack.groupby(['ice type', 'date'])['oil volume fraction'].std().reset_index()

mean['low'] = mean['oil volume fraction']-std['oil volume fraction']
mean['sup'] = sup = mean['oil volume fraction']+std['oil volume fraction']
mean['date']
color = {'columnar':'r', 'granular':'b'}
plt.figure()
for ice_type in ics_stack['ice type'].unique():
    print(ice_type)
    plt.plot(mean.loc[mean['ice type']==ice_type, ['date']], mean.loc[mean['ice type']==ice_type, ['oil volume fraction']], color=color[ice_type])
    plt.plot(mean.loc[mean['ice type']==ice_type, ['date']], mean.loc[mean['ice type']==ice_type, ['low']], linestyle=':', color=color[ice_type])
    plt.plot(mean.loc[mean['ice type']==ice_type, ['date']], mean.loc[mean['ice type']==ice_type, ['sup']], linestyle='--', color=color[ice_type])
plt.ylabel('oil volume fraction')
plt.xlabel('time [day]')
plt.savefig('/home/megavolts/Desktop/mosido-owf')
plt.show()
