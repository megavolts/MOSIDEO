#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
2018/07/10 OK


Ice core references set at the ice bottom

Use
Use

"""


import sys
import os
import configparser  # to read config files
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates ## date format
import matplotlib.cm ## color map
from matplotlib.gridspec import GridSpec  # Manage subplot with a grid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

sys.path.extend(['/home/megavolts/git/seaice'])
import seaice

import datetime as dt
date_start = dt.datetime(2017, 3, 30)
date_end = dt.datetime(2017, 4, 2)

DEBUG = 1
DISCRETIZED = 1

if os.uname()[1] == 'adak':
    HSVA_dir = '/mnt/data/UAF-data/MOSIDEO/HSVA/data/'
    fig_dir = '/home/megavolts/Desktop/paper/HSVA'
    processed_dir = '/mnt/data/UAF-data/MOSIDEO/processed/'
    pkl_subdir = 'python_pkl'
else:
    print('No directory defined for this machine')

core_subdir = os.path.join(HSVA_dir, 'cores')
core_pkl = 'core_data.pkl'

temp_config_fn = 'temperature_string.config'
temp_data_fn = 'MOSIDEO_CIRFA_HSVA_2017_consolidated_data.xlsx'
temp_data_pkl = 'MOSIDEO_CIRFA_HSVA_2017.pkl'
config_file = configparser.ConfigParser()

#----------------------------------------------------------------------------------------------------------------------#
# Import Ice Core Data and extract oil weight fraction (owf)
fname = os.path.join(HSVA_dir, pkl_subdir, core_pkl)
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        ic_df = seaice.core.corestack.CoreStack(pickle.load(f))
else:
    profile_list = seaice.core.list_ic(core_subdir, 'profile')
    owf_df = pd.DataFrame()
    ic_dict = {}
    for profile in profile_list:
        config_file.read(os.path.join(core_subdir, profile))

        o_cores = config_file['coordinate system']['salinity'].split(', ')
        if o_cores[0] is not '':
            ic_list = [os.path.join(core_subdir, core)+'.xlsx' for core in o_cores]
            _ic_dict = seaice.core.import_ic_list(ic_list)

            for ic in _ic_dict:
                # compute salinity

                if 'oil weight fraction' in _ic_dict[ic].variables():
                    _ic_dict[ic].profile.loc[
                        _ic_dict[ic].profile.variable == 'oil weight fraction', 'oil weight fraction'] = \
                    _ic_dict[ic].profile.loc[
                        _ic_dict[ic].profile.variable == 'oil weight fraction', 'oil weight fraction'].fillna(0)

                if 'conductivity' in _ic_dict[ic].variables():
                    c = _ic_dict[ic].profile[_ic_dict[ic].profile.variable == 'conductivity']['conductivity']
                    t = False
                    ## temperature measurement could have one of the following header name
                    if 'measurement temperature' in _ic_dict[ic].profile.keys():
                        t = _ic_dict[ic].profile[_ic_dict[ic].profile.variable == 'conductivity']['measurement temperature']
                    elif 'Conductivity measurement temperature' in _ic_dict[ic].profile.keys():
                        t = _ic_dict[ic].profile[_ic_dict[ic].profile.variable == 'conductivity']['Conductivity measurement temperature']

                    if t is not False:
                        # make a copy of 'conductivity' profile
                        _s_profile = _ic_dict[ic].profile.loc[_ic_dict[ic].profile.variable == 'conductivity'].copy()
                        _s_profile['salinity'] = seaice.property.nacl.sw_con2sal(c, t, 0)
                        _s_profile['variable'] = 'salinity'
                        for col_name in ['conductivity', 'measurement temperature', 'Conductivity measurement temperature']:
                            if col_name in _s_profile.keys():
                                _s_profile.drop(col_name, axis=1, inplace=True)
                        _ic_dict[ic].add_profile(_s_profile)

                # add 'ice type' and 'lens' information to profile
                _ic_dict[ic].profile['lens'] = config_file['general']['lens']
                _ic_dict[ic].profile['ice type'] = config_file['general']['ice type']
                _ic_dict[ic].profile.reset_index(drop=True)
                ic_dict.update(_ic_dict)

    # stacking ice core in a single dataframe
    ic_df = seaice.core.corestack.stack_cores(ic_dict)

    # pickle datae
    with open(fname, 'wb') as f:
        pickle.dump(pd.DataFrame(ic_df), f)
print("Ice core data imported")

# ---------------------------------------------------------------------------------------------------------------------#
# Import Temperature  field to compute permeability
# load temperature config file
temperature_subdir = os.path.join(processed_dir, 'temperature')

fname = os.path.join(temperature_subdir, pkl_subdir, temp_data_pkl)
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        temperature, depth_dict = pickle.load(f)
else:
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

    # dictionnaries of temperature as ice type
    temperature = {ice: data[['Time']+list(depth_dict[ice].keys())] for ice in depth_dict.keys()}

    with open(fname, 'wb') as f:
        pickle.dump([temperature, depth_dict], f)
print("Temperature field imported")

# ---------------------------------------------------------------------------------------------------------------------#
# Data Processing
# (0) Discretize profiles:
ic_df = seaice.core.corestack.CoreStack(ic_df.reset_index(drop=True))

# (0) Extract temperature profile for each sampling event
# ice temperature profile are averaged each day between 10h00 and 14h00, when we cored
t_stats = pd.DataFrame()
for col in ic_df.collection.unique():
    date = ic_df.loc[ic_df.collection == col, 'date'].unique()[0]
    ice = ic_df.loc[ic_df.collection == col, 'ice type'].unique()[0]
    # look for temperature for date in between 10h00 and 14h00
    _t_data = temperature[ice].loc[(date < temperature[ice].Time + np.timedelta64(3600000000000 * 10)) &
                                   (temperature[ice].Time < date + np.timedelta64(3600000000000 * 14))]
    _t_data = _t_data.rename(columns=depth_dict[ice]).set_index('Time')

    # look for maximal ice thickness
    hi = np.nanmax([ic_df.loc[ic_df.collection == col, 'y_sup'].max(), ic_df.loc[ic_df.collection == col, 'y_low'].max()])

    # quick and dirty selection
    keys = sorted(_t_data.keys().values)
    t_select = []
    for n_key in range(keys.__len__()):
        if 0 <= keys[n_key] < hi*100:
            t_select.append(keys[n_key])
        elif 0 <= keys[n_key]:
            t_select.append(keys[n_key])
            break
    _t_data = _t_data[t_select]
    _t_data = _t_data.transpose()
    t_data = pd.DataFrame(_t_data.mean(axis=1), index=_t_data.index, columns=['temperature'])
    t_data = t_data.reset_index().rename(columns={'index': 'y_mid'})
    t_data['y_mid'] = t_data['y_mid'] / 100
    t_data['date'] = date
    t_data['variable'] = 'temperature'
    t_data['ice type'] = ice
    t_data['collection'] = col+', '+pd.to_datetime(str(date)).strftime('%Y%m%d-T-'+ice)
    t_data['ice_thickness'] = t_data['y_mid'].max()
    t_data['length'] = t_data['y_mid'].max()
    t_data['name'] = pd.to_datetime(str(date)).strftime('%Y%m%d-T-'+ice)
    t_data['v_ref'] = 'top'
    t_data['y_sup'] = np.nan
    t_data['y_low'] = np.nan
    t_data = seaice.core.profile.set_profile_orientation(t_data, 'bottom')
    ic_df = ic_df.append(t_data, sort=False)
    ic_df = ic_df.replace(col, col+', '+pd.to_datetime(str(date)).strftime('%Y%m%d-T-'+ice))

    t_data = pd.DataFrame(_t_data.mean(axis=1), index=_t_data.index, columns=['temperature'])
    t_data['stats'] = 'mean'
    t_data_add = pd.DataFrame(_t_data.std(axis=1), index=_t_data.index, columns=['temperature'])
    t_data_add['stats'] = 'std'
    t_data = t_data.append(t_data_add)
    t_data_add = pd.DataFrame(_t_data.min(axis=1), index=_t_data.index, columns=['temperature'])
    t_data_add['stats'] = 'min'
    t_data = t_data.append(t_data_add)
    t_data_add = pd.DataFrame(_t_data.max(axis=1), index=_t_data.index, columns=['temperature'])
    t_data_add['stats'] = 'max'
    t_data = t_data.append(t_data_add)
    t_data = t_data.reset_index().rename(columns={'index': 'y_mid'})
    t_data['y_mid'] = t_data['y_mid'] / 100
    t_data['date'] = date
    t_data['variable'] = 'temperature'
    t_data['ice type'] = ice
    t_data['collection'] = col+', '+pd.to_datetime(str(date)).strftime('%Y%m%d-T-'+ice)
    t_data['ice_thickness'] = t_data['y_mid'].max()
    t_data['length'] = t_data['y_mid'].max()
    t_data['name'] = pd.to_datetime(str(date)).strftime('%Y%m%d-T-'+ice)
    t_data['v_ref'] = 'top'
    t_data['y_sup'] = np.nan
    t_data['y_low'] = np.nan
    t_data = seaice.core.profile.set_profile_orientation(t_data, 'bottom')
    t_stats = t_stats.append(t_data)
print("Temperature profile extracted from temperature field")

# (1) Compute porosity as brine volume fraction (vbf(S)) + oil volume fraction (ovf)
for col in ic_df.collection.unique():
    prop = ['brine volume fraction']
    _ics = []
    # compute permeability with temperature profile and salinity
    for ic in col.split(', '):
        if 'salinity' in ic_df.loc[ic_df.name == ic, 'variable'].unique() and ic not in _ics:
            _ics.append(ic)
            s_data = ic_df.loc[(ic_df.name == ic) & (ic_df.variable == 'salinity')].copy()
            t_data = ic_df.loc[(ic_df.collection == col) & (ic_df.variable == 'temperature')].copy()
            data = seaice.property.compute_phys_prop_from_core(s_data, t_data, prop, display_figure=False, resize_core=False, prop_name='S')
            plt.show()
            ic_df = ic_df.append(data, sort=False)
    del _ics
print("Brine volume fraction computed from T and S profile")

# (2a) For oil measurement, replace np.nan by 0
ic_df.loc[ic_df.variable == 'oil weight fraction', 'oil weight fraction'] = ic_df.loc[ic_df.variable == 'oil weight fraction', 'oil weight fraction'].fillna(0)

# (2b) Compute oil volume fraction:
density_oil = 0.895
_data = ic_df.loc[ic_df.variable == 'oil weight fraction'].copy()
_data['oil volume fraction'] = _data['oil weight fraction']/density_oil
_data = _data.drop('oil weight fraction', axis=1)
_data['variable'] = 'oil volume fraction'
ic_df = ic_df.append(_data, sort=False)
print("Oil weight fraction computed from oil volume fraction with oil density as %s" % str(density_oil))

# (3) Compute total brine volume fraction TVbf = Vbf + ovf
for ic in ic_df.name.unique():
    ovf = ic_df.loc[(ic_df.name == ic) & (ic_df.variable == 'oil weight fraction'), ['oil volume fraction', 'y_mid']].replace(np.nan, 0).copy()
    bvf = ic_df.loc[(ic_df.name == ic) & (ic_df.variable == 'brine volume fraction'), ['brine volume fraction', 'y_mid']].copy()

    vf = pd.merge(ovf, bvf, how='outer', on='y_mid')
    vf['porosity'] = vf['oil volume fraction'].replace(np.nan, 0)+vf['brine volume fraction']

    # copy brine volume fraction profile
    tvbf = ic_df.loc[(ic_df.name == ic) & (ic_df.variable == 'brine volume fraction')].copy()
    tvbf['porosity'] = vf['porosity']
    tvbf.variable = 'porosity'
    tvbf = tvbf.drop('brine volume fraction', axis =1)
    ic_df = ic_df.append(tvbf, sort=False)
print("Porosity computed as the sum of brine and oil volume fraction")

# (4) Compute permeability from porosity
_k_data = ic_df.loc[ic_df.variable == 'porosity'].copy()
_k_data['variable'] = 'permeability'
_k_data['permeability'] = seaice.property.si.permeability_from_porosity(_k_data.porosity)
_k_data = _k_data.drop('porosity', axis=1)
ic_df = ic_df.append(_k_data, sort=False)
del _k_data
print("Permeability computed from porosity")

ic_df = ic_df.reset_index(drop=True)
ic_df = seaice.core.corestack.CoreStack(ic_df)
ic_df_backup = ic_df.copy()
if DISCRETIZED:
    y_bins = np.arange(0, 0.21, 0.025)
    ic_df = ic_df.discretize(y_bins)
    ic_df['date'] = pd.to_datetime(ic_df['date'])
    print("Profiles are discretized")

# (5) Profile statistics
ic_df = seaice.core.corestack.CoreStack(ic_df.reset_index(drop=True))
groups = ['date', 'ice type']
variables = ['salinity', 'permeability', 'porosity', 'oil volume fraction']
stats = ['min', 'mean', 'max', 'std']
ics_stat = ic_df.section_stat(groups=groups, variables=variables, stats=stats)
ics_stat = ics_stat.append(t_stats, sort=True)

# (6) Compute temperature, salinity and permeability field:
field_data = {}
for ice in ic_df['ice type'].unique():
    # Temperature data
    # select thermistor string according to ice type
    t_data = temperature[ice].set_index('Time')[list(depth_dict[ice].keys())]
    depth_dict_m = {key:depth_dict[ice][key]/100 for key in depth_dict[ice]}
    t_data = t_data.rename(columns=depth_dict_m)
    t_data = t_data.loc[(date_start <= t_data.index) & (t_data.index <= date_end+dt.timedelta(1))]

    # 1-hourly temperature
    t_data = t_data.resample('1H').mean()
    t_field = t_data

    core_data = ic_df[ic_df['ice type'] == ice].copy()
    # salinity
    s_data = pd.pivot_table(core_data[['date', 'name', 'y_low', 'salinity']], values='salinity', index=['date', 'name'], columns='y_low')
    s_field = s_data.pivot_table(index='date', aggfunc=np.nanmean).resample('1D').interpolate()

    ovf_data = pd.pivot_table(core_data[['date', 'name', 'y_low', 'oil volume fraction']], values='oil volume fraction', index=['date', 'name'], columns='y_low').replace(np.nan, 0)
    bvf_data = pd.pivot_table(core_data[['date', 'name', 'y_low', 'brine volume fraction']], values='brine volume fraction', index=['date', 'name'], columns='y_low')
    p_data = ovf_data+bvf_data
    p_field = p_data.pivot_table(index='date', aggfunc=np.nanmean).resample('1D').interpolate()

    k_data = p_data.apply(lambda x: seaice.property.si.permeability_from_porosity(x))
    k_field = k_data.pivot_table(index='date', aggfunc=np.nanmean).resample('1D').interpolate()

    field_data[ice] = [t_data, s_field, p_field, k_field]

# (7) 5 cm thick bin
y_bins = [0, 0.05, 0.10, 0.15, 0.2]

ic_df_backup = seaice.core.corestack.CoreStack(ic_df_backup)

groups = ['ice type', 'date', {'y_mid':y_bins}]
variables = ['oil volume fraction']
stats = ['min', 'mean', 'max', 'std']
owf_stat = ic_df_backup.section_stat(groups=groups, variables=variables, stats=stats)
# ---------------------------------------------------------------------------------------------------------------------#
# FIGURES

# 1. Salinity, Temperature and Permeability profile
for ice in ic_df['ice type'].unique():
    dates = sorted(ic_df.loc[ic_df['ice type'] == ice, 'date'].unique())
    ncols = 6
    nrows = dates.__len__()
    row = 0
    fig, ax = plt.subplots(nrows, ncols, sharey=True, figsize=(10.5, 8))  # landscape
    plt.suptitle(ice)
    for date in dates:
        col = ic_df.loc[(ic_df['ice type'] == ice) & (ic_df.date == date), 'collection'].unique()[0]
        _data = ic_df.loc[ic_df.collection == col]
        # salinity
        for ic in _data.name.unique():
            seaice.core.plot.plot_profile(_data.loc[(_data.variable == 'salinity') & (_data.name == ic)], ax=ax[row][0])
            seaice.core.plot.plot_profile(_data.loc[(_data.variable == 'temperature')], ax=ax[row][1])
            seaice.core.plot.plot_profile(_data.loc[(_data.variable == 'brine volume fraction') & (_data.name == ic)], ax=ax[row][2])
            seaice.core.plot.plot_profile(_data.loc[(_data.variable == 'oil volume fraction') & (_data.name == ic)], ax=ax[row][3])
            seaice.core.plot.plot_profile(_data.loc[(_data.variable == 'porosity') & (_data.name == ic)], ax=ax[row][4])
            seaice.core.plot.semilogx_profile(_data.loc[(_data.variable == 'permeability') & (_data.name == ic)], ax=ax[row][5])
            if row == nrows-1:
                ax[row][0].set_xlabel('salinity (psu)')
                ax[row][1].set_xlabel('temperature (C)')
                ax[row][2].set_xlabel('V$_{brine}$ (-)')
                ax[row][3].set_xlabel('V$_{oil}$ (-)')
                ax[row][4].set_xlabel('$\Phi$ (-)')
                ax[row][5].set_xlabel('$\kappa$ (m$^2$)')
            ax[row][0].set_ylabel('ice thickness (m)')
            ax[row][0].set_xlim([0, 15])
            ax[row][1].set_xlim([-8, -2])
            ax[row][2].set_xlim([0, 0.3])
            ax[row][3].set_xlim([0, 0.1])
            ax[row][4].set_xlim([0, 0.3])
            ax[row][5].set_xlim([1e-12, 1e-9])
            ax[row][0].set_ylim([0, 0.2])
        row += 1
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'prop-'+ice+'.pdf'))

# 2. Salinity, temperature and permeability statistic profile:
for ice in ics_stat['ice type'].unique():
    dates = sorted(ics_stat.loc[ics_stat['ice type'] == ice, 'date'].unique())
    ncols = 5
    nrows = dates.__len__()
    row = 0
    fig, ax = plt.subplots(nrows, ncols, sharey=True, figsize=(10.5, 8))  # landscape
    plt.suptitle(ice)
    for date in dates:
        col = ic_df.loc[(ic_df['ice type'] == ice) & (ic_df.date == date), 'collection'].unique()[0]
        _data = ic_df.loc[ic_df.collection == col]
        # salinity
        seaice.core.plot.plot_envelop(ics_stat, variable_dict={'variable': 'salinity', 'date': date, 'ice type':ice}, ax=ax[row][0])
        seaice.core.plot.plot_envelop(ics_stat, variable_dict={'variable': 'temperature', 'date': date, 'ice type':ice}, ax=ax[row][1])
        seaice.core.plot.plot_envelop(ics_stat, variable_dict={'variable': 'porosity', 'date': date, 'ice type':ice}, ax=ax[row][2])
        seaice.core.plot.plot_enveloplog(ics_stat, variable_dict={'variable': 'permeability', 'date': date, 'ice type':ice}, ax=ax[row][3])
        seaice.core.plot.plot_envelop(ics_stat, variable_dict={'variable': 'oil volume fraction', 'date': date, 'ice type':ice}, ax=ax[row][4])

        if row == nrows-1:
            ax[row][0].set_xlabel('salinity (psu)')
            ax[row][1].set_xlabel('temperature (C)')
            ax[row][2].set_xlabel('$\Phi$ (-)')
            ax[row][3].set_xlabel('$\kappa$ (m$^2$)')
            ax[row][4].set_xlabel('$V_{f,oil}$ (-)')

        ax[row][0].set_ylabel('ice thickness (m)')
        ax[row][0].set_xlim([0, 15])
        ax[row][1].set_xlim([-8, -2])
        ax[row][2].set_xlim([0, 0.3])
        ax[row][3].set_xlim([1e-12, 1e-9])
        ax[row][4].set_xlim([0, 0.1])
        ax[row][0].set_ylim([0, 0.2])
        row += 1
    plt.savefig(os.path.join(fig_dir, 'prop_stat-'+ice+'.pdf'))
    plt.show()

# 3. Salinity, temperature and permeability field
for ice in ics_stat['ice type'].unique():
    t_data, s_field, p_field, k_field = field_data[ice]

    # t and z label
    # xaxis
    t_label = [date.strftime("%m-%d") for date in s_field.index]
    t_m = np.arange(0, t_label.__len__()+1)
    # yaxis
    z_label = [str("%0.3f" % key) for key in s_field.keys()]
    z_m = np.arange(0, z_label.__len__()+1)

    fig_row = 4
    fig_col = 2

    fig = plt.figure(figsize=[7.5, 11])
    gs = mpl.gridspec.GridSpec(fig_row, fig_col, height_ratios=[1]*4, width_ratios=[0.8, 0.2])
    ax = []
    plt.suptitle(ice)
    ax.append(fig.add_subplot(gs[0, 0]))  # temperature t_data
    ax.append(fig.add_subplot(gs[1, 0]))  # salinity s_data_m
    ax.append(fig.add_subplot(gs[2, 0], sharex=ax[1]))  # porosity p_data_m
    ax.append(fig.add_subplot(gs[3, 0], sharex=ax[1]))  # permeability k_data_m

    # temperature
    CF = ax[0].contourf(t_field.index.values, t_field.keys(), t_field.values.transpose())
    CS = ax[0].contour(t_field.index.values, t_field.keys(), t_field.values.transpose(), levels=[-10, -7.5, -5, -2.5], colors='w')
    ax[0].clabel(CS, fmt="%0.3f", colors='w', inline=True)
    axins = inset_axes(ax[0], width="5%", height="100%", loc=6, bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[0].transAxes, borderpad=0)
    cbar = plt.colorbar(CF, cax=axins, extend='max')
    cbar.add_lines(CS)
    ax[0].set_ylim([0.2, 0])
    ax[0].xaxis.set_visible(False)
    cbar.set_label('temperature (C)')
    ax[0].set_ylabel('h$_i$ from ice/air interface (m)')

    # salinity
    CF = ax[1].pcolor(t_m, z_m, s_field.values.transpose())
    CS = ax[1].contour(t_m[:-1] + 0.5, z_m[:-1] + 0.5, s_field.values.transpose(), levels=[0, 2.5, 5, 7.5],  colors='w')
    ax[1].clabel(CS, fmt="%0.3f", colors='w', inline=True)
    axins = inset_axes(ax[1], width="5%", height="100%", loc=6, bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[1].transAxes, borderpad=0)
    cbar = plt.colorbar(CF, cax=axins, extend='max')
    cbar.add_lines(CS)
    cbar.set_label('salinity (PSU)')
    ax[1].xaxis.set_visible(False)
    ax[1].set_yticks(z_m[:-1])
    ax[1].yaxis.set_ticklabels(z_label)
    ax[1].set_xticks(t_m[:-1] + 0.5)
    ax[1].set_xticklabels(t_label)
    ax[1].set_ylabel('h$_i$ from ice/oil interface (m)')

    # porosity
    CF = ax[2].pcolor(t_m, z_m, p_field.values.transpose())
    CS = ax[2].contour(t_m[:-1] + 0.5, z_m[:-1] + 0.5, p_field.values.transpose(), levels=[0, 0.05, 0.1, 0.2, 0.3], colors='w')
    ax[2].clabel(CS, fmt="%0.3f", colors='w', inline=True)
    axins = inset_axes(ax[2], width="5%", height="100%", loc=6, bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[2].transAxes, borderpad=0)
    cbar = plt.colorbar(CF, cax=axins, extend='max')
    cbar.add_lines(CS)
    cbar.set_label('porosity (-)')
    ax[2].xaxis.set_visible(False)
    ax[2].set_yticks(z_m[:-1])
    ax[2].yaxis.set_ticklabels(z_label)
    ax[2].set_xticks(t_m[:-1] + 0.5)
    ax[2].set_xticklabels(t_label)
    ax[2].set_ylabel('h$_i$ from ice/oil interface (m)')
    
    # permeability
    CF = ax[3].pcolor(t_m, z_m, np.log10(k_field.values.transpose()))
    CS = ax[3].contour(t_m[:-1]+0.5, z_m[:-1]+0.5, np.log10(k_field.values.transpose()), levels=[-12, -11, -10],  colors='w')
    ax[3].clabel(CS, fmt="%0.3f", colors='w', inline=True)
    axins = inset_axes(ax[3], width="5%", height="100%", loc=6, bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[3].transAxes, borderpad=0)
    cbar = plt.colorbar(CF, cax=axins, extend='max')
    cbar.add_lines(CS)
    cbar.set_label('log$_{10}$(permeability) (log$_{10}$(m$^2$)')
    ax[3].set_yticks(z_m[:-1])
    ax[3].yaxis.set_ticklabels(z_label)
    ax[3].set_xticks(t_m[:-1] + 0.5)
    ax[3].set_xticklabels(t_label)
    ax[3].set_ylabel('h$_i$from ice/oil interface (m)')
    plt.savefig(os.path.join(fig_dir, 'prop_field-'+ice+'.pdf'))
    plt.show()

# 4. Plot oil volume fraction (mean +/- std deviation over permeability
row = 4
y_max = 0.05
fig = plt.figure()
for ice in owf_stat['ice type'].unique():
    print(ice)
    ax_n = 0
    for y_mid in owf_stat.loc[owf_stat['ice type'] == ice].y_mid.unique():
        print(ax_n, y_mid)
        _mean = owf_stat.loc[(owf_stat['ice type'] == ice) & (owf_stat.y_mid == y_mid) & (owf_stat.stats == 'mean'), ['date', 'oil volume fraction']].set_index('date').sort_index().replace(np.nan, 0)+ax_n*y_max
        _std = owf_stat.loc[(owf_stat['ice type'] == ice) & (owf_stat.y_mid == y_mid) & (owf_stat.stats == 'std'), ['date', 'oil volume fraction']].set_index('date').sort_index().replace(np.nan, 0)
        if ice == 'granular':
            plt.errorbar(_mean.index, _mean['oil volume fraction'], yerr=_std['oil volume fraction'], color='r')
        else:
            plt.errorbar(_mean.index, _mean['oil volume fraction'], yerr=_std['oil volume fraction'], color='b')
        plt.plot(_mean.index, [ax_n*y_max]*_mean.index.__len__(), 'k:')
        ax_n += 1
plt.ylim([0, 0.2])
plt.xlim([_mean.index.min(), _mean.index.max()])
plt.title(ice)
plt.show()



for ice in ics_stat['ice type'].unique():
    t_data, s_field, p_field, k_field = field_data[ice]

    # t and z label
    # xaxis
    t_label = [date.strftime("%m-%d") for date in s_field.index]
    t_m = np.arange(0, t_label.__len__()+1)
    # yaxis
    z_label = [str("%0.3f" % key) for key in s_field.keys()]
    z_m = np.arange(0, z_label.__len__()+1)

    fig_row = 1
    fig_col = 2

    fig = plt.figure(figsize=[11, 7.5])
    gs = mpl.gridspec.GridSpec(fig_row, fig_col, height_ratios=[1], width_ratios=[0.8, 0.2])
    ax = []
    plt.suptitle(ice)
    ax.append(fig.add_subplot(gs[0, 0]))  # temperature t_data

    # permeability
    CF = ax[0].pcolor(t_m, z_m, np.log10(k_field.values.transpose()))
    CS = ax[0].contour(t_m[:-1]+0.5, z_m[:-1]+0.5, np.log10(k_field.values.transpose()), levels=[-12, -11, -10],  colors='w')
    ax[0].clabel(CS, fmt="%0.3f", colors='w', inline=True)
    axins = inset_axes(ax[0], width="5%", height="100%", loc=6, bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax[0].transAxes, borderpad=0)
    cbar = plt.colorbar(CF, cax=axins, extend='max')
    cbar.add_lines(CS)
    cbar.set_label('log$_{10}$(permeability) (log$_{10}$(m$^2$)')


    # add oil volume fraction layers
    ax_n = 0
    scale = z_m[:-1].__len__()/0.2
    for y_mid in owf_stat.loc[owf_stat['ice type'] == ice].y_mid.unique():
        _mean = owf_stat.loc[(owf_stat['ice type'] == ice) & (owf_stat.y_mid == y_mid) & (owf_stat.stats == 'mean'),
                             ['date', 'oil volume fraction']].sort_values(by='date').fillna(0)
        _mean['dt'] = pd.to_numeric(_mean['date']-_mean['date'].min())/(24*3600*1e9)+0.5
        _mean =(_mean.set_index('dt', drop=True).drop('date', axis=1)+ax_n*y_max)*scale

        _std = owf_stat.loc[(owf_stat['ice type'] == ice) & (owf_stat.y_mid == y_mid) & (owf_stat.stats == 'std'),
                            ['date', 'oil volume fraction']].sort_values(by='date').fillna(0)
        _std['dt'] = pd.to_numeric(_std['date']-_std['date'].min())/(24*3600*1e9)+0.5
        _std =(_std.set_index('dt', drop=True).drop('date', axis=1))*scale

        ax[0].errorbar(_mean.index, _mean['oil volume fraction'], yerr=_std['oil volume fraction'], color='k')
        ax[0].plot(t_m, [ax_n*0.05*scale]*t_m.__len__(), ":k")
        ax_n += 1
    ax[0].set_yticks(z_m[:-1])
    ax[0].yaxis.set_ticklabels(z_label)
    ax[0].set_xticks(t_m[:-1] + 0.5)
    ax[0].set_xticklabels(t_label)
    ax[0].set_ylim([0, 8])
    ax[0].set_ylabel('h$_i$from ice/oil interface (m)')
    plt.savefig(os.path.join(fig_dir, 'ovf-k_field-' + ice + '.pdf'))
    plt.show()


# 5. Salinity, temperature and permeability field
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[10, 5])
ax_n = 0
for ice in ics_stat['ice type'].unique():
    dates = sorted(ic_df.loc[ic_df['ice type'] == ice, 'date'].unique())
    for date in dates:
        _data = ic_df.loc[(ic_df['ice type'] == ice) & (ic_df.date == date)]
        _data = _data[_data['oil volume fraction'] != 0]
        k = _data.loc[_data.variable == 'permeability', ['permeability', 'name', 'y_low']]
        ovf = _data.loc[_data.variable == 'oil volume fraction', ['oil volume fraction', 'name', 'y_low']]
        k_ovf = pd.merge(k, ovf, on=['name', 'y_low'])
        k_ovf_g = k_ovf.loc[k_ovf.y_low<0.1]
        ax[ax_n].semilogx(k_ovf_g['permeability'], k_ovf_g['oil volume fraction'], 'o', markeredgecolor='grey', markerfacecolor='w',
                          label='h$_i$ < 0.1')
        ax[ax_n].semilogx(k_ovf['permeability'], k_ovf['oil volume fraction'], 'x',
                          label=pd.to_datetime(str(date)).strftime('%m-%d'))
    ax[0].set_ylabel('oil volume fraction (-)')
    ax[ax_n].set_xlabel('permeability')
    ax[0].set_title(ice)
    ax[0].legend()
    ax[1].yaxis.set_visible(False)
    ax_n += 1
plt.savefig(os.path.join(fig_dir, 'ovf_k.pdf'))
plt.show()

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[10, 5])
ax_n = 0
for ice in ics_stat['ice type'].unique():
    dates = sorted(ic_df.loc[ic_df['ice type'] == ice, 'date'].unique())
    for date in dates:
        _data = ic_df.loc[(ic_df['ice type'] == ice) & (ic_df.date == date)]
        _data = _data[_data['oil volume fraction'] != 0]
        k = _data.loc[_data.variable == 'porosity', ['porosity', 'name', 'y_low']]
        ovf = _data.loc[_data.variable == 'oil volume fraction', ['oil volume fraction', 'name', 'y_low']]
        k_ovf = pd.merge(k, ovf, on=['name', 'y_low'])
        k_ovf_g = k_ovf.loc[k_ovf.y_low<0.1]
        ax[ax_n].plot(k_ovf_g['porosity'], k_ovf_g['oil volume fraction'], 'o', markeredgecolor='grey', markerfacecolor='w',
                          label='h$_i$ < 0.1')
        ax[ax_n].plot(k_ovf['porosity'], k_ovf['oil volume fraction'], 'x',
                          label=pd.to_datetime(str(date)).strftime('%m-%d'))
    ax[0].set_ylabel('oil volume fraction (-)')
    ax[ax_n].set_xlabel('porosity (-)')
    ax[0].set_title(ice)
    ax[0].legend()
    ax[1].yaxis.set_visible(False)
    ax_n += 1
plt.savefig(os.path.join(fig_dir, 'p_k.pdf'))
plt.show()
