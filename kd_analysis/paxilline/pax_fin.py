from statistics import median
import numpy as np
import scipy
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import xarray as xr

import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_plotting as kp
import kd_analysis.main.kd_hypno as kh

bp_def = dict(delta1=(0.75, 1.75), delta2=(2.5, 3.5), delta=(0.75, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90))

def pax_path(sub, x):
    path = '/Volumes/paxilline/Data/'+sub+'/'+sub+'_TANK/'+sub+'-'+x
    return path

def get_paths(info_dict):
    paths = {}
    for x in info_dict['complete_key_list']:
        path = pax_path(info_dict['subject'], x)
        paths[x] = path
    return paths

def get_pax_hypnos(info_dict, start_times=None, save=False):
    h = {}
    subject = info_dict['subject']
    key_list = info_dict['complete_key_list']
    if start_times == None:
        for k in key_list:
            bp = pax_path(subject, k)
            block = tdt.read_block(bp, store='EEGr', t1=0, t2=1)
            start_time =  pd.to_datetime(block.info.start_date)

            #get the experiment:
            if k.find('1') != -1:
                exp = 'exp-1'
            elif k.find('2') != -1:
                exp = 'exp-2'
            elif k.find('3') != -1:
                exp = 'exp-3'
            elif k.find('4') != -1:
                exp = 'exp-4'
            else:
                print('Could not determine experiment for '+k)
                break

            h[k] = kh.load_hypnograms(subject, exp, k, start_time)
    
    elif start_times is not None:
        for k in key_list:
            st = start_times[k]
            bp = pax_path(subject, k)
            block = tdt.read_block(bp, t1=st, t2=st+1)
            start_time = pd.to_datetime(block.info.start_date) + pd.to_timedelta(st, 's')

            #get the experiment:
            if k.find('1') != -1:
                exp = 'exp-1'
            elif k.find('2') != -1:
                exp = 'exp-2'
            elif k.find('3') != -1:
                exp = 'exp-3'
            elif k.find('4') != -1:
                exp = 'exp-4'
            else:
                print('Could not determine experiment for '+k)
                break

            h[k] = kh.load_hypnograms(subject, exp, k, start_time)
    if save == True:
        n = info_dict['subject'][4] if len(info_dict['subject']) == 5 else info_dict['subject'][4:6]
        name='p'+n+'h'
        save_hypno_set(h, name)
    return h

def load_complete_dataset_from_blocks(info_dict, store, time=4, start_times=None, save=False):
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict)
    chans = info_dict['echans']
    if start_times==None:
        for key in key_list:
            if key.find('bl') != -1:
                stop=43200
            else:
                stop=time*3600
            data_dict[key] = kd.get_data(path_dict[key], store=store, t1=0, t2=stop, channel=chans, sev=True)
    else:
        for key in key_list:
            if key.find('bl') != -1:
                start=0
                stop=43200
            else:
                start = start_times[key]
                stop = start + (time*3600)
            data_dict[key] = kd.get_data(path_dict[key], store=store, t1=start, t2=stop, channel=chans, sev=False)
    data_dict['x-time'] = str(time)+'-Hour'
    data_dict['sub'] = info_dict['subject']
    data_dict['dtype'] = 'EEG-Data' if store=='EEGr' else 'LFP-Data'
    if save == True:
        sub_num = data_dict['sub'][4] if len(data_dict['sub']) == 5 else data_dict['sub'][4:6]
        prefix = 'p'+sub_num
        data_post_fix = 'de' if store == 'EEGr' else 'df'
        data_name = prefix+data_post_fix
        save_dataset(data_dict, data_name)

    return data_dict

def load_complete_spectroset_from_blocks(info_dict, store, chans, time=4, window_length=8, overlap=1, start_times=None, save=False):
    spg_dict = {}
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict)
    
    spg_dict['x-time'] = str(time)+'-Hour'
    spg_dict['sub'] = info_dict['subject']
    spg_dict['dtype'] = 'EEG-Data' if store=='EEGr' else 'LFP-Data'
    
    if start_times==None:
        for key in key_list:
            if key.find('bl') != -1:
                stop=43200
            else:
                stop=time*3600
            data_dict[key], spg_dict[key] = kd.get_data_spg(path_dict[key], store=store, t1=0, t2=stop, channel=chans, sev=True, window_length=window_length, overlap=overlap)
    else:
        for key in key_list:
            if key.find('bl') != -1:
                start=0
                stop=43200
            else:
                start = start_times[key]
                stop = start + (time*3600)
            data_dict[key], spg_dict[key] = kd.get_data_spg(path_dict[key], store=store, t1=start, t2=stop, channel=chans, sev=False, window_length=window_length, overlap=overlap) 
    if save == True:
        sub_num = spg_dict['sub'][4]
        prefix = 'p'+sub_num
        spg_post_fix = 'se' if store == 'EEGr' else 'sf'
        spg_name = prefix+spg_post_fix
        save_dataset(spg_dict, spg_name, key_list=key_list)
    return data_dict, spg_dict

def load_complete_muscle_from_blocks(info_dict, store='EMG_', chans=[1,2], time=4, window_length=8, overlap=1, start_times=None, save=False):
    spg_dict = {}
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict)
    
    spg_dict['x-time'] = str(time)+'-Hour'
    spg_dict['sub'] = info_dict['subject']
    spg_dict['dtype'] = 'EMG-Data'
    
    if start_times==None:
        for key in key_list:
            if key.find('bl') != -1:
                stop=43200
            else:
                stop=time*3600
            data_dict[key], spg_dict[key] = kd.get_data_spg(path_dict[key], store=store, t1=0, t2=stop, channel=chans, sev=True, window_length=window_length, overlap=overlap)
            data_dict[key] = data_dict[key].sel(channel=1)
            spg_dict[key] = spg_dict[key].sel(channel=1)
    else:
        for key in key_list:
            if key.find('bl') != -1:
                start=0
                stop=43200
            else:
                start = start_times[key]
                stop = start + (time*3600)
            data_dict[key], spg_dict[key] = kd.get_data_spg(path_dict[key], store=store, t1=start, t2=stop, channel=chans, sev=False, window_length=window_length, overlap=overlap)
            data_dict[key] = data_dict[key].sel(channel=1)
            spg_dict[key] = spg_dict[key].sel(channel=1) 
    if save == True:
        sub_num = spg_dict['sub'][4]
        prefix = 'p'+sub_num
        spg_post_fix = 'sm'
        spg_name = prefix+spg_post_fix
        save_dataset(spg_dict, spg_name, key_list=key_list)
    return data_dict, spg_dict

def save_dataset(ds, name, folder=None):
    """saves each component of an experimental 
    dataset dictionary (i.e. xr.arrays of the raw data and of the spectrograms), 
    as its own separate .nc file. All can be loaded back in as an experimental dataset dictionary
    using fetch_xset
    """
    keys = kd.get_key_list(ds)
    analysis_root = '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data/'+folder+'/' if folder is not None else '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data/'

    for key in keys:
        try:
            path = analysis_root + (name + "_" + key + ".nc") 
            ds[key].to_netcdf(path)
        except AttributeError:
            print('excepting attribute error on save_dataset')
            pass

def save_hypno_set(hypno_set, name, folder=None):
    keys = kd.get_key_list(hypno_set)
    analysis_root = '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data/'+folder+'/' if folder is not None else '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data/'
    for key in keys:
        path = analysis_root + (name + "_" + key + ".tsv") 
        hypno_set[key].write(path)

def load_saved_dataset(subject_info, set_name, folder=None):

    """
    Used to load either a spectrogram set, or a hypnogram set, as saved by kd.save_xset()
    -------------------------------------------------------------------------------------
    """
    data_set = {}
    subject = subject_info['subject']
    path_root = '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data/'+folder+'/' if folder is not None else '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data/'
    
    if set_name.find('h') != -1:
        for key in subject_info['complete_key_list']:
            path = path_root+set_name+'_'+key+'.tsv'
            data_set[key] = hp.load_datetime_hypnogram(path)
        data_set['name'] = set_name
            
    else:
        for key in subject_info['complete_key_list']:
            path = path_root+set_name+'_'+key+'.nc'
            data_set[key] = xr.load_dataarray(path)
        data_set['dtype'] = 'EEG-Data' if set_name.find('e') != -1 else 'LFP-Data'
        data_set['sub'] = subject
        data_set['x-time'] = '4-Hour'
        data_set['name'] = set_name
    return data_set

def load_spg_hypno(info, folder=None):
    spg_name = info['prefix']+'se'
    hypno_name = info['prefix']+'h'

    spg_set = load_saved_dataset(info, spg_name, folder=folder)
    hypno_set = load_saved_dataset(info, hypno_name, folder=None)

    return spg_set, hypno_set

def pax_spg_from_dataset(data_dict, window_length=4, overlap=2, save=False):
    spg_set = {}

    kl = kd.get_key_list(data_dict)
    for key in kl:
        spg_set[key] = kd.get_spextrogram(data_dict[key], window_length, overlap)
    
    if save==True:
        sub_num = data_dict['sub'][4] if len(data_dict['sub']) == 5 else data_dict['sub'][4:6]
        prefix = 'p'+sub_num
        spg_post_fix = 'se' if data_dict['dtype'] == 'EEG-Data' else 'error'
        spg_name = prefix+spg_post_fix
        save_dataset(spg_set, spg_name)

    spg_set['x-time'] = data_dict['x-time']
    spg_set['sub'] = data_dict['sub']
    spg_set['dtype'] = data_dict['dtype']
    return spg_set

def get_bp_rel(data, comp, comp_hyp, comp_state, avg='Median'):
    if comp_state is not None:
        comp = kh.keep_states(comp, comp_hyp, comp_state)
    
    data_bp = kd.get_bp_set2(data, bp_def)
    comp_bp = kd.get_bp_set2(comp, bp_def)

    comp_avg = comp_bp.median(dim='datetime') if avg == 'Median' else comp_bp.mean(dim='datetime')
    
    data_rel = (data_bp/comp_avg)*100

    return data_rel