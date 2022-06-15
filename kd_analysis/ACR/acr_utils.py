import xarray as xr
import pandas as pd
import tdt
import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_hypno as kh
import yaml


bp_def = dict(delta1=(0.75, 1.75), delta2=(2.5, 3.5), delta=(0.75, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), hz40 = [39, 41])

def achr_path(sub, x):
    path = '/Volumes/opto_loc/Data/'+sub+'/'+sub+'-'+x
    return path

def get_paths(sub, xl):
    paths = {}
    for x in xl:
        path = achr_path(sub, x)
        paths[x] = path
    return paths

def load_hypnograms(subject_info, subtract_sd=False):
    h = {}
    subject = subject_info['subject']
    root = '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/' + subject + '/' + 'hypnograms-' + subject + '/'
    for key in subject_info['hypnos']:
        bp = achr_path(subject, key)
        block = tdt.read_block(bp, store='EEGr', t1=0, t2=1)
        start_time =  pd.to_datetime(block.info.start_date)

        if subtract_sd is not False:
            t = pd.to_timedelta(subtract_sd, 'h')
            start_time = start_time + t 
            
        path = root + 'hypno_' + key + '.txt'
        h[key] = kh.load_hypno_file(path, st=start_time)
    return h

def load_complete_dataset_from_blocks(info_dict, store, chans, start_at=0, time=4):
    """
    start_at --> number of hours into the recording to start
    time --> number of hours total to load
    """
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict['subject'], key_list)
    
    for key in key_list:
        if key.find('bl') != -1:
            start = 0
            stop=43200
        else:
            start = start_at*3600
            stop = start + (time*3600)
        data_dict[key]= kd.get_data(path_dict[key], store=store, t1=start, t2=stop, channel=chans, sev=True) 
    else:
        data_dict['x-time'] = str(time)+'-Hour'
        data_dict['sub'] = info_dict['subject']
        data_dict['dtype'] = 'EEG-Data' if store=='EEGr' else 'LFP-Data' 
        if store == 'EMG_' or store == 'EMGr':
            data_dict['dtype'] = 'EMG-Data'
            for key in key_list:
                data_dict[key] = data_dict[key].sel(channel=1)
            return data_dict
        else:
            return data_dict


def save_dataset(ds, name, key_list=None, folder=None):
    """saves each component of an experimental 
    dataset dictionary (i.e. xr.arrays of the raw data and of the spectrograms), 
    as its own separate .nc file. All can be loaded back in as an experimental dataset dictionary
    using fetch_xset
    """
    keys = kd.get_key_list(ds) if key_list == None else key_list
    analysis_root = '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data/'+folder+'/' if folder is not None else '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data/'

    for key in keys:
        if type(ds[key]) == str:
            pass
        path = analysis_root + (name + "_" + key + ".nc") 
        ds[key].to_netcdf(path)
        

def save_hypnoset(ds, name, key_list=None, folder=None):
            keys = kd.get_key_list(ds) if key_list == None else key_list
            analysis_root = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data_complete/'+folder+'/' if folder is not None else '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data/'
            for key in keys:
                path = analysis_root + (name + "_" + key + ".tsv") 
                ds[key].write(path)


def load_saved_dataset(subject_info, set_name, folder=None, spg=False):
    """
    Used to load a dataset, can calculate hypnogram automatically also
    -------------------------------------------------------------------------------------
    """
    data_set = {}
    subject = subject_info['subject']
    path_root = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data/'+folder+'/' if folder is not None else '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data/'
    
    if set_name.find('h') != -1:
        for key in subject_info['complete_key_list']:
            path = path_root+set_name+'_'+key+'.tsv'
            data_set[key] = hp.load_datetime_hypnogram(path)
        data_set['name'] = set_name
            
    else:
        for key in subject_info['complete_key_list']:
            path = path_root+set_name+'_'+key+'.nc'
            data_set[key] = xr.load_dataarray(path)
        data_set['name'] = set_name
    if spg == True:
        spg_set = kd.get_spg_from_dataset(data_set)
        spg_set['sub'] = subject_info['subject']
        spg_set['dtype'] = 'EEG-Data' if set_name.find('de') != -1 else 'LFP-Data'
        return data_set, spg_set
    else:
        return data_set