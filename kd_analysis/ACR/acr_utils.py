import xarray as xr
import pandas as pd
import tdt
import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_hypno as kh
import yaml
from pathlib import Path


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
    root = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/' + subject + '/' + 'hypnograms-' + subject + '/'
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

def load_hypno_set(subject, condition, scoring_start_time, hypnograms_yaml_file="/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"):
    
    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)

    root = Path(yaml_data[subject]["hypno-root"])
    hypnogram_fnames = yaml_data[subject][condition]
    hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]

    hypnogram_start_times = pd.date_range(
        start=scoring_start_time, periods=len(hypnogram_paths), freq="7200S"
    )
    hypnograms = [
        hp.load_visbrain_hypnogram(path).as_datetime(start_time)
        for path, start_time in zip(hypnogram_paths, hypnogram_start_times)
    ]

    return pd.concat(hypnograms).reset_index(drop=True)

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


def ss_times(sub, exp, print_=False):
    #Load the relevant times:
    def acr_get_times(sub, exp):
        block_path = '/Volumes/opto_loc/Data/'+sub+'/'+sub+'-'+exp
        ep = tdt.read_block(block_path, t1=0, t2=0, evtype=['epocs'])
        times = {}
        times['bl_sleep_start'] = ep.epocs.Bttn.onset[0]
        times['stim_on'] = ep.epocs.Wdr_.onset[-1]
        times['stim_off'] = ep.epocs.Wdr_.offset[-1]
        dt_start = pd.to_datetime(ep.info.start_date)

        #This get us the datetime values of the stims for later use:
        on_sec = pd.to_timedelta(times['stim_on'], unit='S')
        off_sec = pd.to_timedelta(times['stim_off'], unit='S')
        times['stim_on_dt'] = dt_start+on_sec
        times['stim_off_dt'] = dt_start+off_sec
        return times 
    times = acr_get_times(sub, exp)

    #Start time for scoring is 30 seconds before the button signal was given to inidicate the start of bl peak period. 
    start1 = times['bl_sleep_start'] - 30
    end1 = start1 + 7200
    
    #End time for the second scoring file is when the stim/laser signal is turned off. 
    start2 = end1
    end2 = times['stim_off']
    if print_:
        print('FILE #1'), print(start1), print(end1)
        print('FILE #2'), print(start2), print(end2)
    print('Done loading times')
    return times
