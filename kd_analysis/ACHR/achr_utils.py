import xarray as xr
import hypnogram as hp
import kd_analysis.main.kd_utils as kd


bp_def = dict(delta=(0.5, 4), theta=(4, 8), sigma = (11, 16), beta = (13, 20), low_gamma = (40, 55), high_gamma = (65, 80), omega=(300, 700))

def achr_path(sub, x):
    path = '/Volumes/opto_loc/Data/'+sub+'/'+sub+'_TANK/'+sub+'-'+x
    return path

def get_paths(sub, xl):
    paths = {}
    for x in xl:
        path = achr_path(sub, x)
        paths[x] = path
    return paths

def load_complete_spectroset_from_blocks(info_dict, store, chans, time=4, window_length=8, overlap=1, start_times=None, save=False):
    spg_dict = {}
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict['subject'], key_list)
    
    spg_dict['x-time'] = str(time)+'-Hour'
    spg_dict['sub'] = info_dict['subject']
    spg_dict['dtype'] = 'EEG-Data' if store=='EEGr' else 'LFP-Data'
    
    if start_times==None:
        for key in key_list:
            if key.find('bl') != -1:
                stop=43200
            else:
                stop=(time*3600) #+18000
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
        sub_num = spg_dict['sub'][5]
        prefix = 'a'+sub_num
        spg_post_fix = 'se' if store == 'EEGr' else 'sf'
        spg_name = prefix+spg_post_fix
        save_dataset(spg_dict, spg_name, key_list=key_list)
    return data_dict, spg_dict

def load_complete_muscle_from_blocks(info_dict, store='EMGr', chans=[1,2], time=4, window_length=8, overlap=1, save=False, seperate_sd=False):
    spg_dict = {}
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict['subject'], key_list)
    
    spg_dict['x-time'] = str(time)+'-Hour'
    spg_dict['sub'] = info_dict['subject']
    spg_dict['dtype'] = 'EMG-Data'
    
    for key in key_list:
        if key.find('bl') != -1:
            stop=43200
        else:
            stop=(time*3600)+18000 if seperate_sd==False else time*3600
        data_dict[key], spg_dict[key] = kd.get_data_spg(path_dict[key], store=store, t1=0, t2=stop, channel=chans, sev=True, window_length=window_length, overlap=overlap)
        data_dict[key] = data_dict[key].sel(channel=1)
        spg_dict[key] = spg_dict[key].sel(channel=1)
    if save == True:
        sub_num = spg_dict['sub'][5]
        prefix = 'a'+sub_num
        spg_post_fix = 'sm'
        spg_name = prefix+spg_post_fix
        save_dataset(spg_dict, spg_name, key_list=key_list)
    return data_dict, spg_dict

def save_dataset(ds, name, key_list=None, folder=None):
    """saves each component of an experimental 
    dataset dictionary (i.e. xr.arrays of the raw data and of the spectrograms), 
    as its own separate .nc file. All can be loaded back in as an experimental dataset dictionary
    using fetch_xset
    """
    keys = kd.get_key_list(ds) if key_list == None else key_list
    analysis_root = '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data_complete/'+folder+'/' if folder is not None else '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data_complete/'

    for key in keys:
        try:
            path = analysis_root + (name + "_" + key + ".nc") 
            ds[key].to_netcdf(path)
        except AttributeError:
            pass

def save_hypnoset(ds, name, key_list=None, folder=None):
            keys = kd.get_key_list(ds) if key_list == None else key_list
            analysis_root = '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data_complete/'+folder+'/' if folder is not None else '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data_complete/'
            for key in keys:
                path = analysis_root + (name + "_" + key + ".tsv") 
                ds[key].write(path)


def load_saved_dataset(subject_info, set_name, folder=None):
    """
    Used to load either a spectrogram set, or a hypnogram set, as saved by kd.save_xset()
    -------------------------------------------------------------------------------------
    """
    data_set = {}
    subject = subject_info['subject']
    path_root = '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data_complete/'+folder+'/' if folder is not None else '/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data_complete/'
    
    if set_name.find('h') != -1:
        for key in subject_info['complete_key_list']:
            path = path_root+set_name+'_'+key+'.tsv'
            data_set[key] = hp.load_datetime_hypnogram(path)
        data_set['name'] = set_name
            
    else:
        for key in subject_info['complete_key_list']:
            path = path_root+set_name+'_'+key+'.nc'
            data_set[key] = xr.load_dataarray(path)
        data_set['dtype'] = 'EEG-Data' if set_name.find('se') != -1 else 'LFP-Data'
        data_set['sub'] = subject
        data_set['x-time'] = '4-Hour'
        data_set['name'] = set_name
    return data_set