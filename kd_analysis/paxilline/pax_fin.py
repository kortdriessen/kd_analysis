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

bp_def = dict(sub_delta = (0.5, 2), delta=(0.5, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), omega=(300, 700))


def pax_path(sub, x):
    path = '/Volumes/paxilline/Data/'+sub+'/'+sub+'_TANK/'+sub+'-'+x
    return path

def get_paths(sub, xl):
    paths = {}
    for x in xl:
        path = pax_path(sub, x)
        paths[x] = path
    return paths

def get_pax_hypnos(subject, key_list, start_times=None, save=False):
    h = {}
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
        n = subject[4]
        name='p'+n+'h'
        save_dataset(h, name, key_list=key_list)
    return h

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

def load_complete_muscle_from_blocks(info_dict, store='EMGr', chans=[1,2], time=4, window_length=8, overlap=1, start_times=None, save=False):
    spg_dict = {}
    data_dict = {}
    key_list = info_dict['complete_key_list']
    path_dict = get_paths(info_dict['subject'], key_list)
    
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
            data_dict[key], spg_dict[key] = kd.get_data_spg(path_dict[key], store=store, t1=start, t2=stop, channel=chans, sev=False, window_length=window_length, overlap=overlap).sel(channel=1)
            data_dict[key] = data_dict[key].sel(channel=1)
            spg_dict[key] = spg_dict[key].sel(channel=1) 
    if save == True:
        sub_num = spg_dict['sub'][4]
        prefix = 'p'+sub_num
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
    analysis_root = '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data_complete/'+folder+'/' if folder is not None else '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data_complete/'

    for key in keys:
        try:
            path = analysis_root + (name + "_" + key + ".nc") 
            ds[key].to_netcdf(path)
        except AttributeError:
            print('excepting attribute error, trying to save as .tsv (i.e. saving hypnogram)')
            path = analysis_root + (name + "_" + key + ".tsv") 
            ds[key].write(path)


def load_saved_dataset(subject_info, set_name, folder=None):
    """
    Used to load either a spectrogram set, or a hypnogram set, as saved by kd.save_xset()
    -------------------------------------------------------------------------------------
    """
    data_set = {}
    subject = subject_info['subject']
    path_root = '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data_complete/'+folder+'/' if folder is not None else '/Volumes/paxilline/Data/paxilline_project_materials/analysis_data_complete/'
    
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

"""
MAIN PLOT #1 --> DELTA-BP AS % OF BASELINE OVER COURSE OF ENTIRE REBOUND
------------------------------------------------------------------------
"""
def simple_shaded_bp(bp,
                     hyp,
                     ax,
                     ss=12,
                     color='k',
                     linewidth = 2):
    """ 
    This is just a plotting function, does not do any calculation except, if ss 
    is specified, smooth a copy of the bandpower array for display purposes"""
    if ss:
        bp = kd.get_smoothed_da(bp, smoothing_sigma=ss)
    ax = sns.lineplot(x=bp.datetime, y=bp, color=color, linewidth=linewidth, ax=ax)
    kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax

def get_bp_rel(data, comp, comp_hyp, comp_state):
    if comp_state is not None:
        comp = kh.keep_states(comp, comp_hyp, comp_state)
    
    data_bp = kd.get_bp_set2(data, bp_def)
    comp_bp = kd.get_bp_set2(comp, bp_def)
    
    comp_mean = comp_bp.mean(dim='datetime')
    
    data_rel = (data_bp/comp_mean)*100
    
    return data_rel

def bp_pair_plot(bp1,
                 bp2,
                 h1,
                 h2,
                 names=['name1', 'name2']):
    
    chans = bp1.channel.values
    fig_height = (10/3)*len(chans)
    fig, axes = plt.subplots(nrows=len(chans), ncols=2, figsize=(40, fig_height), sharex='col', sharey='row')
    
    lx1 = fig.add_subplot(111, frameon=False)
    lx1.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    lx1.set_ylabel("Delta Power as % of Baseline NREM Mean", fontsize=16, fontweight='bold')
    
    for chan in chans:
        simple_shaded_bp(bp1.sel(channel=chan), h1, axes[chan-1, 0])
        axes[chan-1, 0].set_title(names[0]+', Ch-'+str(chan), fontweight='bold')
        axes[chan-1, 0].set_ylabel(' ')
        axes[chan-1, 0].axhline(y=100, linestyle='--', linewidth=1.5, color='k')

        simple_shaded_bp(bp2.sel(channel=chan), h2, axes[chan-1, 1], color='firebrick')
        axes[chan-1, 1].set_title(names[1]+', Ch-'+str(chan), fontweight='bold', color='darkred')
        axes[chan-1, 1].axhline(y=100, linestyle='--', linewidth=1.5, color='firebrick')
        
        #plt.subplots_adjust(wspace=0, hspace=0.25)
    return fig, axes 

def bp_plot_set(x, spg, hyp):
    # define names
    bl1 = x[0]+'-bl'
    bl2 = x[1]+'-bl'
    
    #Get relative bandpower states
    bp_rel1 = get_bp_rel(spg[x[0]], spg[bl1], hyp[bl1], ['NREM']).delta
    bp_rel2 = get_bp_rel(spg[x[1]], spg[bl2], hyp[bl2], ['NREM']).delta

    # Plot
    fig, axes = bp_pair_plot(bp_rel1, bp_rel2, hyp[x[0]], hyp[x[1]], names=x)
    plt.tight_layout(pad=2, w_pad=0)
    fig.suptitle(spg['sub']+', '+x[2]+' | '+spg['dtype']+' | Sleep Rebound as % of Baseline | Delta Bandpower (0.5-4Hz) | '+spg['x-time']+' Rebound', x=0.52, y=1, fontsize=20, fontweight='bold')
    plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/DELTA_BP-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)



"""
MAIN PLOT #2 --> PSD AS % OF BASELINE - RAW COMPARISON (0-40Hz), AND ACROSS FREQUENCY BANDS
-------------------------------------------------------------------------------------------
"""
def get_state_spectrogram(spg, hyp, state):
    return spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime))

def get_state_psd(spg, hyp, state):
    return get_state_spectrogram(spg, hyp, state).median(dim="datetime")

def get_psd_rel(x, spg, hyp):
    # define names
    bl1 = x[0]+'-bl'
    bl2 = x[1]+'-bl'
    
    #calc the PSD's
    x1_psd = get_state_psd(spg[x[0]], hyp[x[0]], ['NREM'])
    x2_psd = get_state_psd(spg[x[1]], hyp[x[1]], ['NREM'])
    
    bl1_psd = get_state_psd(spg[bl1], hyp[bl1], ['NREM'])
    bl2_psd = get_state_psd(spg[bl2], hyp[bl2], ['NREM'])
    
    rel_psd1 = (x1_psd/bl1_psd)*100
    rel_psd2 = (x2_psd/bl2_psd)*100
    
    return rel_psd1, rel_psd2

def n_freq_bins(da, f_range):
    return da.sel(frequency=slice(*f_range)).frequency.size


def auc_bandpowers(psd1, psd2, freqs):
    auc_df = pd.DataFrame()
    for f in kd.get_key_list(freqs):
        auc = kd.compare_auc(psd1, psd2, freqs[f], title=f)
        auc_df[f] = auc
    return auc_df.drop(labels='omega', axis=1)

def pax_scatter_quantal(df, chan, ax):
    ax.plot(df.index, df, color='k', linewidth=2, linestyle='--')
    ax.scatter(df.index, df, s=100, c='firebrick')
    ax.axhspan(ax.get_ylim()[0]-5, ymax=0, color='firebrick', alpha=0.2)
    ax.axhspan(ymin=0, ymax=ax.get_ylim()[1]+5, color='k', alpha=0.2)
    ax.axhline(y=0, color='k', linewidth=2)
    #ax.set_xlabel('Frequency Band')
    #ax.set_ylabel('Paxilline AUC - Saline AUC | Relative to Baselines')
    ax.set_title('Ch-'+str(chan), fontweight='bold')
    return ax

def psd_comp_quantal(psd1, psd2, keys, ax):
    df = pd.concat([psd1.to_dataframe("power"), psd2.to_dataframe("power")], keys=keys).rename_axis(index={None: 'Condition'})
    ax=sns.lineplot(data=df, x='frequency', y='power', hue='Condition', palette=['k', 'firebrick'], ax=ax)
    return ax

def auc_master_plot(x, spg, hyp):
    r1, r2 = get_psd_rel(x, spg, hyp)
    auc_df = auc_bandpowers(r1, r2, bp_def)
    r1_comp = r1.sel(frequency=slice(0,40))
    r2_comp = r2.sel(frequency=slice(0,40))
    chans = r1.channel.values
    fig_height = (25/6)*len(chans)

    fig, axes = plt.subplots(figsize=(30, fig_height), nrows=len(chans), ncols=2, sharex='col')
    for chan in chans:
        psd1 = r1_comp.sel(channel=chan)
        psd2 = r2_comp.sel(channel=chan)
        ax = axes[chan-1, 0]
        ax = psd_comp_quantal(psd1, psd2, x, ax=ax)
        ax.set(ylabel=' ', xlabel=' ')
        ax.set_title('Ch-'+str(chan), fontweight='bold')
    for chan in auc_df.index:
        pax_scatter_quantal(auc_df.iloc[chan-1], chan, axes[chan-1, 1])
    
    
    # This block sets the ylabels for both columns
    lx1 = fig.add_subplot(121, frameon=False)
    lx1.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    lx1.set_ylabel("PSD as % of BL", fontsize=16, fontweight='bold')
    lx1.set_xlabel("Frequency", fontsize=16, fontweight='bold')
    lx2 = fig.add_subplot(122, frameon=False)
    lx2.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    lx2.set_ylabel("Paxilline AUC - Saline AUC | Relative to Baselines", fontsize=16, fontweight='bold')
    
    plt.tight_layout(pad=1.5, w_pad=2)
    fig.suptitle(spg['sub']+', '+x[2]+' | '+spg['dtype']+' | PSD as % of Baseline | '+spg['x-time']+' Rebound', x=0.52, y=1, fontsize=20, fontweight='bold')
    plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/AUC-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)


hypno_colors = {
    "Wake": "forestgreen",
    "Brief-Arousal": "chartreuse",
    "Transition-to-NREM": "lightskyblue",
    "Transition-to-Wake": "palegreen",
    "NREM": "royalblue",
    "Transition-to-REM": "plum",
    "REM": "magenta",
    "Transition": "grey",
    "Art": "crimson",
    "Wake-art": "crimson",
    "Unsure": "white",
}

def plot_main_pax(s, p, sbl, pbl, band='delta', t1=None, t2=None):
    """Takes state-specfic bandpower sets"""
    if t2 is not None: 
        s = s.isel(time=slice(t1, t2))
        p = p.isel(time=slice(t1, t2))

    #weird that I couldn't accomplish this step with a list... there must be a way to do it
    s = s[[band]].to_array(name=band).squeeze('variable', drop=True)
    p = p[[band]].to_array(name=band).squeeze('variable', drop=True)
    sbl = sbl[[band]].to_array(name=band).squeeze('variable', drop=True)
    pbl = pbl[[band]].to_array(name=band).squeeze('variable', drop=True)
    
    sr2bl = (s / sbl.mean(dim='time')) * 100
    pr2bl = (p / pbl.mean(dim='time')) * 100
    sdf = sr2bl.to_dataframe().assign(condition='Saline')
    pdf = pr2bl.to_dataframe().assign(condition='Paxilline')
    sdf.reset_index(inplace=True)
    pdf.reset_index(inplace=True)
    
    sp = pd.concat([sdf, pdf])
    
    f, ax = plt.subplots(figsize=(9,7))
    sns.boxplot(x="channel", y=band,
            hue="condition", palette=["plum", "gold"],
            data=sp)
    sns.despine(offset=10, trim=True)
    ax.set_xlabel('Channel')
    ax.set_ylabel(band.capitalize() + ' Power as % of BL Mean')
    #ax.set_title('PAX_4 Experiment-2, Full (4-Hr) NREM Delta Rebound as % of BL Mean')
    return ax


def plot_swa_r2bl(s, p, sbl, pbl, band, channel=1, ss=12, title = ''):
    """Takes state-specfifc bandpower sets"""
    s=s.sel(channel=channel)
    p=p.sel(channel=channel)
    sbl=sbl.sel(channel=channel)
    pbl=pbl.sel(channel=channel)
    srel = s[band] / sbl[band].mean(dim='time') * 100
    srels = kd.get_smoothed_da(srel, smoothing_sigma=ss)
    prel = p[band] / pbl[band].mean(dim='time') * 100
    prels = kd.get_smoothed_da(prel, smoothing_sigma=ss)
    sal = srels.to_dataframe().assign(condition='Saline')
    sal.reset_index(inplace=True)
    pax = prels.to_dataframe().assign(condition='Paxilline')
    pax.reset_index(inplace=True)
    
    pax['rel_time'] = np.arange(0, len(pax), 1)
    sal['rel_time'] = np.arange(0, len(sal), 1)
    
    sp = pd.concat([sal, pax])
    
    f, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x='rel_time', y=band, hue='condition', data=sp, palette=["darkorchid", "goldenrod"], dashes=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(band+' Power as % of Baseline')
    ax.set_xlabel("Time (ignore the raw values)")
    return ax