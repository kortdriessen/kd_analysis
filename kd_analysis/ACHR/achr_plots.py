import numpy as np
import scipy
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_plotting as kp
import kd_analysis.main.kd_hypno as kh
import kd_analysis.paxilline.pax_fin as kpx

bp_def = dict(sub_delta=(0.5, 2), delta=(0.5, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), omega=(300, 700))

def simple_bp_lineplot(bp,
                     ax,
                     ss=12,
                     color='k',
                     linewidth = 2, 
                     hyp=None):
    """ 
    This is just a plotting function, does not do any calculation except, if ss 
    is specified, smooth a copy of the bandpower array for display purposes
    bp --> single channel bandpower data, xr.dataarray

    """
    if ss:
        bp = kd.get_smoothed_da(bp, smoothing_sigma=ss)
    ax = sns.lineplot(x=bp.datetime, y=bp, color=color, linewidth=linewidth, ax=ax)
    if hyp is not None:
        kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax

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

        simple_shaded_bp(bp2.sel(channel=chan), h2, axes[chan-1, 1], color='royalblue')
        axes[chan-1, 1].set_title(names[1]+', Ch-'+str(chan), fontweight='bold', color='darkblue')
        axes[chan-1, 1].axhline(y=100, linestyle='--', linewidth=1.5, color='royalblue')
        
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
    #plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/DELTA_BP-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)
