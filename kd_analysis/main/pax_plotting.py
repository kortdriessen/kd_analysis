import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kd_analysis.main.kd_utils as kd

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