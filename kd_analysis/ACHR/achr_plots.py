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