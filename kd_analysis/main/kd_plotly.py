import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kd_analysis.main.kd_utils as kd
import neurodsp.plts.utils as dspu
import statistics
import plotly as py
import plotly.express as px
import kd_analysis.paxilline.pax_fin as kpx

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
    "Good": "lime"
}





def quantal_shbp(bp, hyp, band='delta'):  
    bp = bp.where(bp<750)
    bp = kd.get_smoothed_ds(bp, smoothing_sigma=8)
    
    bnds = ['delta1', 'delta2', 'delta', 'theta', 'alpha', 'sigma', 'beta', 'low_gamma', 'high_gamma']
    df = bp.to_dataframe().reset_index()
    df = pd.melt(df, id_vars=['channel', 'datetime'], value_vars=bnds, var_name='Band', value_name='Power')
    df = df[df['Band'] == band]
    
    fig = px.line(df, x='datetime', y='Power', template='seaborn', facet_row='channel')
    return fig


    for bout in hyp.itertuples():
            fig.add_vrect(x0=bout.start_time,
                        x1=bout.end_time,
                        fillcolor=hypno_colors[bout.state],
                        opacity=0.25,
                        line_width=0)
    
    

    
    