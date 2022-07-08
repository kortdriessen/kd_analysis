import numpy as np
import pandas as pd
import tdt
import xarray as xr
import kd_analysis.signal.timefrequency as tfr
import streamlit as st

bd = {}
bd['delta'] = slice(0.75, 4.1)
bd['theta'] = slice(4.1, 8.1)
bd['alpha'] = slice(8.1, 13.1)
bd['sigma'] = slice(11.1, 16.1)
bd['beta'] = slice(13.1, 30.1)
bd['gamma'] = slice(30.1, 100.1)

class spectral(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(spectral, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return spectral

    def ch(self, chans):
        """Return all bouts of the given states.

        Parameters:
        -----------
        states: list of str
        """
        return self[self.channel.isin(chans)]

    def ts(self, slice_obj):
        """return dataframe values contained in slice_obj
        there should be a column called 'datetime' 
        """
        self_dt = self.set_index('datetime')
        self_dt = self_dt.loc[slice_obj]
        return self_dt.reset_index()
    
    def bp_melt(self, bp_def=bd):
        """Melts a bandpower set to long-form.

        Parameters:
        -----------
        bp_def: bandpower dictionary
        """
        bp_melt = pd.melt(self, id_vars=['datetime', 'channel'], value_vars=list(bp_def.keys()))
        bp_melt.columns = ['datetime', 'channel', 'Band', 'Bandpower']
        return bp_melt
    
    def filt_state(self, states=['NREM']):
        """Filters a dataframe based on the state column

        Parameters:
        -----------
        bp_def: bandpower dictionary
        """
        return self[self.state.isin(states)].reset_index(drop=True)

#@st.cache()
def tdt_to_pandas(path, t1=0, t2=0, channel=None, store=''):
    # Get the basic info needed from the TDT file:
    data = tdt.read_block(path, t1=t1, t2=t2, store=store, channel=channel)
    store = data.streams[store]
    info = data.info
    chan_cols = list(str(chan) for chan in channel)

    # Convert the TDT times to datetime objects:
    n_channels, n_samples = store.data.shape
    time = np.arange(0, n_samples) / store.fs
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta
    
    # Convert this data to a pandas dataframe. Each channel gets a column, datetime is the index:
    volts_to_microvolts = 1e6
    df = pd.DataFrame(store.data.T*volts_to_microvolts, columns=chan_cols)
    df['datetime'] = datetime
    df['timedelta'] = timedelta
    df['tdt_time'] = time
    #df = df.set_index('datetime')
    df.fs = store.fs
    return df

#@st.cache()
def pd_spg(df, window_length=4, overlap=2, **kwargs):
    
    # Get the raw data from the dataframe:
    d = df.drop(['tdt_time', 'timedelta', 'datetime'], axis=1, inplace=False)
    raw_data = d.to_numpy()
    fs = df.fs
    chans = list(d)
    chans_int = [int(i) for i in chans]
    
    # Compute the spectral powergram:
    kwargs['nperseg'] = int(window_length * fs) # window length in number of samples
    kwargs['noverlap'] = int(overlap * fs) # overlap in number of samples
    kwargs['f_range'] = [0, 100] # frequency range to compute the spectrogram
    freqs, spg_time, spg = tfr.parallel_spectrogram_welch(
    raw_data, fs, **kwargs
    )
    tdt_time = df['tdt_time'].min() + spg_time
    timedelta = df['timedelta'].min() + pd.to_timedelta(spg_time, "s")
    datetime = df['datetime'].min() + pd.to_timedelta(spg_time, "s")
    xrda = xr.DataArray(
        spg,
        dims=("frequency", "datetime", "channel"),
        coords={
            "frequency": freqs,
            "datetime": datetime,
            "channel": chans_int,
            "timedelta": ("datetime", timedelta),
            "tdt_time": ("datetime", tdt_time),
        }
    )
    return spectral(xrda.to_dataframe(name='spg').reset_index())


#@st.cache()
def pd_bp(spg_df, band_dict=bd):
    "expects a spectrogram dataframe from pd_spg"
    td_ix = np.repeat(pd.unique(spg_df['timedelta']), len(pd.unique(spg_df['channel'])))
    spg_df = spg_df.set_index(['frequency', 'datetime', 'channel'])

    #Create a new dataframe to hold the bandpower data:
    bp = spg_df.xs(slice(1,2), level='frequency', drop_level=False)
    bp = bp.groupby(level=['datetime', 'channel']).sum()
    bp_df = pd.DataFrame()
    
    #bp_df[['datetime', 'channel']] = spg_df[['datetime', 'channel']]

    # Calculate the power in each band:
    for band in band_dict:
        bp = spg_df.xs(band_dict[band], level='frequency', drop_level=False)
        bp = bp.groupby(level=['datetime', 'channel']).sum()
        bp_df[band] = bp['spg']
    
    bp_df['timedelta'] = td_ix
    return spectral(bp_df.reset_index())

def combine_data_eeg(data, conds, dtype='bp'):
    for key in conds:
        data[key+'-e-'+dtype]['Condition'] = key
    data['concat'] = spectral(pd.concat(list(data[key+'-e-'+dtype] for key in conds)))
    return data

def combine_data_lfp(data, conds, dtype='bp'):
    for key in conds:
        data[key+'-f-'+dtype]['Condition'] = key
    data['concat'] = spectral(pd.concat(list(data[key+'-f-'+dtype] for key in conds)))
    return data

def add_states_to_data(data, hypno):
    dt = data.datetime.values
    states = hypno.get_states(dt)
    data['state'] = states
    return data

## INCOMPLETE -----------------------------------------------------------------------------------------------------------

def filter_data_by_state(data, state):
    return data[data.state == state]

def get_rel_bp_set(bp_set, hyp, times_cond):
    start = bp_set.datetime.values[0]
    t1 = times_cond['stim_on_dt']
    t2 = times_cond['stim_off_dt']
    avg_period = slice(start, t1)
    bp_bl = bp_set.ts(avg_period)