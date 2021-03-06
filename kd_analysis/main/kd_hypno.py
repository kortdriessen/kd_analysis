import numpy as np
import pandas as pd
import tdt
import xarray as xr
import yaml
from pathlib import Path
import hypnogram as hp
from hypnogram import DatetimeHypnogram
import xarray as xr
import kd_analysis.xrsig as xrsig
import kd_analysis.xrsig.hypnogram_utils as xrhyp
import kd_analysis.main.kd_plotting as kp
from scipy.stats import mode
from scipy.ndimage.filters import gaussian_filter1d

def _infer_bout_start(df, bout):
    """Infer a bout's start time from the previous bout's end time.

    Parameters
    ----------
    h: DataFrame, (n_bouts, ?)
        Hypogram in Visbrain format with 'start_time'.
    row: Series
        A row from `h`, representing the bout that you want the start time of.

    Returns
    -------
    start_time: float
        The start time of the bout from `row`.
    """
    if bout.name == 0:
        start_time = 0.0
    else:
        start_time = df.loc[bout.name - 1].end_time

    return start_time


def load_hypno_file(path, st):
    """Load a Visbrain formatted hypnogram."""
    df = pd.read_csv(path, sep="\t", names=["state", "end_time"], comment="*")
    df["start_time"] = df.apply(lambda row: _infer_bout_start(df, row), axis=1)
    df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
    return to_datetime(df, st)


def to_datetime(df, start_datetime):
    df = df.copy()
    df["start_time"] = start_datetime + pd.to_timedelta(df["start_time"], "s")
    df["end_time"] = start_datetime + pd.to_timedelta(df["end_time"], "s")
    df["duration"] = pd.to_timedelta(df["duration"], "s")
    return hp.DatetimeHypnogram(df)

def load_hypnograms(subject, experiment, condition, scoring_start_time, hypnograms_yaml_file="/Volumes/paxilline/Data/paxilline_project_materials/pax-hypno-paths.yaml"):
    
    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)

    root = Path(yaml_data[subject]["hypno-root"])
    hypnogram_fnames = yaml_data[subject][experiment][condition]
    hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]

    hypnogram_start_times = pd.date_range(
        start=scoring_start_time, periods=len(hypnogram_paths), freq="7200S"
    )
    hypnograms = [
        hp.load_visbrain_hypnogram(path).as_datetime(start_time)
        for path, start_time in zip(hypnogram_paths, hypnogram_start_times)
    ]

    return pd.concat(hypnograms).reset_index(drop=True)

def add_states(dat, hypnogram):
    """Annotate each timepoint in the dataset with the corresponding state label.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`.
    hypnogram: DatetimeHypnogram

    Returns:
    --------
    xarray object with new coordinate `state` on dimension `datetime`.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    states = hypnogram.get_states(dat.datetime)
    return dat.assign_coords(state=("datetime", states))


def keep_states(dat, hypnogram, states):
    """Select only timepoints corresponding to desired states.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`
    hypnogram: DatetimeHypnogram
    states: list of strings
        The states to retain.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    try:
        assert "datetime" in dat.dims, "Data must contain datetime dimension."
    except:
        dat = dat.swap_dims({"time": "datetime"})
    keep = hypnogram.keep_states(states).covers_time(dat.datetime)
    return dat.sel(datetime=keep)


def keep_hypnogram_contents(dat, hypnogram):
    """Select only timepoints covered by the hypnogram.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`
    hypnogram: DatetimeHypnogram
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    keep = hypnogram.covers_time(dat.datetime)
    return dat.sel(datetime=keep)