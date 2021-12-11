from pyrocko import pile
import numpy as num


def load_data_archieve(folder, gf_freq, duration=4,
                       tmin=None):
    waveforms = []
    p = pile.make_pile(folder)
    for traces in p.chopper(tmin=tmin, tmax=tmin+duration, tinc=duration):
        for tr in traces:
            if tr.deltat != gf_freq:
                tr.resample_to(gf_freq)
            waveforms.append(tr)
    return waveforms


def get_max_pga(events, folder_waveforms, gf_freq=10., duration=30,
                forerun=10., quantity="pgv", stations=None,
                absolute_max=False):
    if len(events) > 1:
        event = events[0]
    else:
        event = events
    tmin = event.time-forerun
    waveforms, stations_list = load_data_archieve(folder_waveforms,
                                                  gf_freq,
                                                  duration=duration,
                                                  tmin=tmin)
    pgas_waveforms = []
    for i, pile_data in enumerate(waveforms):
        if stations is None:
            stations = stations_list[i]
        pga = []
        for st in stations:
            max_value = 0
            value_E = 0
            value_N = 0
            value_Z = 0
            for tr in pile_data:
                if absolute_max is True:
                    if st.station == tr.station:
                        if quantity == "pga":
                            value = num.max(num.diff(num.diff(tr.ydata)))
                        if quantity == "pgv":
                            value = num.max(num.abs(num.diff(tr.ydata)))
                        if value > max_value:
                            max_value = value
                else:
                    if st.station == tr.station:
                        if tr.channel[-1] == "E" or tr.channel == "R":
                            if quantity == "pga":
                                value_E = num.max(num.diff(num.diff(tr.ydata)))
                            if quantity == "pgv":
                                value_E = num.max(num.abs(num.diff(tr.ydata)))
                        if tr.channel[-1] == "N" or tr.channel == "T":
                            if quantity == "pga":
                                value_N = num.max(num.diff(num.diff(tr.ydata)))
                            if quantity == "pgv":
                                value_N = num.max(num.abs(num.diff(tr.ydata)))
                        if tr.channel[-1] == "Z":
                            if quantity == "pga":
                                value_Z = num.max(num.diff(num.diff(tr.ydata)))
                            if quantity == "pgv":
                                value_Z = num.max(num.abs(num.diff(tr.ydata)))
            max_value = num.sqrt((value_E**2)+(value_N)**2+(value_Z**2))
            pgas_waveforms.append(max_value)
    return pgas_waveforms
