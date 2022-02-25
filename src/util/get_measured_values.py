from pyrocko import io, trace, model, pile, orthodrome
from pyrocko.guts import Object, Int, String, Float, StringChoice
import numpy as num


class BruneResponse(trace.FrequencyResponse):
    duration = Float.T()

    def evaluate(self, freqs):
        return 1.0 / (1.0 + (freqs*self.duration)**2)


def get_max_pga(events, folder_waveforms, stations, img_extent,
                gf_freq=10., duration=90,
                forerun=100., quantity="pgv",
                absolute_max=False, tmin=None, tmax=None,
                ):
    event = events[0]
    tmin = event.time-forerun
    tmax = event.time+duration
    waveforms = []
    p = pile.make_pile([folder_waveforms])
    for tr in p.chopper(tmin=tmin, tmax=tmax):
        for t in tr:
            t.chop(event.time, event.time+60)
        waveforms.append(tr)
    pgas_waveforms = []
    stf_spec = BruneResponse(duration=0.1)
    # trans_vel = trace.DifferentiationResponse(1)
    # trans_acc = trace.DifferentiationResponse(2)
    # trans_vel = trace.MultiplyResponse(
    #     [trans_vel, stf_spec])
    stations_in_region = []
    for st in stations:
        if st.lon > img_extent[0] and st.lon < img_extent[1] and st.lat >img_extent[2] and  st.lat<img_extent[3]:
            stations_in_region.append(st)

    for st in stations_in_region:
        max_value = 0
        value_E = 0
        value_N = 0
        value_Z = 0
        for tr in waveforms[0]:
            if st.station == tr.station:
                if tr.channel[-1] == "E" or tr.channel == "R":
                    if quantity == "pga":
                        value_E = num.max(num.diff(num.diff(tr.ydata)))
                    if quantity == "pgv":
                        #tr = tr.transfer(transfer_function=trans_vel)
                        tr.highpass(4, 0.25)
                        tr = tr.transfer(transfer_function=trace.DifferentiationResponse(),
                                            demean=False)
                        value_E = num.max(tr.ydata[500:])
                    #    value_E = num.max(num.diff(num.diff(tr.ydata)))
                if tr.channel[-1] == "N" or tr.channel == "T":
                    if quantity == "pga":
                        value_N = num.max(num.diff(num.diff(tr.ydata)))
                    if quantity == "pgv":
                        tr.highpass(4, 0.25)
                        tr = tr.transfer(transfer_function=trace.DifferentiationResponse(),
                                            demean=False)
                        value_N= num.max(tr.ydata[500:])
                        #value_N = num.max(num.diff(num.diff(tr.ydata)))

                if tr.channel[-1] == "Z":
                    if quantity == "pga":
                        value_Z = num.max(num.diff(num.diff(tr.ydata)))
                    if quantity == "pgv":
                        #value_Z = num.max(num.diff(num.diff(tr.ydata)))
                        tr.highpass(4, 0.25)
                        tr = tr.transfer(transfer_function=trace.DifferentiationResponse(),
                                            demean=False)
                        value_Z = num.max(tr.ydata[500:])
            #max_value = num.sqrt((value_E**2)+(value_N)**2+(value_Z**2))
        max_value = abs(num.max([value_E, value_N, value_Z]))
        pgas_waveforms.append(max_value)
        io.save(waveforms[0], "traces_vel.mseed")
    return pgas_waveforms, stations_in_region
