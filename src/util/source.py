from pyrocko import util, model, io, trace, config
from pyrocko.gf import Target, DCSource, RectangularSource, PorePressureLineSource, PorePressurePointSource, VLVDSource, MTSource, ExplosionSource
from pyrocko import gf
import numpy as num
from pyrocko import moment_tensor as pmt
km = 1000.

def rand_source(event, SourceType="MT", pressure=None, volume=None):

    if event.moment_tensor is None:
        mt = pmt.MomentTensor.random_dc(magnitude=event.magnitude)
        event.moment_tensor = mt
    else:
        mt = event.moment_tensor

    if SourceType == "MT":
        source = MTSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            m6=mt.m6(),
            time=event.time)

    if SourceType == "explosion":
        source = ExplosionSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            time=event.time,
            moment=mt.moment)

    if SourceType == "VLVD":
        if volume is None or volume == 0:
            volume = num.random.uniform(0.001, 10000)
            pressure = pressure

        source = VLVDSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            azimuth=mt.strike1,
            dip=mt.dip1,
            time=event.time,
            volume_change=volume, # here synthetic volume change
            clvd_moment=mt.moment) # ?

    if SourceType == "PorePressurePointSource":
        source = PorePressurePointSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            pp=num.random.uniform(1,1),  # here change in pa
            time=event.time) # ?

    if SourceType == "PorePressureLineSource":
        source = PorePressureLineSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            azimuth=event.strike,
            dip=mt.dip1,
            pp=num.random.uniform(1, 1), # here change in pa
            time=event.time,
            length=num.random.uniform(1, 20)*km) # scaling!)

    if SourceType == "rectangular":
        length = num.random.uniform(0.0001, 0.2)*km
        width = num.random.uniform(0.0001, 0.2)*km
        nuc_x = num.random.uniform(-1, 1)
        nuc_y = num.random.uniform(-1, 1)

        strike, dip, rake = pmt.random_strike_dip_rake()
        event.moment_tensor = pmt.MomentTensor(strike=strike, dip=dip,
                                               rake=rake)
        source = RectangularSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            strike=strike,
            nucleation_x=nuc_x,
            nucleation_y=nuc_y,
            dip=dip,
            rake=rake,
            length=length,
            width=width,
            time=event.time,
            magnitude=event.magnitude)
        source = RectangularSource(
            depth=1.6*km,
            strike=240.,
            dip=76.6,
            rake=-.4,
            anchor='top',

            nucleation_x=-.57,
            nucleation_y=-.59,
            velocity=2070.,

            length=27*km,
            width=9.4*km,
            slip=1.4)

    return source, event
