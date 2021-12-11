from collections import defaultdict
import numpy as num
from matplotlib import pyplot as plt
from pyrocko import gf, trace, plot, beachball, util, orthodrome, model
from pyrocko import moment_tensor as pmt
import _pickle as pickle
import matplotlib.cm as cm
import copy
import os
from shake.util import waveform
from shake.util import source as source_shake
from pyrocko.guts import Object, Int, String, Float, StringChoice
from pyrocko.guts_array import Array
from pyrocko import gf
from pyrocko import trace as trd
import cartopy
import cartopy.io.shapereader as shpreader
from cartopy.io import PostprocessedRasterSource, LocatedImage, srtm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
km = 1000.

util.setup_logging('gf_shakemap')
guts_prefix = 'pf'


def coords_2d(norths, easts):
    norths2 = num.repeat(norths, easts.size)
    easts2 = num.tile(easts, norths.size)
    return norths2, easts2


def writeout_values_at_stations(stations_write, values_stations,
                                residuals, folder):

    fobj = open(os.path.join(folder, 'residuals.txt'), 'w')
    for i in range(0, len(residuals)):
        fobj.write('%s %.20f\n' % (stations_write[i],
                                   residuals[i]))
    fobj.close()
    fobj = open(os.path.join(folder, 'synthetic_pga.txt'), 'w')
    for i in range(0, len(residuals)):
        fobj.write('%s %.20f\n' % (stations_write[i],
                                   values_stations[i]))
    fobj.close()


def fill_and_shade(located_elevations):
    """
    Given an array of elevations in a LocatedImage, fill any holes in
    the data and add a relief (shadows) to give a realistic 3d appearance.

    """
    new_elevations = srtm.fill_gaps(located_elevations.image, max_distance=15)
    new_img = srtm.add_shading(new_elevations, azimuth=135, altitude=15)
    return LocatedImage(new_img, located_elevations.extent)


class BruneResponse(trace.FrequencyResponse):
    duration = Float.T()

    def evaluate(self, freqs):
        return 1.0 / (1.0 + (freqs*self.duration)**2)


class ShakeMapQuantity(StringChoice):
    choices = ['pga', 'pgv', 'intensity']


class ShakeMap(Object):
    source = gf.Source.T(optional=True)
    origin = gf.Location.T()
    norths = Array.T(
        dtype=num.float, shape=(None,), serialize_as='base64+meta')
    easts = Array.T(
        dtype=num.float, shape=(None,), serialize_as='base64+meta')
    values = Array.T(
        dtype=num.float, shape=(None, None), serialize_as='base64+meta')
    quantity = ShakeMapQuantity.T(default='pga')


def get_perturbed_mechanisms(source, pertub_degree, n_pertub):
    sources = []
    sources.append(source)
    for i in range(0, n_pertub):
        source_pert = copy.deepcopy(source)
        mts = source_pert.pyrocko_moment_tensor()
        strike, dip, rake = mts.both_strike_dip_rake()[0]
        strike = num.random.uniform(strike-pertub_degree,
                                    strike+pertub_degree)
        dip = num.random.uniform(dip-pertub_degree,
                                 dip+pertub_degree)
        rake = num.random.uniform(rake-pertub_degree,
                                  rake+pertub_degree)
        mtm = pmt.MomentTensor.from_values((strike, dip, rake))
        mtm.moment = mts.moment
        source_pert.mnn = mtm.mnn
        source_pert.mee = mtm.mee
        source_pert.mdd = mtm.mdd
        source_pert.mne = mtm.mne
        source_pert.mnd = mtm.mnd
        source_pert.med = mtm.med
        sources.append(source_pert)


def make_shakemap(engine, source, store_id, folder, stations=None, save=True,
                  stations_corrections_file=None, pertub_mechanism=False,
                  pertub_degree=20, measured=None, n_pertub=0,
                  value_level=0.004, pertub_velocity_model=False,
                  scenario=False, picks=None,
                  folder_waveforms="/home/asteinbe/bgr_data/acquisition",
                  get_measured=True, vs30_topo=False):

    targets, norths, easts, stf_spec = get_scenario(engine,
                                                    source,
                                                    store_id)
    if measured is True:
        event = model.Event(lat=source.lat, lon=source.lon,
                            time=source.time)
        measured = waveform.get_max_pga([event], folder_waveforms,
                                        stations=stations)
    if measured is True and get_measured is False:
        measured = num.genfromtxt(folder+"measured_pgv",
                                  delimiter=',', dtype=None)

    values_pertubed = []
    values_stations_pertubed = []
    if pertub_mechanism is True:
        sources = get_perturbed_mechanisms(source, pertub_degree, n_pertub)
    else:
        sources = [source]
    for source in sources:
        if pertub_velocity_model is True:
            response = engine.process(source, targets)
            values = post_process(response, norths, easts, stf_spec,
                                  savedir=folder,
                                  save=save)
            values_pertubed.append(values)
        else:
            response = engine.process(source, targets)
            values = post_process(response, norths, easts, stf_spec,
                                  savedir=folder,
                                  save=save)
            values_pertubed.append(values)
        if stations is not None:
            targets_stations, norths_stations, easts_stations, stf_spec = get_scenario(engine,
                                                                                       source,
                                                                                       store_id,
                                                                                       stations=stations)

            response_stations = engine.process(source, targets_stations)
            values_stations = post_process(response_stations, norths_stations,
                                           easts_stations,
                                           stf_spec, stations=True,
                                           savedir=folder, save=save)
            values_stations = values_stations[0][0:len(stations)]
            if stations_corrections_file is not None:
                stations_corrections_file = num.genfromtxt(stations_corrections_file,
                                                           delimiter=",",
                                                           dtype=None)
                stations_corrections_value = []
                for i_st, st in enumerate(stations):
                    for stc in stations_corrections_file:
                        # scalar correction
                        if st.station == stc[0].decode():
                            stations_corrections_value.append(float(stc[1]))
                            values_stations[i_st] = values_stations[i_st] + (values_stations[i_st]*float(stc[1]))
            values_stations_pertubed.append(values_stations)

    if stations is not None:
        plot_shakemap(sources, norths, easts, values_pertubed,
                      'gf_shakemap.png', folder,
                      stations,
                      values_stations_list=values_stations_pertubed,
                      norths_stations=norths_stations,
                      easts_stations=easts_stations,
                      value_level=value_level, measured=measured,
                      engine=engine, plot_values=True,
                      store_id=store_id, vs30_topo=vs30_topo)
        if measured is not None:
            plot_shakemap(sources, norths, easts, values_pertubed,
                          'gf_shakemap_residuals.png', folder,
                          stations,
                          values_stations_list=values_stations_pertubed,
                          norths_stations=norths_stations,
                          easts_stations=easts_stations,
                          measured=measured,
                          value_level=value_level, engine=engine,
                          store_id=store_id, vs30_topo=vs30_topo)
    else:
        plot_shakemap(sources, norths, easts, values_pertubed,
                      'gf_shakemap.png', folder,
                      stations, value_level=value_level, engine=engine,
                      store_id=store_id, vs30_topo=vs30_topo)


def get_scenario(engine, source, store_id, extent=30, ngrid=40,
                 stations=None, source_type="brune"):
    '''
    Setup scenario with source model, STF and a rectangular grid of targets.
    '''

    # physical grid size in [m]
    grid_extent = extent*km
    lat, lon = source.lat, source.lon
    # number of grid points
    nnorth = neast = ngrid
    if source_type == "brune":
        try:
            stf_spec = BruneResponse(duration=source.duration)
        except AttributeError:
            stf_spec = BruneResponse(duration=0.5)
    store = engine.get_store(store_id)

    if stations is None:
        # receiver grid
        r = grid_extent / 2.0
        norths = num.linspace(-r, r, nnorth)
        easts = num.linspace(-r, r, neast)
        norths2, easts2 = coords_2d(norths, easts)
        targets = []
        for i in range(norths2.size):
            for component in 'ZNE':
                target = gf.Target(
                    quantity='displacement',
                    codes=('', '%04i' % i, '', component),
                    lat=lat,
                    lon=lon,
                    north_shift=float(norths2[i]),
                    east_shift=float(easts2[i]),
                    store_id=store_id,
                    interpolation='nearest_neighbor')
                # in case we have not calculated GFs for zero distance
                if source.distance_to(target) >= store.config.distance_min:
                    targets.append(target)
    else:
        targets = []
        norths = []
        easts = []
        for i, st in enumerate(stations):
            north, east = orthodrome.latlon_to_ne_numpy(
                lat,
                lon,
                st.lat,
                st.lon,
                )
            norths.append(north[0])
            easts.append(east[0])
            norths2, easts2 = coords_2d(north, east)

            for cha in st.channels:
                target = gf.Target(
                    quantity='displacement',
                    codes=(str(st.network), i, str(st.location),
                           str(cha.name)),
                    lat=lat,
                    lon=lon,
                    north_shift=float(norths2),
                    east_shift=float(easts2),
                    store_id=store_id,
                    interpolation='nearest_neighbor')
                # in case we have not calculated GFs for zero distance
                if source.distance_to(target) >= store.config.distance_min:
                    targets.append(target)
        norths = num.asarray(norths)
        easts = num.asarray(easts)
    return targets, norths, easts, stf_spec


def post_process(response, norths, easts, stf_spec,
                 savedir=None, save=True, quantity="velocity"):
    nnorth = norths.size
    neast = easts.size
    norths2, easts2 = coords_2d(norths, easts)
    by_i = defaultdict(list)
    traces = []
    for source, target, tr in response.iter_results():
        tr = tr.copy()
        if quantity == "velocity":
            trans = trace.DifferentiationResponse(1)
        if quantity == "acceleration":
            trans = trace.DifferentiationResponse(2)

        if quantity is not "displacement":
            trans = trace.MultiplyResponse(
                [trans, stf_spec])
            tr = tr.transfer(transfer_function=trans)
        tr_resamp = tr.copy()
        tr_resamp.resample(tr.deltat*0.25)
        by_i[int(target.codes[1])].append(tr_resamp)
        traces.append(tr)
    values = num.zeros(nnorth*neast)
    plot_trs = []
    for i in range(norths2.size):
        trs = by_i[i]
        if trs:
            ysum = num.sqrt(sum(tr.ydata**2 for tr in trs))
            ymax = num.max(ysum)
            values[i] = ymax
            if norths2[i] == easts2[i]:
                plot_trs.extend(trs)
    values = values.reshape((norths.size, easts.size))
    if save is True:
        path = savedir + '/shakemap.pkl'
        f = open(path, 'wb')
        pickle.dump([values, easts, norths], f)
        f.close()
    return values


def load_shakemap(path):
    path = path + '/shakemap.pkl'
    f = open(path, 'rb')
    values, easts, norths = pickle.load(f)
    f.close()
    return values, easts, norths


def pepare_list(sources, values_list):
    mts = []
    for i, source in enumerate(sources):
        mts.append(source.pyrocko_moment_tensor())
        if i == 0:
            best_mt = source.pyrocko_moment_tensor()
    for i, values_pertubed in enumerate(values_list):
        if i == 0:
            values = values_pertubed
            values_cum = num.zeros(num.shape(values))
            values_cum = values_cum + values
        else:
            values_cum = values_cum + values_pertubed
    values_cum = values_cum/float(len(values_list))
    return mts, values_cum, values, best_mt


def save_with_rasterio(norths, easts, values, fname="synthetic_shakemap.tiff"):
    from rasterio.transform import from_bounds
    import rasterio
    transform = from_bounds(num.min(norths), num.min(easts), num.max(norths),
                            num.max(easts), len(easts), len(norths))
    dataset = rasterio.open(fname, 'w', driver='GTiff',
                            height=len(easts), width=len(norths),
                            count=1, dtype=str(values.dtype),
                            crs={"init": "EPSG:4326"},
                            transform=transform)
    dataset.write(values, 1)
    dataset.close()


def plot_shakemap(sources, norths, easts, values_list, filename, folder,
                  stations,
                  values_stations_list=None, easts_stations=None,
                  norths_stations=None, latlon=True, show=True,
                  plot_background_map=True, measured=None,
                  value_level=0.001, quantity="velocity",
                  scale="mm", plot_values=False, vs30=True,
                  vs30_topo=False,
                  engine=None, type_factors=None, store_id=None,
                  plot_snr=False, save=True, img_puffer=0.2):

    plot.mpl_init()
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
    mts, values_cum, values, ref_mt = pepare_list(sources, values_list)
    if scale == "mm":
        values = values*1000.
    if values_stations_list is not None:
        values_stations = values_stations_list[0]
        if scale == "mm":
            values_stations = values_stations*1000.

    lats_map = []
    lons_map = []
    source = sources[0]
    for east, north in zip(easts, norths):
        lat, lon = orthodrome.ne_to_latlon(source.lat,
                                           source.lon,
                                           north, east)
        lats_map.append(lat)
        lons_map.append(lon)

    if save is True:
        save_with_rasterio(norths, easts, values)

    if vs30_topo is True:
        from shake.util import vs30
        from scipy import ndimage
        values_vs30 = vs30.extract_rectangle(num.min(lons_map),
                                             num.max(lons_map),
                                             num.min(lats_map),
                                             num.max(lats_map))
        factor_x = num.shape(values)[0]/num.shape(values_vs30)[0]
        factor_y = num.shape(values)[1]/num.shape(values_vs30)[1]
        values_vs30_resam = ndimage.zoom(values_vs30, (factor_x, factor_y))
        store = engine.get_store(store_id)
        if type_factors is None:
            layer0 = store.config.earthmodel_1d.layer(0)
            base_velocity = layer0.mtop.vs
            base_velocity = 600.
            type_factors = 1
            amp_factor = (base_velocity/values_vs30_resam)**type_factors
            values = values*amp_factor

    if plot_background_map is True:
        axes = plt.axes(projection=projection)
        gl = axes.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = False
        axes.add_feature(cartopy.feature.OCEAN, zorder=-4)
        axes.add_feature(cartopy.feature.LAND, zorder=-4, edgecolor='black')

    if source.base_key()[6] is not "ExplosionSource":
        plot_kwargs = {
            'size': 0.03,
            'edgecolor': 'black',
            }

        beachball.plot_fuzzy_beachball_mpl_pixmap(
             mts, axes, ref_mt,
             position=(source.lon, source.lat),
             color_t='black',
             zorder=2,
             **plot_kwargs)

    if plot_background_map is True:
        values[values == 0] = 'nan'
        alpha = 0.5
    else:
        alpha = 1.
    _, vmax = num.min(values), num.max(values)

    if values_stations_list is not None:
        st_lats, st_lons = [], []
        for st in stations:
            st_lats.append(st.lat)
            st_lons.append(st.lon)
        if plot_values is True and measured is not None:
            measured_values = []
        try:
            for k, st in enumerate(stations):
                for data in measured:
                    if data[0].decode() == st.station:
                        if scale == "mm":
                            measured_values.append(data[1]*1000.)
            _, vmax = num.min(measured_values), num.max(measured_values)

        except:
            measured_values = []
            if measured is not None:
                for data in measured:
                    if scale == "mm":
                        measured_values.append(data*1000.)
                _, vmax = num.min(measured_values), num.max(measured_values)

        if plot_values is True and measured is None:
            plt.scatter(st_lats, st_lons,
                        c=num.asarray(values_stations), s=36,
                        cmap=plt.get_cmap('YlOrBr'),
                        vmin=0., vmax=vmax, edgecolor="k", alpha=alpha)
            for k, st in enumerate(stations):
                plt.text(st_lats[k], st_lons[k], str(st.station))
        if plot_values is True and measured is not None:
            plt.scatter(st_lats, st_lons,
                        c=num.asarray(measured_values), s=36,
                        cmap=plt.get_cmap('YlOrBr'),
                        vmin=0., vmax=vmax, edgecolor="k", alpha=alpha)
            for k, st in enumerate(stations):
                plt.text(st_lats[k], st_lons[k], str(st.station))

        if measured is not None and plot_values is False:
            residuals = []
            stations_write = []
            try:
                for k, st in enumerate(stations):
                    for data in measured:
                        # case for manual input
                        if data[0].decode() == st.station:
                            residuals.append(values_stations[k]-data[1])
                            stations_write.append(st.station)
            except:
                if measured is not None:
                    for data in measured:
                        # case for measured input
                        residuals.append(values_stations[k]-data)
                        stations_write.append(st.station)

            writeout_values_at_stations(stations_write, values_stations,
                                        residuals, folder)

            plt.scatter(st_lats, st_lons,
                        c=residuals, s=36, cmap=plt.get_cmap('YlOrBr'),
                        vmin=0., vmax=vmax, edgecolor="k", alpha=alpha)

    if quantity == "velocity":
        if scale == "mm":
            m = plt.cm.ScalarMappable(cmap=plt.get_cmap('YlOrBr'))
            m.set_array(values)
            m.set_clim(0., vmax)
            plt.colorbar(m, boundaries=num.linspace(0, vmax, 6))
        else:
            fig.colorbar(im, label='Velocity [m/s]')
    if quantity == "acceleration":
        if scale == "mm":
            fig.colorbar(im, label='Velocity [mm/s]')
        else:
            fig.colorbar(im, label='Velocity [m/s]')

    img_extent = (num.min(lons_map)-img_puffer, num.max(lons_map)-img_puffer,
                  num.min(lats_map)+img_puffer, num.max(lats_map)+img_puffer)
    axes.contour(lons_map, lats_map, values, cmap='brg',
                 levels=[value_level],
                 transform=ccrs.PlateCarree(), zorder=0)
    axes.contourf(lons_map, lats_map, values,
                  transform=ccrs.PlateCarree(), zorder=-1,
                  vmin=0., vmax=vmax,
                  cmap=plt.get_cmap('YlOrBr'), extent=img_extent)

    fig.savefig(folder+filename)
    if show is True:
        plt.show()
    else:
        plt.close()


def fwd_shakemap_post(projdir, wanted_start=0,
                      store_id="insheim_100hz", gf_store_superdirs=None,
                      n_pertub=0, pertub_degree=20,
                      pertub_velocity_model=False,
                      pertub_mechanism=False,
                      value_level=0.004,
                      measured=True,
                      strike=None,
                      dip=None,
                      rake=None,
                      moment=None,
                      depth=None,
                      plot_directivity=False,
                      source_type="MT",
                      vs30_topo=False,
                      stations_corrections_file=None):

    # Load engine
    if gf_store_superdirs is None:
        engine = gf.LocalEngine(use_config=True)
    else:
        engine = gf.LocalEngine(store_superdirs=[gf_store_superdirs])

    event = model.load_events(projdir+"/event.txt")[0]

    # Set event parameters if strike/dip/rake
    if strike is not None:
        mtm = pmt.MomentTensor.from_values((strike, dip, rake))
        event.moment_tensor.mnn = mtm.mnn
        event.moment_tensor.mee = mtm.mee
        event.moment_tensor.mdd = mtm.mdd
        event.moment_tensor.mne = mtm.mne
        event.moment_tensor.mnd = mtm.mnd
        event.moment_tensor.med = mtm.med
        event.moment_tensor.moment = moment
    if depth is not None:
        event.depth = depth
    if moment is not None:
        event.moment_tensor.moment = moment
    if moment is None:
        try:
            moment = event.moment_tensor.moment
        except:
            pass
    source, event = source_shake.rand_source(event, SourceType=source_type)

    # Load stations if available
    try:
        stations = model.load_stations(projdir+"stations.pf")
    except:
        stations = None

    # Check if pertub flag is wrongfully not set
    if n_pertub != 0:
        pertub_mechanism = True

    if plot_directivity is True:
        from pyrocko.plot.directivity import plot_directivity
        resp = plot_directivity(
            engine, source, store_id,
            distance=100*km, dazi=5., component='R',
            plot_mt='full', show_phases=True,
            phases={
                'First': 'first{stored:begin}-10%',
                'Last': 'last{stored:end}+20'
            },
            quantity='displacement', envelope=True)

    else:
        make_shakemap(engine, source, store_id,
                      projdir, stations=stations,
                      n_pertub=n_pertub,
                      pertub_mechanism=pertub_mechanism,
                      pertub_degree=pertub_degree,
                      pertub_velocity_model=pertub_velocity_model,
                      value_level=value_level,
                      measured=measured,
                      vs30_topo=vs30_topo,
                      stations_corrections_file=stations_corrections_file)
