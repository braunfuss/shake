from collections import defaultdict
import numpy as num
from matplotlib import pyplot as plt
from pyrocko import gf, trace, plot, beachball, util, orthodrome, model
from pyrocko import moment_tensor as pmt
import _pickle as pickle
import matplotlib.cm as cm
import copy
import os
from shake.util import waveform, get_measured_values
from shake.util import source as source_shake
from pyrocko.guts import Object, Int, String, Float, StringChoice
from pyrocko.guts_array import Array
from pyrocko import gf
from pyrocko import trace as trd
import cartopy
import cartopy.io.shapereader as shpreader
from cartopy.io import PostprocessedRasterSource, LocatedImage, srtm
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import csv
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
import numba as nb

def image_spoof(self, tile): # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy
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
        #source_pert.strike1 = strike
        #source_pert.dip1 = dip
        #source_pert.rake1 = rake

        mtm = pmt.MomentTensor.from_values((strike, dip, rake))
        mtm.moment = mts.moment
        source_pert.mnn = mtm.mnn
        source_pert.mee = mtm.mee
        source_pert.mdd = mtm.mdd
        source_pert.mne = mtm.mne
        source_pert.mnd = mtm.mnd
        source_pert.med = mtm.med

        sources.append(source_pert)
    return sources


def make_shakemap(engine, source, store_ids, folder, stations=None, save=True,
                  stations_corrections_file=None, pertub_mechanism=False,
                  pertub_degree=20, measured=None, n_pertub=0,
                  value_level=0.004, pertub_velocity_model=False,
                  scenario=False, picks=None,
                  folder_waveforms="/home/asteinbe/bgr_data/acquisition",
                  get_measured=True, vs30_topo=False, quantity="velocity",
                  extent=30, geometries=None, geometries_vs30=None):

    targets, norths, easts, stf_spec = get_scenario(engine,
                                                    source,
                                                    store_ids[0],
                                                    extent)
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
            for store_id_pert in store_ids:
                targets, norths, easts, stf_spec = get_scenario(engine,
                                                                source,
                                                                store_id_pert,
                                                                extent)
                response = engine.process(source, targets)
                values = post_process(response, norths, easts, stf_spec,
                                      savedir=folder,
                                      save=save, quantity=quantity)
                values_pertubed.append(values)
        else:
            response = engine.process(source, targets)
            values = post_process(response, norths, easts, stf_spec,
                                  savedir=folder,
                                  save=save, quantity=quantity)
            values_pertubed.append(values)

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
                      store_ids=store_ids, vs30_topo=vs30_topo,
                      geometries=geometries, geometries_vs30=geometries_vs30)


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
            stf_spec = BruneResponse(duration=0.05)
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
                    quantity='velocity',
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
                    quantity='velocity',
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
    #    if quantity == "velocity":
    #        trans = trace.DifferentiationResponse(1)
        if quantity == "acceleration":
            trans = trace.DifferentiationResponse(2)
        if quantity is not "velocity":
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


def prepare_list(sources, values_list, scale="mm"):
    mts = []
    values_per_list = []
    for i, source in enumerate(sources):
        mts.append(source.pyrocko_moment_tensor())
        if i == 0:
            best_mt = source.pyrocko_moment_tensor()
    for i, values_pertubed in enumerate(values_list):
        if i == 0:
            if scale == "mm":
                values_pertubed = values_pertubed*1000.
            values_per_list = [values_pertubed]

            values_cum = num.zeros(num.shape(values_pertubed))
            values_cum = values_cum + values_pertubed
        else:
            values_cum = values_cum + values_pertubed
            if scale == "mm":
                values_pertubed = values_pertubed*1000.
            values_per_list.append(values_pertubed)
    values_cum = values_cum/float(len(values_list))
    return mts, values_cum, values_per_list, best_mt


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
                  scale="mm", plot_values=True, vs30=True,
                  vs30_topo=False,
                  engine=None, type_factors=None, store_ids=None,
                  plot_snr=False, save=True, img_puffer=0,
                  data_folder="/media/asteinbe/aki/playground/data/events/groebers/waveforms",
                  geometries=None, geometries_vs30=None):

    mts, values_cum, values_prep, ref_mt = prepare_list(sources, values_list, scale=scale)
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
    for i, values in enumerate(values_prep):
        if values_stations_list is not None:
            values_stations = values_stations_list[0]
            if scale == "mm":
                values_stations = abs(values_stations)*1000.

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
        store = engine.get_store(store_ids[0])
        values_prep_new = []

        if type_factors is None:
            layer0 = store.config.earthmodel_1d.layer(0)
            base_velocity = layer0.mtop.vs
            base_velocity = 600.
            type_factors = 1
            amp_factor = (base_velocity/values_vs30_resam)**type_factors
            for values in values_prep:
                values = values*amp_factor
                values_prep_new.append(values)
        values_prep = values_prep_new
    if plot_background_map is True:
        axes = plt.axes(projection=projection)
        gl = axes.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = False

        axes.add_feature(cartopy.feature.OCEAN, zorder=-4)
        axes.add_feature(cartopy.feature.LAND, zorder=-4, edgecolor='black')
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
    #    import cartopy.io.shapereader as shpr
        # axes.add_feature(states_provinces, edgecolor='gray')
        # stt = cfeature.NaturalEarthFeature(category='cultural',
        #     name='admin_0_boundary_lines_land',
        #     scale='10m',facecolor='none')
        # stt_prv = cfeature.NaturalEarthFeature(category='cultural',
        #     name='admin_1_states_provinces_lines',
        #     scale='10m',facecolor='none')
        # fname = shpr.natural_earth(resolution='10m', category='cultural', name='populated_places')
        # reader = shpr.Reader(fname)
        # points = list(reader.geometries())
        # axes.scatter([point.x for point in points],
        #            [point.y for point in points],
        #            transform=projection,
        #             c='k')
        # axes.add_feature(stt, linewidth=0.2, edgecolor='black')
        # axes.add_feature(stt, linewidth=0.5, edgecolor='black')
        # axes.add_feature(stt_prv, linewidth=0.2, edgecolor='black')
    if source.base_key()[6] is not "ExplosionSource":
        plot_kwargs = {
            'size': 0.03,
            'edgecolor': 'black',
            }

        # beachball.plot_fuzzy_beachball_mpl_pixmap(
        #      mts, axes, ref_mt,
        #      position=(source.lon, source.lat),
        #      color_t='black',
        #      zorder=2,
        #      **plot_kwargs)
        beachball.plot_beachball_mpl(
            mts[0], axes,
            beachball_type='full',
              position=(source.lon, source.lat),
              zorder=2,

            linewidth=1.0)

    if plot_background_map is True:
        values[values == 0] = 'nan'
        alpha = 0.5
    else:
        alpha = 1.
    #_, vmax = num.nanmin(values), num.nanmax(values)

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
            _, vmax = num.nanmin(measured_values), num.nanmax(measured_values)

        except:
            measured_values = []
            if measured is not None:
                for data in measured:
                    if scale == "mm":
                        measured_values.append(data*1000.)
                _, vmax = num.nanmin(measured_values), num.nanmax(measured_values)

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


    if quantity == "acceleration":
        if scale == "mm":
            fig.colorbar(im, label='Velocity [mm/s]')
        else:
            fig.colorbar(im, label='Velocity [m/s]')

    img_extent = (num.min(lons_map)-img_puffer, num.max(lons_map)-img_puffer,
                  num.min(lats_map)+img_puffer, num.max(lats_map)+img_puffer)

    # axes.contour(lons_map, lats_map, values, cmap='brg',
    #              levels=[value_level],
    #              transform=ccrs.PlateCarree(), zorder=0)
    from pyrocko import orthodrome as od
    from pyrocko.dataset import topo

    dem_name = 'SRTMGL3'

    # extract gridded topography (possibly downloading first)
    img_extent_topo = (num.min(lons_map)-0.5, num.max(lons_map)+0.5,
                        num.min(lats_map)-0.5, num.max(lats_map)+0.5)
    img_extent_pyrocko = (num.min(lons_map)-0.05, num.max(lons_map)+0.05,
                        num.min(lats_map)-0.05, num.max(lats_map)+0.05)
    tile = topo.get(dem_name, img_extent_topo)

    # geographic to local cartesian coordinates
    lons = tile.x()
    lats = tile.y()
    lons2 = num.tile(lons, lats.size)
    lats2 = num.repeat(lats, lons.size)
    norths = lats2.reshape((lats.size, lons.size))
    easts = lons2.reshape((lats.size, lons.size))

    # x, y = np.gradient(tile.data)
    #
    # slope = num.pi/2. - num.arctan(num.sqrt(x*x + y*y))
    #
    # # -x here because of pixel orders in the SRTM tile
    # aspect = num.arctan2(-x, y)
    #
    # altitude = num.pi / 4.
    # azimuth = num.pi / 2.
    #
    # shaded_srtm = np.sin(altitude) * np.sin(slope)\
    #     + np.cos(altitude) * np.cos(slope)\
    #     * np.cos((azimuth - np.pi/2.) - aspect)
    #srtm = tile.data
#    axes.add_raster(shaded_srtm, cmap='Greys', transform=projection)
    if plot_values is True:

        stations_file = data_folder+"/stations.prepared.txt"
        folder_waveforms = data_folder+"/rest"
        event_file = data_folder+"/../event.txt"

        stations = model.load_stations(stations_file)
        stf_spec = BruneResponse(duration=0.1)
        events = model.load_events(event_file)

        pgas_waveforms, stations_region = get_measured_values.get_max_pga(events, folder_waveforms, stations, img_extent_pyrocko)
        st_lats, st_lons = [], []
        if scale == "mm":
            pgas_waveforms = num.asarray(pgas_waveforms)*1000.
        i = 0
        for st in stations_region:
            st_lats.append(st.lat)
            st_lons.append(st.lon)
            i = i+1
    vmax = num.max(pgas_waveforms)
    if quantity == "velocity":
        if scale == "mm":
            m = plt.cm.ScalarMappable(cmap=plt.get_cmap('YlOrBr'))
            m.set_array(values_list[0])
            m.set_clim(0., vmax)
            plt.colorbar(m, boundaries=num.linspace(0, round(vmax,2), 6),  label='PGV [mm/s]')
        else:
            fig.colorbar(im, label='Velocity [m/s]')
    plt.scatter(st_lons, st_lats,
                c=num.asarray(pgas_waveforms), s=36,
                cmap=plt.get_cmap('YlOrBr'),
                vmin=0., vmax=vmax, edgecolor="k", zorder=3,
                marker="^", transform=projection)
    for k, st in enumerate(stations_region):
        plt.text(st_lons[k], st_lats[k], str(st.station),
        transform=projection, zorder=4)
    # cbar = axes.pcolormesh(easts, norths, shaded_srtm,
    #                        cmap='gray', zorder=-2)
    longs, lats, geometries = geometries
    geometries = geometries*100
    longs, lats, geometries_vs30 = geometries_vs30
    geometries_vs30 = geometries_vs30*100
    #print(num.min(geometries), num.max(geometries), num.min(values), num.max(values), num.min(pgas_waveforms), num.max(pgas_waveforms))
    # axes.contourf(lats, longs, geometries,
    #               transform=projection, zorder=3,
    #               vmin=0., vmax=vmax, alpha=0.5,
    #               cmap=plt.get_cmap('YlOrBr'))
    # axes.contour(lats, longs, geometries,
    #               transform=projection, zorder=3,
    #               vmin=0., vmax=vmax, alpha=0.5,
    #               cmap=plt.get_cmap('YlOrBr'))
    axes.contour(lats, longs, geometries_vs30,
                  transform=projection, zorder=3,
                  vmin=0., vmax=vmax, alpha=0.5, levels=num.linspace(0, round(vmax,2), 6),
                  colors="k")
    print(num.shape(values_prep))
    print(num.max(values_prep[0]), num.min(values_prep[0]))
    axes.contourf(lons_map, lats_map, values_prep[0],
                  transform=projection, zorder=-1,
                  vmin=0., vmax=vmax,
                  cmap=plt.get_cmap('YlOrBr'), alpha=0.5)
    #for geom in geometries:
    #    axes.scatter(geom[0][:], geom[1][:])

    #axes.add_geometries([geometries], projection, lw=0.5)
    # axes.set_yticks([51, 52], crs=projection)
    # axes.set_yticklabels([51, 52], color='red', weight='bold')
    # axes.set_xticks([11, 12], crs=projection)
    # axes.set_xticklabels([11, 12])

    fig.savefig(folder+filename)
    if show is True:
        plt.show()
    else:
        plt.close()

    plt.figure()
    dists = []
    for st in stations_region:
        dists.append(orthodrome.distance_accurate50m(st.lat, st.lon, source.lat, source.lon))

    plt.scatter(dists, pgas_waveforms, c="b",  marker="^")
    dists = []
    lons2 = num.tile(lons_map, len(lats_map))
    lats2 = num.repeat(lats_map, len(lons_map))
    for lon, lat in zip(lons2.flatten(), lats2.flatten()):
        dists.append(orthodrome.distance_accurate50m(lat, lon, source.lat, source.lon))

    values = num.asarray(values_prep[0].flatten())
    values = values.tolist()
    values_med = []
    for valuesp in values_prep[1:]:
        ds = num.asarray(valuesp.flatten())
        ps = ds.tolist()
        values_med.append(ps)

    for k, dist in enumerate(dists):
        vads = []
        for vals in values_med:
            vads.append(vals[k])
        #vals = vals[k]
        #print(num.shape(vals), values[k])
        minp = values[k]-num.min(vads)
        maxp = values[k]-num.max(vads)
    #    plt.plot(dist, num.max(vads)-num.min(vads), c="r", alpha=0.3)
        plt.plot([dist, dist], [values[k]-minp, values[k]+maxp], c="r", alpha=0.3)
        #plt.plot(dist, values[k]+maxp, c="r", alpha=0.3)

#        plt.scatter(dists, values.flatten(), c="r", alpha=0.3)


    #plt.fill_between(dists, (values-ci), (values+ci), color='blue', alpha=0.1)
    plt.scatter(dists,values, c="r")
    #plt.scatter(dists,values, c="r")
    dists = []
    geom = []


    # for alon, alat in zip(lats.flatten(), longs.flatten()):
    #     dists.append(orthodrome.distance_accurate50m(alat, alon, source.lat, source.lon))
    # plt.plot(dists, geometries.flatten(), c="k")
    geom = geometries.flatten()
    gd = []
    for alon, alat, g in zip(lats.flatten()[::2], longs.flatten()[::2], geom[::2]):
        dist = int(orthodrome.distance_accurate50m(alat, alon, source.lat, source.lon))
        if dist in dists:
             idx = dists.index(dist)
             gs = gd[idx]
             gs.append(g)
             gd[idx] = gs

        else:
             dists.append(dist)
             gd.append([g])
        #print(gd)
    means = []
    maxs = []
    mins = []
    for part in gd:
        means.append(num.mean(part))
        maxs.append(num.min(part))
        mins.append(num.max(part))

    # geom = num.asarray(geometries.flatten())
    #dist_set = set(dists)
    # print(len(dist_set))
    # for dist in dist_set:
    #     #indexes = [i for i, x in enumerate(dists) if x == dist]
    # #    gen = index_find_all(dists, dist)
    #     #gs = []
    #     #for k in indexes:
    #     #    gs.append(geom[k])
    #     #gs = num.take(geom, [*index_find_all(dists, dist)])
    #     gs = ndtaker(geom, num.asarray([*index_find_all(dists, dist)]))
    #     gd.append(gs)
    #     print(len(gd))
    # print("collected")
    #p50 = num.median(gd, axis=0)
#    plt.scatter(set(dists), p50, c="k")
    # import seaborn as sns
    # import pandas as pd
    # df = pd.DataFrame(num.transpose([dists,geom]), columns = ['d', 'g'])
    # #sns.lineplot(data=df, x="d", y="g", hue="g")
    # print("df_made")
    # sns.lmplot(
    #     data=df,
    #     x="d", y="g")
#    mean = num.mean(gd, axis=0)
    # std = num.mean(gd, axis=0)
    #
    # ci = 0.1 * std / mean
    # Plot the sinus function
    #plt.scatter(dists, gd)
    # coefficients = num.polyfit(dists, means, 4)
    # poly = num.poly1d(coefficients)
    # new_y = poly(dists)
    # #print(num.shape(mean))
    # plt.plot(dists, new_y)
    # Plot the confidence interval
    means_n = num.asarray(means)
    maxs_n = num.asarray(maxs)
    mins_n = num.asarray(mins)
    plt.plot(dists, means_n)

    dists = []
    geom = geometries_vs30.flatten()
    gd = []
    for alon, alat, g in zip(lats.flatten()[::2], longs.flatten()[::2], geom[::2]):
        dist = int(orthodrome.distance_accurate50m(alat, alon, source.lat, source.lon))
        if dist in dists:
             idx = dists.index(dist)
             gs = gd[idx]
             gs.append(g)
             gd[idx] = gs

        else:
             dists.append(dist)
             gd.append([g])
        #print(gd)
    means = []
    maxs = []
    mins = []
    for part in gd:
        means.append(num.mean(part))
        maxs.append(num.min(part))
        mins.append(num.max(part))

    means = num.asarray(means)
    maxs = num.asarray(maxs)
    mins = num.asarray(mins)
    plt.fill_between(dists, (means_n-(means-mins)), (means_n+(maxs-means)), color='blue', alpha=0.1)
    #plt.plot(dists, means)

#     half = int((len(gd)-1)/2)
#     colormap = cm.Blues # change this for the colormap of choice
# #    plt.scatter(dists, geometries.flatten(), c="k")
#     dist_set = set(dists)
#     for i in range(half):
#         axes.fill_between(dist_set[i], gd[i]],color=colormap(i/half))

    plt.ylabel('Velocity [mm/s]')
    plt.xlabel('distance [m]')

    plt.show()


@nb.jit(nopython=True, cache=True, nogil=True)
def ndtaker(x, idx):
    x.take(idx)
    return x

def index_finder(lst, item):
    """A generator function, if you might not need all the indices"""
    start = 0
    while True:
        try:
            start = lst.index(item, start)
            yield start
            start += 1
        except ValueError:
            break

import array
def index_find_all(lst, item, results=None):
    """ If you want all the indices.
    Pass results=[] if you explicitly need a list,
    or anything that can .append(..)
    """
    if results is None:
        length = len(lst)
        results = (array.array('B') if length <= 2**8 else
                   array.array('H') if length <= 2**16 else
                   array.array('L') if length <= 2**32 else
                   array.array('Q'))
    start = 0
    while True:
        try:
            start = lst.index(item, start)
            results.append(start)
            start += 1
        except ValueError:
            return results




def fwd_shakemap_post(projdir, wanted_start=0,
                      store_ids=["insheim_100hz"], gf_store_superdirs=None,
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
                      quantity="velocity",
                      source_type="MT",
                      vs30_topo=False,
                      stations_corrections_file=None,
                      extent=30,
                      geometries_vs30=None,
                      geometries=None):

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
            engine, source, store_ids[0],
            distance=100*km, dazi=5., component='R',
            plot_mt='full', show_phases=True,
            phases={
                'First': 'first{stored:begin}-10%',
                'Last': 'last{stored:end}+20'
            },
            quantity='velocity', envelope=True)

    else:
        make_shakemap(engine, source, store_ids,
                      projdir, stations=stations,
                      n_pertub=n_pertub,
                      pertub_mechanism=pertub_mechanism,
                      pertub_degree=pertub_degree,
                      pertub_velocity_model=pertub_velocity_model,
                      value_level=value_level,
                      measured=measured,
                      vs30_topo=vs30_topo,
                      quantity=quantity,
                      geometries=geometries,
                      geometries_vs30=geometries_vs30,
                      stations_corrections_file=stations_corrections_file,
                      extent=extent)
