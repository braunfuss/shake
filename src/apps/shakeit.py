#!/usr/bin/env python

import sys
import logging
from optparse import OptionParser

from pyrocko import util
import multiprocessing
import shake
import rasterio
import numpy as num

logger = logging.getLogger('main')


def d2u(d):
    if isinstance(d, dict):
        return dict((k.replace('-', '_'), v) for (k, v) in d.items())
    else:
        return d.replace('-', '_')


def str_to_time(s):
    try:
        return util.str_to_time(s)
    except util.TimeStrError as e:
        raise shake.shakeError(str(e))


subcommand_descriptions = {
    'init': 'create initial configuration file',
    'search': 'detect seismic events',
    'map-geometry': 'make station map',
    'snuffle': 'snuffle',
    'oq_shakemap': 'oq_shakemap',
    'fwd_shakemap': 'fwd_shakemap',
    'combined_shakemap': 'combined_shakemap'

}

subcommand_usages = {
    'init': 'init',
    'oq_shakemap': 'oq_shakemap',
    'fwd_shakemap': 'fwd_shakemap',
    'combined_shakemap': 'combined_shakemap',
    'search': 'search <configfile> [options]',
    'map-geometry': 'map-geometry <configfile> [options] <output.(png|pdf)',
    'snuffle': 'snuffle <configfile>',
}

subcommands = subcommand_descriptions.keys()

program_name = 'shake'

usage_tdata = d2u(subcommand_descriptions)
usage_tdata['program_name'] = program_name

usage = program_name + ''' <subcommand> [options] [--] <arguments> ...

Subcommands:

    init          %(init)s
    search        %(search)s
    map-geometry  %(map_geometry)s
    snuffle       %(snuffle)s
    oq_shakemap        %(oq_shakemap)s

To get further help and a list of available options for any subcommand run:

    %(program_name)s <subcommand> --help

''' % usage_tdata


def add_common_options(parser):
    parser.add_option(
        '--loglevel',
        action='store',
        dest='loglevel',
        type='choice',
        choices=('critical', 'error', 'warning', 'info', 'debug'),
        default='info',
        help='set logger level to '
             '"critical", "error", "warning", "info", or "debug". '
             'Default is "%default".')


def process_common_options(options):
    util.setup_logging(program_name, options.loglevel)


def cl_parse(command, args, setup=None):
    usage = subcommand_usages[command]
    descr = subcommand_descriptions[command]

    if isinstance(usage, str):
        usage = [usage]

    susage = '%s %s' % (program_name, usage[0])
    for s in usage[1:]:
        susage += '\n%s%s %s' % (' '*7, program_name, s)

    parser = OptionParser(
        usage=susage,
        description=descr[0].upper() + descr[1:] + '.')

    if setup:
        setup(parser)

    add_common_options(parser)
    (options, args) = parser.parse_args(args)
    process_common_options(options)
    return parser, options, args


def die(message, err=''):
    if err:
        sys.exit('%s: error: %s \n %s' % (program_name, message, err))
    else:
        sys.exit('%s: error: %s' % (program_name, message))


def help_and_die(parser, message):
    parser.print_help(sys.stderr)
    sys.stderr.write('\n')
    die(message)


def escape(s):
    return s.replace("'", "\\'")


def command_init(args):
    def setup(parser):
        parser.add_option(
            '--stations', dest='stations_path',
            metavar='PATH',
            help='stations file')

        parser.add_option(
            '--data', dest='data_paths',
            default=[],
            action='append',
            metavar='PATH',
            help='data directory (this option may be repeated)')

    parser, options, args = cl_parse('init', args, setup=setup)

    if options.data_paths:
        s_data_paths = '\n'.join(
            "- '%s'" % escape(x) for x in options.data_paths)
    else:
        s_data_paths = "- 'DATA'"

    if options.stations_path:
        stations_path = options.stations_path
    else:
        stations_path = 'STATIONS_PATH'

    print('''%%YAML 1.1
--- !shake.Config

## Configuration file for shake, your friendly earthquake detector
##
## Receiver coordinates can be read from a stations file in Pyrocko format:
stations_path: '%(stations_path)s'

## Receivers can also be listed in the config file, lat/lon and carthesian
## (x/y/z) = (North/East/Down) coordinates are supported and may be combined
## (interpreted as reference + offset). Omitted values are treated as zero.
# receivers:
# - !shake.Receiver
#   codes: ['', 'ACC13', '']
#   lat: 10.
#   lon: 12.
#   x: 2397.56
#   y: 7331.94
#   z: -404.1

## List of data directories. shake will recurse into subdirectories to find
## all contained waveform files.
data_paths:
%(s_data_paths)s

## name template for shake's output directory. The placeholder
## "${config_name}" will be replaced with the basename of the config file.
run_path: '${config_name}.shake'

## Processing time interval (default: use time interval of available data)
# tmin: '2012-02-06 04:20:00'
# tmax: '2012-02-06 04:30:00'

## Whether to create a figure for every detection and save it in the output
## directory
save_figures: true

## Mapping of phase ID to phase definition in cake syntax (used e.g. in the
## CakePhaseShifter config sections)
tabulated_phases:
- !pf.TPDef
  id: 'p'
  definition: 'P,p'
- !pf.TPDef
  id: 's'
  definition: 'S,s'

## Mapping of earthmodel ID  to the actual earth model in nd format (used in
## the CakePhaseShifter config sections)
earthmodels:
- !shake.CakeEarthmodel
  id: 'swiss'
  earthmodel_1d: |2
    0.0 5.53 3.10  2.75
    2.0 5.53 3.10  2.75
    2.0 5.80 3.25  2.75
    5.0 5.80 3.25  2.75
    5.0 5.83 3.27  2.75
    8.0 5.83 3.27  2.75
    8.0 5.95 3.34  2.8
    13.0 5.95 3.34  2.8
    13.0 5.96 3.34  2.8
    22.0 5.96 3.34  2.8
    22.0 6.53 3.66  2.8
    30.0 6.53 3.66  2.8
    30.0 7.18 4.03 3.3
    40.0 7.18 4.03 3.3
    40.0 7.53 4.23 3.3
    50.0 7.53 4.23 3.3
    50.0 7.83 4.39 3.3
    60.0 7.83 4.39 3.3
    60.0 8.15 4.57 3.3
    120.0 8.15 4.57 3.3
''' % dict(
        stations_path=stations_path,
        s_data_paths=s_data_paths))


def command_oq_shakemap(args):
    from shake import oq_shakemap
    def setup(parser):
        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing files')
        parser.add_option(
            "--qml",
            dest="qml",
            type=str,
            default=True,
            help="qml")
        parser.add_option(
            "--events",
            dest="events",
            type=str,
            help="Pyrocko event file")

    export_folder = args[0]
    util.ensuredir(export_folder)

    parser, options, args = cl_parse("oq_shakemap", args, setup)
    if options.qml is not True:
        oq_shakemap.workflows.shakemaps_from_quakeml(options.qml,
                                                     export_folder=export_folder)

    if options.events:
        from pyrocko.io import quakeml as q
        from pyrocko import model
        events = model.load_events(options.events)
        event = events[0]
        table = [(event.name, event.time, event.lat, event.lon, event.depth,
                  event.magnitude)]
        event_qml = q.QuakeML(
            event_parameters=q.EventParameters(
                public_id='quakeml:test/eventParameters/test',
                event_list=[
                    q.Event(
                        preferred_origin_id='quakeml:test/origin/%s' % name,
                        preferred_magnitude_id='quakeml:test/magnitude/%s' % name,
                        public_id='quakeml:test/event/%s' % name,
                        origin_list=[
                            q.Origin(
                                public_id='quakeml:test/origin/%s' % name,
                                time=q.TimeQuantity(value=time),
                                longitude=q.RealQuantity(value=lon),
                                latitude=q.RealQuantity(value=lat),
                                depth=q.RealQuantity(value=depth),
                            ),
                        ],
                        magnitude_list=[
                            q.Magnitude(
                                public_id='quakeml:test/magnitude/%s' % name,
                                origin_id='quakeml:test/origin/%s' % name,
                                mag=q.RealQuantity(value=magnitude),
                            )
                        ],

                    )
                    for (name, time, lat, lon, depth, magnitude) in table
                ]
            )
        )

        event_qml.dump_xml(filename=export_folder+"/%s.qml" % event.time)
        oq_shakemap.workflows.shakemaps_from_quakeml(export_folder+"/%s.qml" % event.time,
                                                     export_folder=export_folder)



def get_coords_from_geotiff(fname, array):
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
    cols, rows = num.meshgrid(num.arange(array.shape[1]),
                              num.arange(array.shape[0]))
    T1 = T0 * Affine.translation(0.5, 0.5)
    rc2en = lambda r, c: (c, r) * T1
    eastings, northings = num.vectorize(rc2en,
                                        otypes=[num.float, num.float])(rows, cols)
    return eastings, northings


def to_latlon(fname):
    import rasterio
    import numpy as np
    from affine import Affine
    from pyproj import Proj, transform

    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = num.meshgrid(num.arange(A.shape[2]), num.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = num.vectorize(rc2en, otypes=[num.float, num.float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong', datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    return longs, lats


def command_combined_shakemap(args):
    def setup(parser):
        parser.add_option(
            "--wanted_start",
            dest="wanted_start",
            type=int,
            default=0,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--wanted_end",
            dest="wanted_end",
            type=int,
            default=1,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--stations_file",
            dest="stations_file",
            type=str,
            default="stations.raw.txt",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--store_id",
            dest="store_id",
            type=str,
            default="insheim_100hz",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--store_ids",
            dest="store_ids",
            help="Comma-separated list of directories containing GF stores",
        )
        parser.add_option(
            "--measured",
            dest="measured",
            type=str,
            default=None,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--scenario",
            dest="scenario",
            type=str,
            default=False,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--pertub_velocity_model",
            dest="pertub_velocity_model",
            type=str,
            default=False,
            help="pertub_velocity_model.",
        )
        parser.add_option(
            "--pertub_mechanism",
            dest="pertub_mechanism",
            type=str,
            default=False,
            help="pertub_mechanism",
        )
        parser.add_option(
            "--gf-store-superdirs",
            dest="gf_store_superdirs",
            help="Comma-separated list of directories containing GF stores",
        )
        parser.add_option(
            "--n_pertub",
            dest="n_pertub",
            type=int,
            default=0,
            help="number of pertubations to create (default: %default)",
        )
        parser.add_option(
            "--pertub_degree",
            dest="pertub_degree",
            type=float,
            default=20,
            help="number of pertubations to create (default: %default)",
        )
        parser.add_option(
            "--pgv_outline",
            dest="value_level",
            type=float,
            default=0.005,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--extent",
            dest="extent",
            type=float,
            default=30,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--strike",
            dest="strike",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--dip",
            dest="dip",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--rake",
            dest="rake",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--moment",
            dest="moment",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--depth",
            dest="depth",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--source_type",
            dest="source_type",
            type=str,
            default="MT",
            help="Source Type (default: %default)",)
        parser.add_option(
            "--stations_corrections_file",
            dest="stations_corrections_file",
            type=str,
            default=None,
            help="stations_corrections_file",)
        parser.add_option(
            "--vs30_topo",
            dest="vs30_topo",
            type=str,
            default=False,
            help="Vs30 from topography",)
        parser.add_option(
            "--plot_directivity",
            dest="plot_directivity",
            type=str,
            default=False,
            help="plot directivity instead of shakemap",)
        parser.add_option(
            "--events",
            dest="events",
            type=str,
            help="Pyrocko event file")
        parser.add_option(
            "--qml",
            dest="qml",
            type=str,
            default=True,
            help="qml")

    from shake import oq_shakemap
    export_folder = args[0]
    util.ensuredir(export_folder)

    parser, options, args = cl_parse("oq_shakemap", args, setup)

    if options.qml is not True:
        oq_shakemap.workflows.shakemaps_from_quakeml(options.qml,
                                                     export_folder=export_folder)

    if options.events:
        from pyrocko.io import quakeml as q
        from pyrocko import model
        events = model.load_events(options.events)
        event = events[0]

        table = [(event.name, event.time, event.lat, event.lon, event.depth,
                  event.magnitude)]

        event_qml = q.QuakeML(
            event_parameters=q.EventParameters(
                public_id='quakeml:test/eventParameters/test',
                event_list=[
                    q.Event(
                        preferred_origin_id='quakeml:test/origin/%s' % name,
                        preferred_magnitude_id='quakeml:test/magnitude/%s' % name,
                        public_id='quakeml:test/event/%s' % name,
                        origin_list=[
                            q.Origin(
                                public_id='quakeml:test/origin/%s' % name,
                                time=q.TimeQuantity(value=time),
                                longitude=q.RealQuantity(value=lon),
                                latitude=q.RealQuantity(value=lat),
                                depth=q.RealQuantity(value=depth),
                            ),
                        ],
                        magnitude_list=[
                            q.Magnitude(
                                public_id='quakeml:test/magnitude/%s' % name,
                                origin_id='quakeml:test/origin/%s' % name,
                                mag=q.RealQuantity(value=magnitude),
                            )
                        ],

                    )
                    for (name, time, lat, lon, depth, magnitude) in table
                ]
            )
        )

        event_qml.dump_xml(filename=export_folder+"/%s.qml" % event.time)
        results = oq_shakemap.workflows.shakemaps_from_quakeml(export_folder+"/%s.qml" % event.time,
                                                     export_folder=export_folder, vs30_topo=False)

    # p = results.contours["mean"]["PGA"].values.tolist()
    # geometries = []
    # for ps in p:
    #     line = ps[1]
    #     geometries.append(line.coords.xy)

    fname = "groebers/groebers_PGA_mean.tif"
    comb = rasterio.open(fname)
    longs, lats = to_latlon(fname)
    geometries = comb.read(1)
    geometries = (longs, lats, geometries)

    results = oq_shakemap.workflows.shakemaps_from_quakeml(export_folder+"/%s.qml" % event.time,
                                                 export_folder=export_folder, vs30_topo=True)

    fname = "groebers/groebers_PGA_mean.tif"
    comb = rasterio.open(fname)
    longs, lats = to_latlon(fname)
    geometries_vs30 = comb.read(1)
    geometries_vs30 = (longs, lats, geometries_vs30)

    gf_store_superdirs = None
    if options.gf_store_superdirs:
        gf_store_superdirs = options.gf_store_superdirs.split(",")
    else:
        gf_store_superdirs = None
    if options.measured is not None:
        options.measured = True
    if options.scenario is not False:
        options.scenario = True
    if options.plot_directivity is not False:
        options.plot_directivity = True
    if options.vs30_topo is not False:
        options.vs30_topo = True
    project_dir = args[0]
    from matplotlib import pyplot as plt
    from shake import syn_shake
    if options.store_ids:
        store_ids = options.store_ids.split(',')
    print(store_ids)
    scenario = syn_shake.fwd_shakemap_post(
                project_dir,
                wanted_start=options.wanted_start,
                store_ids=store_ids,
                gf_store_superdirs=options.gf_store_superdirs,
                pertub_degree=options.pertub_degree,
                n_pertub=options.n_pertub,
                value_level=options.value_level,
                pertub_velocity_model=options.pertub_velocity_model,
                pertub_mechanism=options.pertub_mechanism,
                measured=options.measured,
                strike=options.strike,
                dip=options.dip,
                rake=options.rake,
                moment=options.moment,
                depth=options.depth,
                source_type=options.source_type,
                vs30_topo=options.vs30_topo,
                plot_directivity=options.plot_directivity,
                stations_corrections_file=options.stations_corrections_file,
                geometries=geometries,
                geometries_vs30=geometries_vs30,
                extent=options.extent)


def command_fwd_shakemap(args):
    def setup(parser):

        parser.add_option(
            "--wanted_start",
            dest="wanted_start",
            type=int,
            default=0,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--wanted_end",
            dest="wanted_end",
            type=int,
            default=1,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--stations_file",
            dest="stations_file",
            type=str,
            default="stations.raw.txt",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--store_id",
            dest="store_id",
            type=str,
            default="insheim_100hz",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--measured",
            dest="measured",
            type=str,
            default=None,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--scenario",
            dest="scenario",
            type=str,
            default=False,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--pertub_velocity_model",
            dest="pertub_velocity_model",
            type=str,
            default=False,
            help="pertub_velocity_model.",
        )
        parser.add_option(
            "--gf-store-superdirs",
            dest="gf_store_superdirs",
            help="Comma-separated list of directories containing GF stores",
        )
        parser.add_option(
            "--n_pertub",
            dest="n_pertub",
            type=int,
            default=0,
            help="number of pertubations to create (default: %default)",
        )
        parser.add_option(
            "--pertub_degree",
            dest="pertub_degree",
            type=float,
            default=20,
            help="number of pertubations to create (default: %default)",
        )
        parser.add_option(
            "--pgv_outline",
            dest="value_level",
            type=float,
            default=0.005,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--extent",
            dest="extent",
            type=float,
            default=30,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--strike",
            dest="strike",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--dip",
            dest="dip",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--rake",
            dest="rake",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--moment",
            dest="moment",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--depth",
            dest="depth",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",)
        parser.add_option(
            "--source_type",
            dest="source_type",
            type=str,
            default="MT",
            help="Source Type (default: %default)",)
        parser.add_option(
            "--stations_corrections_file",
            dest="stations_corrections_file",
            type=str,
            default=None,
            help="stations_corrections_file",)
        parser.add_option(
            "--vs30_topo",
            dest="vs30_topo",
            type=str,
            default=False,
            help="Vs30 from topography",)
        parser.add_option(
            "--plot_directivity",
            dest="plot_directivity",
            type=str,
            default=False,
            help="plot directivity instead of shakemap",)
    parser, options, args = cl_parse("fwd_shakemap", args, setup)
    gf_store_superdirs = None
    if options.gf_store_superdirs:
        gf_store_superdirs = options.gf_store_superdirs.split(",")
    else:
        gf_store_superdirs = None
    if options.measured is not None:
        options.measured = True
    if options.scenario is not False:
        options.scenario = True
    if options.plot_directivity is not False:
        options.plot_directivity = True
    if options.vs30_topo is not False:
        options.vs30_topo = True
    project_dir = args[0]

    from shake import syn_shake
    scenario = syn_shake.fwd_shakemap_post(
                project_dir,
                wanted_start=options.wanted_start,
                store_id=options.store_id,
                gf_store_superdirs=options.gf_store_superdirs,
                pertub_degree=options.pertub_degree,
                n_pertub=options.n_pertub,
                value_level=options.value_level,
                pertub_velocity_model=options.pertub_velocity_model,
                measured=options.measured,
                strike=options.strike,
                dip=options.dip,
                rake=options.rake,
                moment=options.moment,
                depth=options.depth,
                source_type=options.source_type,
                vs30_topo=options.vs30_topo,
                plot_directivity=options.plot_directivity,
                stations_corrections_file=options.stations_corrections_file,
                extent=options.extent)


def command_search(args):
    def setup(parser):
        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing files')

        parser.add_option(
            '--show-detections', dest='show_detections', action='store_true',
            help='show plot for every detection found')

        parser.add_option(
            '--show-movie', dest='show_movie', action='store_true',
            help='show movie when showing detections')

        parser.add_option(
            '--show-window-traces', dest='show_window_traces',
            action='store_true',
            help='show preprocessed traces for every processing time window')

        parser.add_option(
            '--stop-after-first', dest='stop_after_first', action='store_true',
            help='show plot for every detection found')

        parser.add_option(
            '--tmin', dest='tmin', metavar="'YYYY-MM-DD HH:MM:SS.XXX'",
            help='beginning of processing time window '
                 '(overrides config file settings)')

        parser.add_option(
            '--tmax', dest='tmax', metavar="'YYYY-MM-DD HH:MM:SS.XXX'",
            help='end of processing time window '
                 '(overrides config file settings)')

        parser.add_option(
            '--nworkers', dest='nworkers', metavar="N",
            help='use N cpus in parallel')

    parser, options, args = cl_parse('search', args, setup=setup)
    if len(args) != 1:
        help_and_die(parser, 'missing argument')

    config_path = args[0]
    config = shake.read_config(config_path)
    try:
        tmin = tmax = None

        if options.tmin:
            tmin = str_to_time(options.tmin)

        if options.tmax:
            tmax = str_to_time(options.tmax)

        if options.nworkers:
            nparallel = int(options.nworkers)
        else:
            nparallel = multiprocessing.cpu_count()

        shake.search(
            config,
            override_tmin=tmin,
            override_tmax=tmax,
            force=options.force,
            show_detections=options.show_detections,
            show_movie=options.show_movie,
            show_window_traces=options.show_window_traces,
            stop_after_first=options.stop_after_first,
            nparallel=nparallel,
            bark=options.bark)

    except shake.shakeError as e:
        die(str(e))


def command_map_geometry(args):
    parser, options, args = cl_parse('map-geometry', args)
    if len(args) != 2:
        help_and_die(parser, 'missing arguments')

    config_path = args[0]
    output_path = args[1]
    config = shake.read_config(config_path)
    shake.map_geometry(config, output_path)


def command_snuffle(args):
    parser, options, args = cl_parse('snuffle', args)
    if len(args) != 1:
        help_and_die(parser, 'missing arguments')

    config_path = args[0]
    config = shake.read_config(config_path)

    shake.snuffle(config)


if __name__ == '__main__':
    main()


def main():
    usage_sub = 'shake %s [options]'
    if len(sys.argv) < 2:
        sys.exit('Usage: %s' % usage)

    args = list(sys.argv)
    args.pop(0)
    command = args.pop(0)

    if command in subcommands:
        globals()['command_' + d2u(command)](args)

    elif command in ('--help', '-h', 'help'):
        if command == 'help' and args:
            acommand = args[0]
            if acommand in subcommands:
                globals()['command_' + acommand](['--help'])

        sys.exit('Usage: %s' % usage)

    else:
        die('no such subcommand: %s' % command)
