"""
Defines complete end-to-end shakemap calculation workflows
"""
import os
import numpy as np
from scipy import stats
from typing import Optional, Dict, List, Tuple, Union
from openquake.hazardlib.gsim.base import RuptureContext, SitesContext, DistancesContext
from shake.oq_shakemap.earthquake import Earthquake
from shake.oq_shakemap.site_model import SiteModel
from shake.oq_shakemap.regionalization import DEFAULT_GLOBAL_REGIONALIZATION, RegionalizationSet
from shake.oq_shakemap.shakemap import Shakemap
from shake.oq_shakemap.gmice import AtkinsonKaka2007MRDependent, WordenEtAl2012MRDependent


DEFAULT_CONFIGURATION = {
    # Bounding Box
    "spcx": 1.0 / 120.0,  # 30 arc-seconds
    "spcy": 1.0 / 120.0,  # 30 arc-seconds
    "max_distance_bbox": None,
    # Site Properties
    "default_vs30": 600.0,
    "vs30measured": True,
    "z1pt0": None,
    "z2pt5": None,
    "xvf": 0.0,
    "backarc": False,
    "region": 0,
    "geology": "UNCLASSIFIED",
    # Shakemap
    "num_rupture_samples": 100,
    "rdim": 0.0,
    "synth_dist_weights": [0.25, 0.25, 0.25, 0.25],
    "synthetic_rupture_max_site_distance": 200.0,
    "synthetic_rupture_site_spacing": 0.05,
}


class ShakemapWorkflowResult(object):
    """
    Object to hold output information from shakemap calculations, including
    configuration metadata
    """

    __slots__ = [
        "shakemaps",
        "contours",
        "num_sites",
        "bbox_properties",
        "ground_motion_models",
        "tectonic_region",
        "earthquake",
        "statistics",
    ]

    def __getitem__(self, key):
        return getattr(self, key)


# Initially we consider only two different GMICEs: AtkinsonKaka2007 for Stable Continental
# Regions and Worden et al. (2012) everywhere else. Further GMICEs may be added in future.
GMICE_REGIONALIZATION = {
    "Global Stable": AtkinsonKaka2007MRDependent,
}


def get_mmi_shakemaps(
    region: str,
    mean_shakemaps: np.ndarray,
    stddev_shakemaps: np.ndarray,
    sctx: SitesContext,
    rctx: RuptureContext,
    dctx: DistancesContext,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves shakemaps in terms of macroseismic intensity by applying ground motion intensity
    conversion equations (GMICEs) to the mean and standard deviation shakemaps.

    GMICEs are usually only defined for a small number of ground motion parameters (e.g.
    PGA, PGV, SA(0.3), SA(1.0), SA(3.0)) and as the resulting variance can be smaller when
    some parameters such as PGV or SA(1.0) are used then the conversions are applied in a
    defined order of preference, meaning that if the preferred IMT is found then that will
    be used, otherwise it will try the next in the list, and so on. If none of the applicable
    IMTs are found then no conversion will be undertaken and an error raised instead.

    Args:
        region: Tectonic region of event (as defined by GMPE regionalization
        mean_shakemaps: The set of shakemaps for mean logarithm of ground motion
        stddev_shakemaps: The set of shakemaps for the standard deviation of the logarithm of
                          ground motion
        sctx: The shakemap SitesContext object
        rctx: The shakemap RuptureContext object
        dctx: The shakemap DistancesContext object

    Returns:
        mmi: The shakemap in terms of mean macroseismic intensity
        stddev_mmi: The shakemap of standard deviation of macroseismic intensity
    """
    # Select the appropriate GMICE from the regionalization
    if region in GMICE_REGIONALIZATION:
        gmice = GMICE_REGIONALIZATION[region](sctx, rctx, dctx)
    else:
        # If not defined in the regionalization then use Worden et al. (2012) as the default
        gmice = WordenEtAl2012MRDependent(sctx, rctx, dctx)

    # Identify the preferred intensity measure type for the GMICE
    preferred_imt = None
    for imt in gmice.ORDER_OF_PREFERENCE:
        if imt in mean_shakemaps.dtype.names:
            preferred_imt = imt
            break
    if not preferred_imt:
        raise ValueError(
            "MMI conversion with %s failed as none of the IMTs needed for "
            "conversion were found in the shakemaps" % gmice.__class__.__name__
        )
    mmi = np.zeros(mean_shakemaps.shape[0], dtype=np.dtype([("MMI", np.float64)]))
    stddev_mmi = np.zeros(mean_shakemaps.shape[0], dtype=np.dtype([("MMI", np.float64)]))
    # Get the MMI and standard deviation
    mmi["MMI"], stddev_mmi["MMI"] = gmice.get_mean_intensity_and_stddev(
        preferred_imt, mean_shakemaps[preferred_imt], stddev_shakemaps[preferred_imt], clip=True
    )
    return mmi, stddev_mmi


def shakemaps_from_quakeml(
    event_id: str,
    imts: List = ["PGA", "SA(0.3)", "SA(1.0)", "MMI"],
    config: Optional[Dict] = None,
    export_folder: Optional[str] = None,
    cache_file: Optional[str] = None,
    contour_levels_mean: Union[int, List] = 10,
    contour_levels_stddev: Union[int, List] = 10,
    regionalization: RegionalizationSet = DEFAULT_GLOBAL_REGIONALIZATION,
) -> Tuple[Dict, Dict]:
    """
    Implements the complete shakemap workflow for use with the GEOFON FDSN Web Service

    Args:
        event_id: GEOFON event ID or path to QuakeML file for event
        imts: List of intensity measure types e.g., PGA, SA(0.1), SA(0.2), ... etc.
        config: Dictionary of configuration properties (will use the GEOFON default
                configuration if not supplied
        export_folder: Path to export the geotiff and contour data (if storing locally)
        cache_file: File to cache the shakemap data (optional)
        contour_levels_mean: Number of levels for contouring the mean shakemaps, or list of
                             pre-defined levels
        contour_levels_stddev: Number of levels for contouring the standard deviation
                               shakemaps, or list of pre-defined levels

    Returns:
        Tuple of dictionaries containing the mean and standard deviation shakemaps
        as instances of BytesIO objects, organised by intensity measure type
    """
    if not config:
        config = DEFAULT_CONFIGURATION
    if export_folder:
        if os.path.exists(export_folder):
            raise IOError("Designated export folder %s already exists!" % export_folder)
        os.mkdir(export_folder)

    # Create the event from the GEOFON event ID (or the path to the QuakeML file)
    results = ShakemapWorkflowResult()
    results.earthquake = Earthquake.from_quakeml(event_id)
    # Split the configuration into those parts relating to the bounding box,
    # the site model and the shakemap
    bbox_config = {}
    for key in ["spcx", "spcy", "max_distance_bbox"]:
        bbox_config[key] = config.get(key, DEFAULT_CONFIGURATION[key])
    site_config = {}
    for key in [
        "default_vs30",
        "vs30measured",
        "z1pt0",
        "z2pt5",
        "xvf",
        "backarc",
        "region",
        "geology",
    ]:
        site_config[key] = config.get(key, DEFAULT_CONFIGURATION[key])
    # Build the site model
    bbox = results.earthquake.get_maximum_distance_bbox(bbox_config["max_distance_bbox"])
    vs30 = site_config.pop("default_vs30")
    site_model = SiteModel.from_bbox(
        bbox, bbox_config["spcx"], bbox_config["spcy"], vs30, **site_config
    )
    results.num_sites = len(site_model)
    results.bbox_properties = site_model.bbox_properties
    results.bbox_properties["cross_antimeridian"] = site_model.cross_antimeridian
    # Get the ground motion models
    results.tectonic_region, results.ground_motion_models = regionalization(results.earthquake)
    shakemap_config = {}
    for key in [
        "num_rupture_samples",
        "rdim",
        "synth_dist_weights",
        "synthetic_rupture_max_site_distance",
        "synthetic_rupture_site_spacing",
    ]:
        shakemap_config[key] = config.get(key, DEFAULT_CONFIGURATION[key])
    # Run the shakemap
    shakemap = Shakemap(
        results.earthquake,
        site_model,
        results.ground_motion_models,
        results.tectonic_region,
        cache_file=cache_file,
        **shakemap_config
    )
    mean_shakemaps, stddev_shakemaps, _ = shakemap.get_shakemap(imts)
    # Export to file (if an export directory is given) or to a dictionary of byte arrays
    results.shakemaps = {"mean": {}, "stddevs": {}}
    results.contours = {"mean": {}, "stddevs": {}}
    results.statistics = {"mean": {}}
    for imt in imts:
        if imt == "MMI":
            # Get MMI and its standard deviation from the mean and stddev shakemaps
            mmi, stddev_mmi = get_mmi_shakemaps(
                results.tectonic_region,
                mean_shakemaps,
                stddev_shakemaps,
                shakemap.sctx,
                shakemap.rctx,
                shakemap.dctx,
            )
            # MMIs are not in logarithmic domain, so setting is_stddev to True
            results.shakemaps["mean"][imt] = shakemap.to_geotiff(mmi, imt, is_stddev=True)
            results.shakemaps["stddevs"][imt] = shakemap.to_geotiff(
                stddev_mmi, imt, is_stddev=True
            )
            results.contours["mean"][imt] = shakemap.get_contours(
                imt, mmi, np.arange(1, 11, 1), is_stddev=True
            )
            results.contours["stddevs"][imt] = shakemap.get_contours(
                imt, stddev_mmi, contour_levels_stddev, is_stddev=True
            )
            # Retrieve summary statistics - not in logarithmic domain
            results.statistics["mean"][imt] = {
                "maximum": np.max(mmi[imt]),
                "median": np.mean(mmi[imt]),
                "minimum": np.min(mmi[imt]),
                "IQR": stats.iqr(mmi[imt]),
            }
        else:
            results.shakemaps["mean"][imt] = shakemap.to_geotiff(mean_shakemaps, imt)
            results.shakemaps["stddevs"][imt] = shakemap.to_geotiff(
                stddev_shakemaps, imt, is_stddev=True
            )
            results.contours["mean"][imt] = shakemap.get_contours(
                imt, mean_shakemaps, contour_levels_mean
            )
            results.contours["stddevs"][imt] = shakemap.get_contours(
                imt, stddev_shakemaps, contour_levels_stddev, is_stddev=True
            )
            # Retrieve summary statistics
            results.statistics["mean"][imt] = {
                "maximum": np.exp(np.max(mean_shakemaps[imt])),
                "median": np.exp(np.mean(mean_shakemaps[imt])),
                "minimum": np.exp(np.min(mean_shakemaps[imt])),
                "IQR": stats.iqr(mean_shakemaps[imt]),
            }
        if export_folder:
            filestem = "{:s}_{:s}".format(results.earthquake.id, imt)
            # Export the bytes to raster files
            fname_raster_mean = os.path.join(export_folder, filestem + "_mean.tif")
            with open(fname_raster_mean, "wb") as f:
                f.write(results.shakemaps["mean"][imt])
            fname_raster_stddev = os.path.join(export_folder, filestem + "_stddev.tif")
            with open(fname_raster_stddev, "wb") as f:
                f.write(results.shakemaps["stddevs"][imt])
            # Export the contour dataframes to geojson
            fname_contour_mean = os.path.join(export_folder, filestem + "_contour_mean.geojson")
            results.contours["mean"][imt].to_file(fname_contour_mean, driver="GeoJSON")
            if results.contours["stddevs"][imt].shape[0]:
                # If all the sites have the same standard deviation then skip this as the
                # contours will yield an empty dataframe
                fname_contour_stddev = os.path.join(
                    export_folder, filestem + "_contour_stddev.geojson"
                )
                results.contours["stddevs"][imt].to_file(fname_contour_stddev, driver="GeoJSON")
    return results
