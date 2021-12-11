"""
Defines a set of input validation methods to check physically correct or
consistent quantities
"""
import numpy as np
from copy import deepcopy
from geopandas import GeoDataFrame
from typing import Tuple, Optional, Union, Dict
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.gsim import get_available_gsims
from shake.oq_shakemap.magnitude_scaling_relations import (
    BaseScalingRelation,
    PEERScalingRelation,
)


# OpenQuake GMPE List
GSIM_LIST = get_available_gsims()


def longitude(lon: float) -> float:
    """
    Verify that the longitude is between -180 and 180 degrees
    """
    if lon < -180.0 or lon > 180.0:
        raise ValueError("Longitude %.4f is not between -180 and 180" % lon)
    return lon


def latitude(lat: float) -> float:
    """
    Verify that the latitude is between -90 and 90 degrees
    """
    if lat < -90.0 or lat > 90.0:
        raise ValueError("Latitude %.4f is not between -90 and 90" % lat)
    return lat


def positive_float(value: float, key: str) -> float:
    """
    Verify the value is a positive float (or zero) and raise error otherwise
    """
    if value < 0.0:
        raise ValueError("%s must be positive float or zero, %.6f input" % (key, value))
    return value


def strike(value: Optional[float]) -> float:
    """
    Verify that the strike is within the range 0 to 360 degrees, or else return None if
    unspecified
    """
    if value is not None and (value < 0 or value >= 360.0):
        raise ValueError("Strike %.2f not in the range 0 to 360 degrees" % value)
    return value


def dip(value: Optional[float]) -> float:
    """
    Verify that the dip is within the range 0 to 90 degrees, or else return None if unspecified
    """
    if value is not None and (value < 0 or value > 90.0):
        raise ValueError("Dip %.2f not in the range 0 to 90 degrees" % value)
    return value


def rake(value: Optional[float]) -> float:
    """
    Verify that the rake is within the range -180 to 180 degrees, according to the Aki &
    Richards (1980) convention, or else return None if unspecified
    """
    if value is not None and (value < -180.0 or value > 180.0):
        raise ValueError("Rake %.2f not in the range -180 to 180 degrees" % value)
    return value


def mechanism(
    istrike: float, idip: float, irake: float
) -> Union[Tuple[Optional[float], Optional[float], Optional[float]], NodalPlane]:
    """
    Verifies that a valid focal mechanism is defined. A valid focal mechanism requires a
    strike, dip and rake value, which are limited to the range (0, 360), (0, 90) and
    (-180, 180) respectively. The Aki & Richards (1980) definition of rake is applied.

    Note that this only checks validity of the mechanism in terms of input value,
    not in terms of the complete physical properties of the mechanism
    """
    mechanism = (strike(istrike), dip(idip), rake(irake))
    if None in mechanism:
        # Not a valid mechanism, return tuple
        return (istrike, idip, irake)
    else:
        # Is a valid mechanism, so return openquake.hazardlib.nodalplane.NodalPlane object
        return NodalPlane(*mechanism)


def focal_mechanism(focal_mech: Optional[Dict]) -> Dict:
    """
    A focal mechanism is represented by two orthogonal planes, each plane described by a
    strike, dip and rake
    """
    if not focal_mech:
        return focal_mech
    assert "nodal_plane_1" in focal_mech, "Focal mechanism missing nodal plane 1"
    assert "nodal_plane_2" in focal_mech, "Focal mechanism missing nodal plane 2"
    focal_mechanism = {}
    for plane in ["nodal_plane_1", "nodal_plane_2"]:
        focal_mechanism[plane] = mechanism(
            focal_mech[plane]["strike"],
            focal_mech[plane]["dip"],
            focal_mech[plane]["rake"],
        )
    return focal_mechanism


def seismogenic_thickness(
    upper_seismo_depth: float, lower_seismo_depth: float
) -> Tuple[float, float]:
    """
    Verifies that a valid seismogenic thickness of the crust is defined
    """
    usd = positive_float(upper_seismo_depth, "upper seismogenic depth")
    lsd = positive_float(lower_seismo_depth, "lower seismogenic depth")
    if lsd < usd:
        raise ValueError(
            "Lower seismogenic depth %.2f km shallower than upper seismogenic "
            "depth %.2f km" % (lsd, usd)
        )
    return usd, lsd


def hypocenter_position(hypo_pos: Tuple[float, float]) -> Tuple[float, float]:
    """
    Verifies that a hypocenter position is valid within the range [0, 1] for both the
    along-strike and down-dip cases
    """
    along_strike, down_dip = hypo_pos
    if along_strike < 0.0 or along_strike > 1.0:
        raise ValueError(
            "Along strike position %.3f should be in the range 0 to 1" % along_strike
        )
    if down_dip < 0.0 or down_dip > 1.0:
        raise ValueError("Down dip position %.3f should be in the range 0 to 1" % down_dip)
    return along_strike, down_dip


def scaling_relation(msr: Optional[BaseScalingRelation]):
    """
    Verifies that the magnitude scaling relation is one supported by the
    software, or return a default is none is provided
    """
    if not msr:
        # If no relation is defined then use the default
        return PEERScalingRelation()
    if not isinstance(msr, BaseScalingRelation):
        raise TypeError(
            "Magnitude Scaling Relation %s not instance of BaseScalingRelation" % str(msr)
        )
    return msr


def regionalization_mapping(mapping: Dict) -> Dict:
    """
    Velidates a ground motion mapping to parse the ground motion model strings to instances
    of the ground motion models. Also checks the weights sum correctly to 1.0
    """
    new_mapping = {}
    for key in mapping:
        new_mapping[key] = []
        # Verify that weights sum to 1.0
        weight_sum = sum([gmm["weight"] for gmm in mapping[key]])
        weight_check = np.isclose(weight_sum, 1.0)
        assert (
            weight_check
        ), "Ground motion model weights for region %s do not sum to 1 " "(sum = %.6f)" % (
            key,
            weight_sum,
        )
        for gmm in deepcopy(mapping[key]):
            gmm_id = gmm.pop("id")
            gmm_weight = gmm.pop("weight")
            gmm_name = gmm.pop("model")
            new_mapping[key].append((gmm_id, GSIM_LIST[gmm_name](**gmm), gmm_weight))
    return new_mapping


def regionalization(regions: GeoDataFrame, mapping: Dict) -> Tuple[GeoDataFrame, Dict]:
    """
    A regionalisation is represented by a geometry set (as a geopandas.GeoDataFrame) and a
    corresponding dictionary to map the regions in the geometry set to a set of ground motion
    models (as strings of the OpenQuake input names) and respective weights. Function verifies
    that the region file has the necessary information and that a mapping for each region is
    present. Returns the region set and the mapping with instantiated ground motion model.
    """
    if not regions.crs:
        # If no coordinate reference system is defined then assume WGS84
        regions.crs = {"init": "epsg:4326"}
    if str(regions.crs) != "+init=epsg:4326 +type=crs":
        regions = regions.to_crs({"init": "epsg:4326"})
    # Verify that the region set has the necessary columns
    for col in ["REGION", "UPPER DEPTH", "LOWER DEPTH", "geometry"]:
        if col not in regions.columns:
            raise IOError("Regionalization has missing attribute %s" % col)
    # Verify that every region in the regionalization has a corresponding mapping
    for region in regions.REGION.unique():
        if region not in mapping:
            raise IOError(
                "Region %s has no corresponding ground motion model in mapping" % region
            )
    return regions, regionalization_mapping(mapping)
