"""
Classes to manage the regionalisation of ground motion models and the selection of the
ground motion model set to be used for a given earthquake
"""
from __future__ import annotations
import os
import json
import rtree
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union, Dict, Tuple, Optional, List
from pyproj import Transformer
from shapely import geometry
from openquake.hazardlib.gsim import get_available_gsims
from shake.oq_shakemap import valid
from shake.oq_shakemap.earthquake import Earthquake


# For point in polygon tests need to transform geodetic coordinates into Cartesian. For this
# we use the World Equidistance Cylindrical Projection (EPSG 4087)
transformer_world_equidistant = Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=True)

# Full GMPE set
GSIM_LIST = get_available_gsims()
# Default Regionalisation for shallow and deep regions
DEFAULT_REGIONALIZATION = {
    "default shallow": [
        ("ASK14", GSIM_LIST["AbrahamsonEtAl2014"](), 0.25),
        ("BSSA14", GSIM_LIST["BooreEtAl2014"](), 0.25),
        ("CB14", GSIM_LIST["CampbellBozorgnia2014"](), 0.25),
        ("CY14", GSIM_LIST["ChiouYoungs2014"](), 0.25),
    ],
    "default deep": [
        ("BCHydroSlabLow", GSIM_LIST["AbrahamsonEtAl2015SSlabLow"](), 0.2),
        ("BCHydroSlab", GSIM_LIST["AbrahamsonEtAl2015SSlab"](), 0.6),
        ("BCHydroSlabHigh", GSIM_LIST["AbrahamsonEtAl2015SSlabHigh"](), 0.2),
    ],
}


class Regionalization(object):
    """
    A regionalisation is defined as a set of polyogns, each of which is associated with a
    set of ground motion models and their respective weights. This class manages each
    regionalisations and, in particular, the identification of the appropriate ground
    motion model set given the location of an earthquake.

    A regionalisation is a three dimensional problem the regionalisations must be associated
    with an upper and a lower depth.

    The geometry of the regionalisation is assumed to be input as a set of polygons with
    coordinates given in terms of the of the WGS84 global geodetic coordinate system

    Attributes:
        name: A unique name describing the regionalisation
        regions: The regionalisation information as a geopandas GeoDataFrame containing the
                 columns [id, REGION, UPPER DEPTH, LOWER DEPTH, geometry]
        gsims: Dictionary of ground motion models per region in the regionalisation and the
               corresponding weights
        cartesian_regions: The regionalisation reprojected into a Cartesian framework, as an
                           instance of :class:`geopandas.GeoDataFrame`
        tree: For efficient selection of the region to which the earthquake belongs, an rtree
              spatial index is used. Maps the polygons to a corresponding rtree.index.Index
              object
    """

    def __init__(self, name: str, regions: gpd.GeoDataFrame, gsim_mapping: Dict = {}):
        """
        Instantiates the Regionalization from a complete set of regions and ground motion
        model mapping

        Args:
            name: A unique name describing the regionalisation
            regions: The regionalisation information as a geopandas GeoDataFrame containing the
                     columns [id, REGION, UPPER DEPTH, LOWER DEPTH, geometry]
            gsim_mapping: Dictionary of ground motion models per region in the regionalisation
                          and the corresponding weights
        """
        self.name = name
        self.regions, self.gsims = valid.regionalization(regions, gsim_mapping)
        self.cartesian_regions = regions.to_crs({"init": "epsg:4087"})
        # Setup the rtree
        self.tree = rtree.index.Index()
        for i, geom in enumerate(self.cartesian_regions.geometry):
            self.tree.insert(i, geom.bounds)

    def __repr__(self):
        # Returns a simple summary of the regionalization characteristics
        return "{:s} ({:g} Polygons - BBOX [{:.4f}, {:.4f}, {:.4f}, {:.4f}])".format(
            self.name,
            len(self),
            self.bounds[0],
            self.bounds[1],
            self.bounds[2],
            self.bounds[3],
        )

    def __len__(self):
        # Returns the number of regions in the regionalisation
        return self.regions.shape[0]

    def __getitem__(self, key: Union[int, str]) -> Union[pd.Series, gpd.GeoSeries]:
        """
        Returns the column of the regions GeoDataFrame if called with a string, or the
        specific row if called with an integer, otherwise raises a TypeError

        Args:
            key: Either the Region attribute (column) from the dataframe or an integer to
                 retrieve a specific region (row)

        Returns:
            Column or row from the dataframe
        """
        if isinstance(key, int):
            return self.regions.iloc[key, :]
        elif isinstance(key, str):
            return self.regions[key]
        else:
            raise TypeError("Unrecognised data type %s used for indexing" % type(key))

    def __contains__(self, earthquake: Earthquake):
        """
        Determines if an earthquake object is within the bounds of the
        region set

        Args:
            earthquake: An earthquake represented by the
                        :class:`shakyground2.earthquake.Earthquake`
        Returns:
            True (if the earthquake is within the bounding box of the regionalisation) or
            False otherwise
        """
        llon, llat, ulon, ulat = self.bounds
        return (
            (earthquake.lon >= llon)
            & (earthquake.lon <= ulon)
            & (earthquake.lat >= llat)
            & (earthquake.lat <= ulat)
        )

    def __call__(self, earthquake: Earthquake) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Returns the tectonic region and corresponding set of ground motion models and weights
        to be used for the earthquake depending on the region in which the earthquake falls

        Args:
            earthquake: An earthquake represented by the
                        :class:`shakyground2.earthquake.Earthquake`
        Returns:
            region: The name of the region to which the earthquake belongs, or None if the
                    earthquake is not within the regionalization
            gmm: The ground motion model set (with weights) of the region to which the
                 earthquake belongs, or None if the earthquake is within the regionalization
        """
        if earthquake not in self:
            return None, None
        # Transform event long, lat into cartesian x, y and store as shapely Point object
        eqx, eqy = transformer_world_equidistant.transform(earthquake.lon, earthquake.lat)
        eqxy = geometry.Point(eqx, eqy)
        for idx in self.tree.intersection(eqxy.bounds):
            depth_idx = (earthquake.depth >= self.regions["UPPER DEPTH"][idx]) and (
                earthquake.depth <= self.regions["LOWER DEPTH"][idx]
            )
            if depth_idx and self.cartesian_regions.geometry[idx].contains(eqxy):
                # Earthquake within the depth range and within the zone
                region = self[idx].REGION
                return region, self.gsims[region]
        # In theory this can only happen if the earthquake is within the
        # bounding box of the region but outside of the depth range
        return None, None

    @property
    def bounds(self) -> np.ndarray:
        # Bounding box of the entire regionalisation
        return self.regions.total_bounds

    @classmethod
    def from_json(cls, name: str, geojson_file: str, gsim_mapping_file: str) -> Regionalization:
        """
        Construct the Regionalization from a json representation of the regionalization
        and the ground motion model mapping

        Args:
            name: Name of regionalization
            geojson_file: Path to the geojson file containing the region geometries and
                          related attributes
            gsim_mapping_file: Path to the json file containing the ground motion model
                               mappings
        """
        dframe = gpd.GeoDataFrame.from_file(geojson_file, driver="GeoJSON")
        # If a mapping file is provided then load one in
        with open(gsim_mapping_file, "r") as f:
            gsim_mapping = json.load(f)
        return cls(name, dframe, gsim_mapping)


class RegionalizationSet(object):
    """
    A Regionalization defines a set of geographical regions and their associated set of
    ground motion models and weights. But comprehensive partition of a target region (which
    may correspond to the entire globe) may require multiple regionalizations to be defined.
    One such example might be that a particular regionalization is required for a country that
    may define different region types or different ground motion model sets from that which
    may be defined elsewhere. The RegionalizationSet represents a set of regionalizations,
    the order of which defines the priority in which they are applied.

    As an example:

    Consider three regionalizations: A) Regions within a country, ii) Regions within a
    Countinent to which country A belongs, C) A set of regions spanning the globe.

    If the regionalizations are input in the order A B C, then an earthquake falling within
    country A will be subject to the regionalization of A rather than B or C even though it
    falls within the domain covered by all three. If the order were reversed (C B A) then the
    global regionalization would be applied to the earthquake even if it fell within the
    domains covered by A and B.

    If the earthquake does not fall within the domain of any of the regions in the set then
    a "default" regionalisation is applied, which depends on whether the earthquake is
    "shallow" (depth <= 40 km) or "deep" (> 40 km).

    For the "shallow" regionalization the four primary NGA West 2 ground motion models are
    adopted with equal weighting (Abrahamson et al., 2014; Boore et al., 2014;
    Campbell & Bozorgnia, 2014; Chiou & Youngs, 2014)

    For the "deep" regionalisation the subduction inslab ground motion model of
    Abrahamson et al. (2016) is adopted, with the additional epistemic uncertainty factors.

    Attributes:
        regionalizations: A set of regionalizations as a list of :class:`Regionalization`
    """

    def __init__(self, regionalizations):
        self.regionalizations = regionalizations

    @classmethod
    def from_json(
        cls, names: List, regionalization_files: List, gsim_mapping_files: List
    ) -> RegionalizationSet:

        # Check if any file is missing before parsing the regionalizations
        assert len(names) == len(regionalization_files) == len(gsim_mapping_files)
        # Before importing model, check that all files are present
        for regionalization_file, gsim_mapping_file in zip(
            regionalization_files, gsim_mapping_files
        ):
            if not os.path.exists(regionalization_file):
                raise IOError("Regionalization file %s not found" % regionalization_file)
            if not os.path.exists(gsim_mapping_file):
                raise IOError("GSIM mapping file %s not found" % gsim_mapping_file)
        # Load in the regionalizations
        regionalizations = []
        for name, regionalization_file, mapping_file in zip(
            names, regionalization_files, gsim_mapping_files
        ):
            regionalizations.append(
                Regionalization.from_json(name, regionalization_file, mapping_file)
            )
        return cls(regionalizations)

    def __len__(self):
        return len(self.regionalizations)

    def __iter__(self):
        for regionalization in self.regionalizations:
            yield regionalization

    def __call__(self, earthquake: Earthquake):
        """
        Returns the tectonic region and corresponding set of ground motion models and weights
        to be used for the earthquake depending on the region in which the earthquake falls.
        If no region is defined then a default region type and ground motion model set is
        assigned depending on whether the earthquake is "shallow" (< 40 km) or "deep" (> 40 km)

        Args:
            earthquake: An earthquake represented by the
                        :class:`shakyground2.earthquake.Earthquake`
        Returns:
            region: The name of the region to which the earthquake belongs, or None if the
                    earthquake is not within the regionalization
            gmm: The ground motion model set (with weights) of the region to which the
                 earthquake belongs, or None if the earthquake is within the regionalization
        """
        for regionalization in self:
            region, gmms = regionalization(earthquake)
            if region and gmms:
                return region, gmms
        # If earthquake is not assigned to any zone then use the default ground motion model
        # set, depending on whether the earthquake depth is shallow or deep
        if earthquake.depth > 40.0:
            default_reg = "default deep"
        else:
            default_reg = "default shallow"
        return default_reg, DEFAULT_REGIONALIZATION[default_reg]


# Path to data file directory
REGIONALIZATION_DIRECTORY = os.path.join(os.path.dirname(__file__), "regionalization_files")


# Path to default regionalization data files
DEFAULT_GLOBAL_REGIONALIZATION_REGIONS = [
    os.path.join(REGIONALIZATION_DIRECTORY, "germany.geojson"),  # Germany
    os.path.join(REGIONALIZATION_DIRECTORY, "eshm20.geojson"),  # ESHM20
    os.path.join(REGIONALIZATION_DIRECTORY, "global_volcanic.geojson"),  # Global volcanic zones
    os.path.join(REGIONALIZATION_DIRECTORY, "global_stable.geojson"),  # Global stable regions
]


# Corresponding default GSIM mapping files
DEFAULT_GLOBAL_GSIM_MAPPING = [
    os.path.join(REGIONALIZATION_DIRECTORY, "germany.json"),
    os.path.join(REGIONALIZATION_DIRECTORY, "eshm20.json"),
    os.path.join(REGIONALIZATION_DIRECTORY, "global_volcanic.json"),
    os.path.join(REGIONALIZATION_DIRECTORY, "global_stable.json"),
]


# Default global regionalization
DEFAULT_GLOBAL_REGIONALIZATION = RegionalizationSet.from_json(
    ["Germany", "ESHM20", "Global Volcanic", "Global Stable"],  # Name set
    DEFAULT_GLOBAL_REGIONALIZATION_REGIONS,  # Geographical region set
    DEFAULT_GLOBAL_GSIM_MAPPING,  # GSIM mapping set
)
