"""
Core class to define a set of target sites for the shakemap calculation
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Union
from openquake.hazardlib.site import SiteCollection


# Note that OpenQuake unit conventions are being adopted here for consistency
# with the OpenQuake software itself
SITE_PROPERTIES = {
    "sids": np.int64,  # Unique site identifiers
    "lon": np.float64,  # Longitudes (in decimal degrees)
    "lat": np.float64,  # Latitudes (in decimal degrees)
    "depth": np.float64,  # Site depth (in km, negative indicates above ground)
    "vs30": np.float64,  # Vs30 of the sites (in m/s)
    "vs30measured": np.bool,  # Vs30 is measured (True) or inferred (False)
    "z1pt0": np.float64,  # Depth (m) to the Vs 1.0 km/s layer
    "z2pt5": np.float64,  # Depth (km) to the Vs 2.5 km/s layer
    "xvf": np.float64,  # Distance (km) to the volcanic front
    "backarc": np.bool,  # Site is in the subduction backarc (True) or else
    "region": np.int32,  # Region to which the site belongs
    "geology": (np.string_, 20),  # Geological classification for the site
}


# In some cases reasonable default values can be used for relevant ground motion models
# unless specifically defined by a user
SITE_DEFAULTS = {
    "vs30measured": True,
    "xvf": 150.0,
    "region": 0,
    "geology": "UNCLASSIFIED",
    "backarc": False,
}


class SiteModel(object):
    """
    Class to manage the definition of the site properties for application in a Shakemap

    Attributes:
        site_array: Array of sites and their corresponding properties as numpy.ndarray
        bbox_properties: Properties of the bounding box used to create the site array, if it
                         has been constructed from a bounding box. If the site model was not
                         constructed from a bounding box then "bbox_properties" is not needed
                         except if the site model spans more than one hemisphere without
                         crossing the meridian, in which case this should contain
                         "cross_antimeridian: False".
    """

    def __init__(self, site_array: np.ndarray, bbox_properties: Optional[Dict] = None):
        if bbox_properties is None:
            bbox_properties = {}
        self.site_array = site_array
        if "cross_antimeridian" in bbox_properties:
            self.cross_antimeridian = bbox_properties.pop("cross_antimeridian")
        else:
            bbox = [
                np.min(site_array["lon"]),
                np.min(site_array["lat"]),
                np.max(site_array["lon"]),
                np.max(site_array["lat"]),
            ]
            if (bbox[2] - bbox[0]) > 180.0:
                warnings.warn(
                    "Upper longitude of bounding box exceeds lower longitude "
                    "by more than 180.0. Assuming antimeridian crossing. If this is "
                    "not the case set `bbox_properties={'cross_antimeridian': False}"
                    " in the call to the invoking function"
                )
                self.cross_antimeridian = True
            else:
                self.cross_antimeridian = False
        self.bbox_properties = bbox_properties

    def __len__(self):
        # Returns the number of sites in the site array
        return self.site_array.shape[0]

    def __getitem__(self, key: Union[str, int]):
        # Returns the named column from the site array (if provided with a string),
        # or the corresponding row if provided with an integer
        if isinstance(key, str) or isinstance(key, int):
            return self.site_array[key]
        else:
            raise KeyError("Site model has no attribute %s" % key)

    def __repr__(self):
        # Returns an informative string regarding the composition of the sites,
        # including the bounding box information when relevant
        info_string = "SiteModel(%g sites)" % len(self)
        if self.bbox_properties:
            bbox_string = " BBOX: %.3f/%.3f/%.3f/%.3f DX: %.3e DY: %.3e" % (
                self.bbox_properties["bbox"][0],
                self.bbox_properties["bbox"][1],
                self.bbox_properties["bbox"][2],
                self.bbox_properties["bbox"][3],
                self.bbox_properties["spcx"],
                self.bbox_properties["spcy"],
            )
            info_string += bbox_string
        return info_string

    @property
    def shape(self):
        return self.site_array.shape[0]

    @property
    def dataframe(self):
        # Return the site array as a dataframe
        return pd.DataFrame(self.site_array)

    def get_site_collection(self, idx: Optional[np.ndarray] = None):
        """
        Convert the site model to an OpenQuake SiteCollection, filtering the site
        model where defined
        Args:
            idx: Index of rows in the site model to be returned in the SiteCollection, input
                 as either a logical vector (numpy.ndarray) or a vector of indices
                 (numpy.ndarray)
        Returns:
            Site collection as instance of :class:`openquake.hazardlib.site.SiteCollection`
        """
        # The site collection is usually instantiated with a list of single Site objects
        # and then passed to the site array. This step is bypassed and the SiteCollection's
        # array is build directly from the site arrays
        site_col = object.__new__(SiteCollection)
        site_col.complete = site_col
        if idx is not None:
            # Filter the site array
            site_col.array = self.site_array[idx]
        else:
            site_col.array = self.site_array.copy()
        site_col.array.flags.writeable = False
        return site_col

    @classmethod
    def from_bbox(
        cls,
        bbox: Tuple[float, float, float, float],
        spcx: float,
        spcy: float,
        vs30: float,
        vs30measured: bool = True,
        z1pt0: Optional[float] = None,
        z2pt5: Optional[float] = None,
        xvf: float = 150.0,
        backarc: bool = False,
        region: int = 0,
        geology: np.string_ = "UNCLASSIFIED",
    ) -> SiteModel:
        """
        Builds a site model from a bounding box and scalar site properties to be
        fixed for all sites

        Args:
            bbox:
                Bounding box of a region as [llon, llat, ulon, ulat]
            spcx:
                Fixed longitude spacing of successive points in decimal degrees
            spcy:
                Fixed latitude spacing of successive points in decimal degrees
            vs30:
                Fixed Vs30 assigned to all points
            vs30measured:
                Indicates whether the Vs30 refers to a "measured" site condition
                (True) or "inferred" (False)
            z1pt0:
                Depth to Vs 1.0 km layer (in m)
            z2pt5:
                Depth to Vs 2.5 km layer (in km)

        Returns:
            Instance of :class`shakygroundv2.site_model.SiteModel`
        """

        # Generate the mesh of points
        llon, llat, ulon, ulat = bbox
        if (llon > ulon) and (np.fabs(llon - ulon) > 180.0):
            # Bounding box crosses the antimeridian
            lons, lats = np.meshgrid(
                np.arange(llon, ulon + 360.0 + spcx, spcx), np.arange(llat, ulat + spcy, spcy)
            )
            cross_antimeridian = True
            lons[lons > 180.0] -= 360.0
        else:

            lons, lats = np.meshgrid(
                np.arange(llon, ulon + spcx, spcx), np.arange(llat, ulat + spcy, spcy)
            )
            cross_antimeridian = False
        nrow, ncol = lons.shape
        nsites = nrow * ncol
        self = object.__new__(cls)
        # To re-shape or export of the shakemap back into a regular grid you need the original
        # configuration properties used to build the site model
        bbox_properties = {
            "bbox": bbox,
            "spcx": spcx,
            "spcy": spcy,
            "ncol": ncol,
            "nrow": nrow,
            "cross_antimeridian": cross_antimeridian,
        }
        # Site array has a set of defined datatypes
        site_dtype = np.dtype(list(SITE_PROPERTIES.items()))
        site_array = np.zeros(nsites, site_dtype)
        site_array["sids"] = np.arange(nsites, dtype=np.int64)
        site_array["lon"] = lons.flatten()
        site_array["lat"] = lats.flatten()
        site_array["vs30"] = vs30 * np.ones(nsites)
        site_array["vs30measured"] = np.ones(nsites, dtype=bool)
        # In the cases of Z1.0 and Z2.5 these can be calculated from
        # Vs30 if not defined explicitly
        if z1pt0 is None:
            site_array["z1pt0"] = cls.vs30_to_z1pt0_cy14(site_array["vs30"])
        else:
            site_array["z1pt0"] = z1pt0 * np.ones(nsites)

        if z2pt5 is None:
            site_array["z2pt5"] = cls.vs30_to_z2pt5_cb14(site_array["vs30"])
        else:
            site_array["z2pt5"] = z2pt5 * np.ones(nsites)

        # Get other default options from kwargs if specified
        site_array["xvf"] = xvf * np.ones(nsites, dtype=SITE_PROPERTIES["xvf"])
        site_array["backarc"] = self._get_boolean_array(nsites, backarc)
        site_array["region"] = region * np.ones(nsites, dtype=SITE_PROPERTIES["region"])
        site_array["geology"] = self._get_string_array(
            nsites, geology, SITE_PROPERTIES["geology"]
        )
        return cls(site_array, bbox_properties=bbox_properties)

    @classmethod
    def from_dataframe(
        cls, dframe: pd.DataFrame, bbox_properties: Optional[Dict] = None
    ) -> SiteModel:
        """
        Builds the site collection directly from a pandas dataframe

        Args:
            dframe: Input site model as a pandas.Dataframe

        Returns:
            Instance of :class`shakygroundv2.site_model.SiteModel`
        """
        if bbox_properties is None:
            bbox_properties = {}
        nsites = dframe.shape[0]
        site_dtype = np.dtype(list(SITE_PROPERTIES.items()))
        site_array = np.zeros(nsites, site_dtype)
        site_array["sids"] = np.arange(nsites)
        for col in dframe.columns:
            if col not in SITE_PROPERTIES:
                warnings.warn(
                    "Input site property %s not recognised" " for use - skipping" % col
                )
                continue
            site_array[col] = dframe[col].to_numpy(dtype=SITE_PROPERTIES[col])
        # Fill in any missing information from the defaults
        if "z1pt0" not in dframe.columns:
            site_array["z1pt0"] = cls.vs30_to_z1pt0_cy14(site_array["vs30"])
        if "z2pt5" not in dframe.columns:
            site_array["z2pt5"] = cls.vs30_to_z2pt5_cb14(site_array["vs30"])
        for key, value in SITE_DEFAULTS.items():
            if key not in dframe.columns:
                # Use the default
                if SITE_PROPERTIES[key] in ((np.bytes_, 20),):
                    site_array[key] = cls._get_string_array(nsites, value, SITE_PROPERTIES[key])
                elif SITE_PROPERTIES[key] == np.bool:
                    site_array[key] = cls._get_boolean_array(nsites, value)
                else:
                    site_array[key] = value * np.ones(nsites, dtype=SITE_PROPERTIES[key])
        return cls(site_array, bbox_properties)

    @staticmethod
    def _get_string_array(num_elem: int, key: str, dtype) -> np.ndarray:
        """
        Returns an array of constant string values

        Args:
            num_elem: Number of elements in the array
            key: String to fill the array
            dtype: Numpy datatype for the string

        Returns:
            Numpy string vector
        """
        arr = np.empty(num_elem, dtype=dtype)
        arr[:] = key
        return arr

    @staticmethod
    def _get_boolean_array(num_elem: int, value: bool) -> np.ndarray:
        """
        Returns an array of boolean values

        Args:
            num_elem: Number of elements in the array
            value: Indicates if the array should be True of False
        """
        arr = np.empty(num_elem, dtype=bool)
        arr[:] = value
        return arr

    @staticmethod
    def vs30_to_z1pt0_cy14(vs30: np.ndarray, japan: bool = False) -> np.ndarray:
        """
        Returns the estimate depth to the 1.0 km/s velocity layer based on Vs30
        from Chiou & Youngs (2014) California model or Japan model

        Args:
            vs30:
                Input Vs30 values in m/s
            japan:
                If true then this returns the Z1.0 values according to the Japan model,
                otherwise the California model
        Returns:
                Z1.0 values in m
        """
        if japan:
            c1 = 412.0 ** 2.0
            c2 = 1360.0 ** 2.0
            return np.exp((-5.23 / 2.0) * np.log((np.power(vs30, 2.0) + c1) / (c2 + c1)))
        else:
            c1 = 571.0 ** 4.0
            c2 = 1360.0 ** 4.0
            return np.exp((-7.15 / 4.0) * np.log((vs30 ** 4.0 + c1) / (c2 + c1)))

    @staticmethod
    def vs30_to_z2pt5_cb14(vs30: np.ndarray, japan: bool = False) -> np.ndarray:
        """
        Converts vs30 to depth to 2.5 km/s interface using model proposed by
        Campbell & Bozorgnia (2014)

        Args:
            vs30:
                Input Vs30 values in m/s
            japan:
                If true then this returns the Z1.0 values according to the Japan model,
                otherwise the California model
        Returns:
                Z2.5 values in km
        """
        if japan:
            return np.exp(5.359 - 1.102 * np.log(vs30))
        else:
            return np.exp(7.089 - 1.144 * np.log(vs30))
