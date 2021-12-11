"""
Defines a set of Ground Motion to Intensity Conversion Equations (GMICEs) to convert
shakemaps from the ground motion to macroseismic intensity. Includes:
`class`: GMICE,
`class`: WordenEtAl2012,
`class`: WordenEtAl2012MRDependent,
`class`: AtkinsonKaka2007,
`class`: AtkinsonKaka2007MRDependent
"""

import abc
import numpy as np
from typing import Dict, Tuple, Set
from scipy.constants import g
from openquake.hazardlib.gsim.base import SitesContext, DistancesContext, RuptureContext

LOG_FACT = np.log10(np.e)


class GMICE(metaclass=abc.ABCMeta):
    """
    Abstract base class for conversion equations between observed ground motion parameters and
    macroseismic intensity

    Attributes:
        sctx: Site properties as instance of `class`:
              openquake.hazardlib.gsim.base.SitesContext
        rctx: Rupture properties as instance of `class`:
              openquake.hazardlib.gsim.base.RuptureContext
        dctx: Distances as instance of `class`:
              openquake.hazardlib.gsim.base.DistancesContext
    """

    @property
    @abc.abstractmethod
    def REQUIRES_DISTANCES(self) -> Set:
        """Shakemap source-to-site distances required by the GMICE"""
        pass

    @property
    @abc.abstractmethod
    def REQUIRES_SITES_PARAMETERS(self) -> Set:
        """Shakemap site properties required by the GMICE"""
        pass

    @property
    @abc.abstractmethod
    def REQUIRES_RUPTURE_PARAMETERS(self) -> Set:
        """Rupture attributes required by the GMICE"""
        pass

    @property
    @abc.abstractmethod
    def COEFFS(self) -> Dict:
        """Coefficients of the GMICE as a dictionary organised by intensity measure type"""
        pass

    def __init__(self, sctx, rctx, dctx):
        """
        Constructs the class using the SitesContext, RuptureContext and DistancesContext
        objects
        """
        self.sctx = SitesContext()
        self.dctx = DistancesContext()
        self.rctx = RuptureContext()
        self._setup_params(sctx, rctx, dctx)

    def _setup_params(self, sctx, rctx, dctx):
        """
        Retrieves the required site, rupture and distance attributes from the input context
        objects
        """
        for key in self.REQUIRES_RUPTURE_PARAMETERS:
            setattr(self.rctx, key, getattr(rctx, key))
        for key in self.REQUIRES_SITES_PARAMETERS:
            setattr(self.sctx, key, getattr(sctx, key))
        for key in self.REQUIRES_DISTANCES:
            setattr(self.dctx, key, getattr(dctx, key))

    @abc.abstractmethod
    def get_mean_intensity_and_stddev(
        self, imt: str, log_gmvs: np.ndarray, gmv_total_stddev: np.ndarray, clip: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main function to convert ground motions into macroseismic intensities

        Args:
            imt: Intensity measure type
            log_gmvs: Natural logarithm of the ground motion values
            gmv_total_stddevs: Total standard deviation of the natural logarithm
                               of ground motion values
            clip: Clips the resulting MMI to the range MMI 1 to MMI 10
        Returns:
            mmi: Converted macroseismic intensity
            stddev: Standard deviation of the macroseismic intensity after full
                    error propagation of the ground motion uncertainty
        """


class WordenEtAl2012(GMICE):
    """
    Implements the Ground Motion Intensity Conversion Equation of Worden et al. (2012) -
    without magnitude and distance predictor variables

    Worden CB, Gerstenberger MC, Rhoades DA and Wald DJ (2012), "Probabilistic Relationships
    between Ground-Motion Parameters and Modified Mercalli Intensity in California", Bulletin
    of the Seismological Society of America, 102(1), pp 204 - 221,  doi: 10.1785/0120110156

    Includes a low intensity extension for MMI < 2 as explained in
    https://github.com/usgs/shakemap/blob/master/shakelib/gmice/wgrw12.py
    """

    @property
    def REQUIRES_DISTANCES(self) -> Set:
        """No distance predictors required"""
        return {}

    @property
    def REQUIRES_SITES_PARAMETERS(self) -> Set:
        """No site predictors required"""
        return {}

    @property
    def REQUIRES_RUPTURE_PARAMETERS(self) -> Set:
        """No rupture predictors required"""
        return {}

    # Define the order of preference of the IMTs for conversion
    ORDER_OF_PREFERENCE = ["PGV", "PGA", "SA(1.0)", "SA(0.3)", "SA(3.0)"]

    def get_mean_intensity_and_stddev(
        self, imt: str, log_gmvs: np.ndarray, gmv_total_stddev: np.ndarray, clip: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean intensity and standard deviation. A tri-linear model is assumed in
        which two segments are defined according to Equation 3. The third segment extends the
        model to MMI < 2. Returned MMI values are limited to the range 1 - 10
        """
        if imt not in self.COEFFS:
            raise ValueError(
                "Conversion to intensity for IMT %s unsupported for %s"
                % (imt, self.__class__.__name__)
            )
        if imt.upper() != "PGV":
            # Accelerations are in terms of fractions of g, convert to cm/s/s
            log10_gmvs = np.log10(100.0 * g) + log_gmvs * LOG_FACT
        else:
            # PGV in terms of ln cm/s, convert to log10
            log10_gmvs = log_gmvs * LOG_FACT
        C = self.COEFFS[imt]
        C2 = self.COEFFS_2[imt]
        mmi = np.zeros(log_gmvs.shape)
        idx = log10_gmvs <= C2["t1"]
        mmi[idx] = C2["c1"] + C2["c2"] * log10_gmvs[idx]
        idx = np.logical_and(log10_gmvs > C2["t1"], log10_gmvs <= C["t1"])
        mmi[idx] = C["c1"] + C["c2"] * log10_gmvs[idx]
        idx = log10_gmvs > C["t1"]
        mmi[idx] = C["c3"] + C["c4"] * log10_gmvs[idx]
        stddev = self.get_stddev(C, C2, gmv_total_stddev, log10_gmvs)
        if clip:
            mmi = np.clip(mmi, 1.0, 10.0)
        return mmi, stddev

    def get_stddev(
        self, C: Dict, C2: Dict, gmv_total_stddev: np.ndarray, log_gmvs: np.ndarray
    ) -> np.ndarray:
        """
        Returns the standard deviation of MMI after error propagation. Note that in this case
        the input ground motion values and ground motion standard deviations are given in terms
        of their natural logarithm

        Args:
            C: Coefficients for model above MMI 2
            C2: Coefficients for model below MMI 2
            gmv_total_stddevs: Total standard deviations of the original ground motion values
                               in common logarithm units
            log_gmvs: Ground motion values in common logarithm units

        Returns:
            Standard deviation of MMI
        """
        stddevs = np.zeros(gmv_total_stddev.shape)
        idx = log_gmvs <= C2["t1"]
        stddevs[idx] = (gmv_total_stddev[idx] * C2["c2"] * LOG_FACT) ** 2.0 + C[
            "sigma_mmi"
        ] ** 2.0
        idx = np.logical_and(log_gmvs > C2["t1"], log_gmvs <= C["t1"])
        stddevs[idx] = (gmv_total_stddev[idx] * C["c2"] * LOG_FACT) ** 2.0 + C[
            "sigma_mmi"
        ] ** 2.0
        idx = log_gmvs >= C["t1"]
        stddevs[idx] = (gmv_total_stddev[idx] * C["c4"] * LOG_FACT) ** 2.0 + C[
            "sigma_mmi"
        ] ** 2.0
        return np.sqrt(stddevs)

    @property
    def COEFFS(self) -> Dict:
        """Coefficients for the bi-linear model according to Table 1"""
        return {
            "PGA": {
                "c1": 1.78,
                "c2": 1.55,
                "c3": -1.6,
                "c4": 3.7,
                "t1": 1.57,
                "sigma_mmi": 0.73,
            },
            "PGV": {
                "c1": 3.78,
                "c2": 1.47,
                "c3": 2.89,
                "c4": 3.16,
                "t1": 0.53,
                "sigma_mmi": 0.65,
            },
            "SA(0.3)": {
                "c1": 1.26,
                "c2": 1.69,
                "c3": -4.15,
                "c4": 4.14,
                "t1": 2.21,
                "sigma_mmi": 0.84,
            },
            "SA(1.0)": {
                "c1": 2.50,
                "c2": 1.51,
                "c3": 0.20,
                "c4": 2.90,
                "t1": 1.65,
                "sigma_mmi": 0.80,
            },
            "SA(3.0)": {
                "c1": 3.81,
                "c2": 1.17,
                "c3": 1.99,
                "c4": 3.01,
                "t1": 0.99,
                "sigma_mmi": 0.95,
            },
        }

    # Coefficients for the low intensity extension from
    # https://github.com/usgs/shakemap/blob/master/shakelib/gmice/wgrw12.py
    COEFFS_2 = {
        "PGA": {"c1": 1.71, "c2": 2.08, "t1": 0.14},
        "PGV": {"c1": 4.62, "c2": 2.17, "t1": -1.21},
        "SA(0.3)": {"c1": 1.15, "c2": 1.92, "t1": 0.44},
        "SA(1.0)": {"c1": 2.71, "c2": 2.17, "t1": -0.33},
        "SA(3.0)": {"c1": 7.35, "c2": 3.45, "t1": -1.55},
    }


class WordenEtAl2012MRDependent(WordenEtAl2012):
    """
    Implements the Ground Motion Intensity Conversion Equation of Worden et al. (2012) - with
    magnitude and distance predictor variables

    Worden CB, Gerstenberger MC, Rhoades DA and Wald DJ (2012), "Probabilistic Relationships
    between Ground-Motion Parameters and Modified Mercalli Intensity in California", Bulletin
    of the Seismological Society of America, 102(1), pp 204 - 221,  doi: 10.1785/0120110156

    Includes a low intensity extension for MMI < 2 as explained in
    https://github.com/usgs/shakemap/blob/master/shakelib/gmice/wgrw12.py
    """

    # Requires rupture distance
    @property
    def REQUIRES_DISTANCES(self) -> Set:
        return {
            "rrup",
        }

    # Requires magnitude
    @property
    def REQUIRES_RUPTURE_PARAMETERS(self) -> Set:
        return {
            "mag",
        }

    def get_mean_intensity_and_stddev(
        self, imt: str, log_gmvs: np.ndarray, gmv_total_stddev: np.ndarray, clip: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean intensity and standard deviation by adding on the magitude and
        distance factors to the mean intensity provided by the magnitude- and distance-
        independent form of the model.
        """
        if imt not in self.COEFFS:
            raise ValueError(
                "Conversion to intensity for IMT %s unsupported for %s"
                % (imt, self.__class__.__name__)
            )
        C = self.COEFFS[imt]
        mmi, stddev = super().get_mean_intensity_and_stddev(
            imt, log_gmvs, gmv_total_stddev, clip=False
        )
        mmi += C["c5"] + C["c6"] * np.log10(self.dctx.rrup) + C["c7"] * self.rctx.mag
        if clip:
            mmi = np.clip(mmi, 1.0, 10.0)
        return mmi, stddev

    @property
    def COEFFS(self) -> Dict:
        """Coefficients for the magnitude- and distance-dependent bi-linear model according to
        Tables 1 and 2
        """
        return {
            "PGA": {
                "c1": 1.78,
                "c2": 1.55,
                "c3": -1.6,
                "c4": 3.7,
                "t1": 1.57,
                "c5": -0.91,
                "c6": 1.02,
                "c7": -0.17,
                "sigma_mmi": 0.66,
            },
            "PGV": {
                "c1": 3.78,
                "c2": 1.47,
                "c3": 2.89,
                "c4": 3.16,
                "t1": 0.53,
                "c5": 0.90,
                "c6": 0.0,
                "c7": -0.18,
                "sigma_mmi": 0.63,
            },
            "SA(0.3)": {
                "c1": 1.26,
                "c2": 1.69,
                "c3": -4.15,
                "c4": 4.14,
                "t1": 2.21,
                "c5": -1.05,
                "c6": 0.60,
                "c7": 0.00,
                "sigma_mmi": 0.82,
            },
            "SA(1.0)": {
                "c1": 2.50,
                "c2": 1.51,
                "c3": 0.20,
                "c4": 2.90,
                "t1": 1.65,
                "c5": 2.27,
                "c6": -0.49,
                "c7": -0.29,
                "sigma_mmi": 0.75,
            },
            "SA(3.0)": {
                "c1": 3.81,
                "c2": 1.17,
                "c3": 1.99,
                "c4": 3.01,
                "t1": 0.99,
                "c5": 1.91,
                "c6": -0.57,
                "c7": -0.21,
                "sigma_mmi": 0.89,
            },
        }

    # Coefficients for the low intensity extension from
    # https://github.com/usgs/shakemap/blob/master/shakelib/gmice/wgrw12.py
    COEFFS_2 = {
        "PGA": {"c1": 1.71, "c2": 2.08, "t1": 0.14},
        "PGV": {"c1": 4.62, "c2": 2.17, "t1": -1.21},
        "SA(0.3)": {"c1": 1.15, "c2": 1.92, "t1": 0.44},
        "SA(1.0)": {"c1": 2.71, "c2": 2.17, "t1": -0.33},
        "SA(3.0)": {"c1": 7.35, "c2": 3.45, "t1": -1.55},
    }


class AtkinsonKaka2007(GMICE):
    """
    Implements the Ground Motion Intensity Conversion Equation (GMICE) of Atkinson & Kaka
    (2007) - without magnitude and distance predictors

    Atkinson GA and Kaka SI (2007), "Relationships between Felt Intensity and Instrumental
    Ground Motion in the Central United States and California", Bulletin of the Seismological
    Society of America, 97(2), pp 497 - 510, doi: 10.1785/0120060154

    Note that in their original formulation Atkinson & Kaka (2007) define conversion
    coefficients for SA(2.0), yet the USGS Shakemap applies these to SA(3.0) instead,
    presumably for consistency with the other GMICEs (e.g. Worden et al., 2012). We follow
    the approach of the USGS and apply these coefficients to SA(3.0) rather than SA(2.0)
    """

    @property
    def REQUIRES_DISTANCES(self) -> Set:
        """No distance predictors required"""
        return {}

    @property
    def REQUIRES_SITES_PARAMETERS(self) -> Set:
        """No site predictors required"""
        return {}

    @property
    def REQUIRES_RUPTURE_PARAMETERS(self) -> Set:
        """No rupture predictors required"""
        return {}

    # Define the order of preference of the IMTs for conversion
    ORDER_OF_PREFERENCE = ["PGV", "PGA", "SA(1.0)", "SA(3.0)", "SA(0.3)"]

    def get_mean_intensity_and_stddev(
        self, imt: str, log_gmvs: np.ndarray, gmv_total_stddev: np.ndarray, clip: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean intensity and standard deviation for the magnitude- and distance-
        independent form of the model
        """
        if imt not in self.COEFFS:
            raise ValueError(
                "Conversion to intensity for IMT %s unsupported for %s"
                % (imt, self.__class__.__name__)
            )
        if imt.upper() != "PGV":
            # Accelerations are in terms of fractions of g, convert to cm/s/s
            log10_gmvs = np.log10(100.0 * g) + log_gmvs * LOG_FACT
        else:
            # PGV in terms of ln cm/s, convert to log10
            log10_gmvs = log_gmvs * LOG_FACT
        # Convert stddevs from natural log units to common log units
        C = self.COEFFS[imt]
        mmi = np.zeros(log_gmvs.shape)
        idx = log10_gmvs <= C["t1"]
        mmi[idx] = C["c1"] + C["c2"] * log10_gmvs[idx]
        idx = np.logical_not(idx)
        mmi[idx] = C["c3"] + C["c4"] * log10_gmvs[idx]
        stddev = self.get_stddev(C, gmv_total_stddev, log10_gmvs)
        if clip:
            mmi = np.clip(mmi, 1.0, 10.0)
        return mmi, stddev

    def get_stddev(
        self, C: Dict, gmv_total_stddev: np.ndarray, log_gmvs: np.ndarray
    ) -> np.ndarray:
        """
        Returns the standard deviation of MMI after error propagation. Note that in this case
        the input ground motion values and ground motion standard deviations are given in terms
        of common logarithm rather than their original natural logarithm

        Args:
            C: Coefficients for model above MMI 2
            C2: Coefficients for model below MMI 2
            gmv_total_stddevs: Total standard deviations of the original ground
                               motion values in common logarithm units
            log_gmvs: Ground motion values in common logarithm units

        Returns:
            Standard deviation of MMI
        """
        stddevs = np.zeros(gmv_total_stddev.shape)
        idx = log_gmvs <= C["t1"]
        stddevs[idx] = (gmv_total_stddev[idx] * C["c2"] * LOG_FACT) ** 2.0 + C[
            "sigma_mmi"
        ] ** 2.0
        idx = np.logical_not(idx)
        stddevs[idx] = (gmv_total_stddev[idx] * C["c4"] * LOG_FACT) ** 2.0 + C[
            "sigma_mmi"
        ] ** 2.0
        return np.sqrt(stddevs)

    @property
    def COEFFS(self) -> Dict:
        """Coefficients of the GMICE as defined in Table 4"""
        return {
            "PGA": {
                "c1": 2.65,
                "c2": 1.39,
                "c3": -1.91,
                "c4": 4.09,
                "t1": 1.69,
                "sigma_mmi": 1.01,
            },
            "PGV": {
                "c1": 4.37,
                "c2": 1.32,
                "c3": 3.54,
                "c4": 3.03,
                "t1": 0.48,
                "sigma_mmi": 0.8,
            },
            "SA(0.3)": {
                "c1": 2.40,
                "c2": 1.36,
                "c3": -1.83,
                "c4": 3.56,
                "t1": 1.92,
                "sigma_mmi": 0.88,
            },
            "SA(1.0)": {
                "c1": 3.23,
                "c2": 1.18,
                "c3": 0.57,
                "c4": 2.95,
                "t1": 1.50,
                "sigma_mmi": 0.84,
            },
            "SA(3.0)": {
                "c1": 3.72,
                "c2": 1.29,
                "c3": 1.99,
                "c4": 3.0,
                "t1": 1.00,
                "sigma_mmi": 0.86,
            },
        }


class AtkinsonKaka2007MRDependent(AtkinsonKaka2007):
    """
    Implements the Ground Motion Intensity Conversion Equation (GMICE) of Atkinson & Kaka
    (2007) with magnitude and distance predictors

    Atkinson GA and Kaka SI (2007), "Relationships between Felt Intensity and Instrumental
    Ground Motion in the Central United States and California", Bulletin of the Seismological
    Society of America, 97(2), pp 497 - 510, doi: 10.1785/0120060154
    """

    @property
    def REQUIRES_DISTANCES(self) -> Set:
        """Requires rupture distance"""
        return {
            "rrup",
        }

    @property
    def REQUIRES_RUPTURE_PARAMETERS(self) -> Set:
        """Requires magnitude"""
        return {
            "mag",
        }

    # Define the order of preference of the IMTs for conversion
    ORDER_OF_PREFERENCE = ["SA(3.0)", "SA(1.0)", "PGV", "SA(0.3)", "PGA"]

    def get_mean_intensity_and_stddev(
        self, imt: str, log_gmvs: np.ndarray, gmv_total_stddev: np.ndarray, clip: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean intensity and standard deviation by adding on the magitude and
        distance factors to the mean intensity provided by the magnitude- and distance-
        independent form of the model.

        """
        if imt not in self.COEFFS:
            raise ValueError(
                "Conversion to intensity for IMT %s unsupported for %s"
                % (imt, self.__class__.__name__)
            )
        C = self.COEFFS[imt]
        mmi, stddev = super().get_mean_intensity_and_stddev(
            imt, log_gmvs, gmv_total_stddev, clip=False
        )
        mmi += C["c5"] + C["c6"] * self.rctx.mag + C["c7"] * np.log10(self.dctx.rrup)
        if clip:
            mmi = np.clip(mmi, 1.0, 10.0)
        return mmi, stddev

    @property
    def COEFFS(self) -> Dict:
        """Coefficients of the GMICE as defined in Tables 4 and 5"""
        return {
            "PGA": {
                "c1": 2.65,
                "c2": 1.39,
                "c3": -1.91,
                "c4": 4.09,
                "t1": 1.69,
                "c5": -1.96,
                "c6": 0.02,
                "c7": 0.98,
                "sigma_mmi": 0.89,
            },
            "PGV": {
                "c1": 4.37,
                "c2": 1.32,
                "c3": 3.54,
                "c4": 3.03,
                "t1": 0.48,
                "c5": 0.47,
                "c6": -0.19,
                "c7": 0.26,
                "sigma_mmi": 0.76,
            },
            "SA(0.3)": {
                "c1": 2.40,
                "c2": 1.36,
                "c3": -1.83,
                "c4": 3.56,
                "t1": 1.92,
                "c5": -0.11,
                "c6": -0.20,
                "c7": 0.64,
                "sigma_mmi": 0.79,
            },
            "SA(1.0)": {
                "c1": 3.23,
                "c2": 1.18,
                "c3": 0.57,
                "c4": 2.95,
                "t1": 1.50,
                "c5": 1.92,
                "c6": -0.39,
                "c7": 0.04,
                "sigma_mmi": 0.73,
            },
            "SA(3.0)": {
                "c1": 3.72,
                "c2": 1.29,
                "c3": 1.99,
                "c4": 3.0,
                "t1": 1.00,
                "c5": 2.24,
                "c6": -0.33,
                "c7": -0.31,
                "sigma_mmi": 0.72,
            },
        }
