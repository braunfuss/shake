"""
Implements magnitude scaling relations for calculating the finite rupture dimensions
"""
import abc
from typing import Tuple
from math import radians, sin, exp, sqrt, pi, log
from scipy.stats import norm, chi2


class BaseScalingRelation(metaclass=abc.ABCMeta):
    """
    Abstract base class representing an implementation of a magnitude scaling relation

    """

    @abc.abstractmethod
    def get_rupture_dimensions(
        self,
        magnitude: float,
        rake: float = 0.0,
        dip: float = 90.0,
        aspect: float = 1.0,
        thickness: float = 20.0,
        epsilon: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Args:
            magnitude:
                Earthquake magnitude
            rake:
                Rake of the earthquake rupture in degrees (in the range -180 to 180)
            dip:
                Dip of the earthquake rupture in degrees (in the range 0.0 to 90.0)
            aspect:
                Aspect ratio of the rupture
            thickness:
                Seismogenic thickness of the crust (in km)

        Returns:
            Dimensions of the rupture as a tuple of rupture area (in km^2), rupture length
            (in km) and downdip rupture width (in km)
        """


class PEERScalingRelation(BaseScalingRelation):
    """
    A simple magnitude-area scaling relation to use as a placeholder until
    updated by the synthetic rupture generator
    """

    def get_rupture_dimensions(
        self,
        magnitude: float,
        rake: float = 0.0,
        dip: float = 90.0,
        aspect: float = 1.0,
        thickness: float = 20.0,
        epsilon: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Returns the area (km^2), length (km) and width (km) of the rupture using the PEER
        Magnitude-Area scaling relation and constrained by the crustal thickness
        """
        # PEER defined magnitude-area scaling relation returns area in km^2
        area = self.get_area(magnitude)

        width = sqrt(area / aspect)
        # dz is the vertical dimension of the rupture (in km)
        dz = width * sin(radians(dip))
        if dz > thickness:
            # Rupture is wider than the crustal thickness, so rescale it to the
            # maximum width and break the aspect ratio
            width = thickness / sin(radians(dip))
        length = area / width
        return area, length, width

    @staticmethod
    def get_area(magnitude: float) -> float:
        """
        Returns the area from the PEER magnitude-area scaling relation

        A = 10.0 ^ (magnitude - 4.0)
        """
        return 10.0 ** (magnitude - 4.0)


class StrasserEtAl2010Interface(PEERScalingRelation):
    """
    Implements the magnitude-area scaling relation for subduction interface earthquakes
    from Strasser et al. (2010)

    Strasser FO, Arango MC, Bommer, JJ (2010) "Scaling of the Source Dimensions of
    Interface and Intraslab Subduction-zone Earthquakes with Moment Magnitude", Seismological
    Research Letters, 81: 941 - 950, doi:10.1785/gssrl.81.6.941
    """

    @staticmethod
    def get_area(magnitude: float) -> float:
        """
        Returns the median area from the magnitude-area scaling relation
        """
        return 10.0 ** (-3.476 + 0.952 * magnitude)


class StrasserEtAl2010Inslab(PEERScalingRelation):
    """
    Implements the magnitude-area scaling relation for subduction in-slab earthquakes
    from Strasser et al. (2020)

    Strasser FO, Arango MC, Bommer, JJ (2010) "Scaling of the Source Dimensions of
    Interface and Intraslab Subduction-zone Earthquakes with Moment Magnitude", Seismological
    Research Letters, 81: 941 - 950, doi:10.1785/gssrl.81.6.941
    """

    @staticmethod
    def get_area(magnitude: float) -> float:
        """
        Returns the median area from the magnitude-area scaling relation
        """
        return 10.0 ** (-3.225 + 0.89 * magnitude)


class Stafford2014(BaseScalingRelation):
    """
    Implements the hazard-consistent magnitude scaling relation described by Stafford (2014)

    Stafford PJ (2014) "Source-Scaling Relationships for the Simulation of Rupture
    Geometry within Probabilistic Seismic-Hazard Analysis", Bulletin of the Seismological
    Society of America, 104(4): 1620 - 1635, doi: 10.1785/012013024
    """

    # Model coefficients for the specific style of faulting
    COEFFS = {
        # Coefficients from Table 1
        1: {
            "U": {"a0": -27.4922, "a1": 4.6656, "a2": -0.2033},
            "SS": {"a0": -30.8395, "a1": 5.4184, "a2": -0.3044},
            "N": {"a0": -36.9770, "a1": 6.3070, "a2": -0.1696},
            "R": {"a0": -35.8239, "a1": 5.0680, "a2": -0.0457},
        },
        # Coefficients from Table 2
        2: {
            "U": {"b0": -2.300, "b1": 0.7167, "sigma": 0.2337},
            "SS": {"b0": -2.300, "b1": 0.7167, "sigma": 0.2337},
            "N": {"b0": -4.1055, "b1": 1.0370, "sigma": 0.2509},
            "R": {"b0": -3.8300, "b1": 0.9982, "sigma": 0.2285},
        },
        # Coefficients from Table 3
        3: {
            "U": {"gamma0": -9.3137, "sigma": 0.3138, "rho": 0.3104},
            "SS": {"gamma0": -9.3137, "sigma": 0.3138, "rho": 0.3104},
            "N": {"gamma0": -9.2483, "sigma": 0.3454, "rho": 0.4336},
            "R": {"gamma0": -9.2749, "sigma": 0.2534, "rho": 0.1376},
        },
        # Coefficients from Table 4
        4: {"U": 0.7574, "SS": 0.7574, "N": 0.8490, "R": 0.7496},
    }

    def get_rupture_dimensions(
        self,
        magnitude: float,
        rake: float = 0.0,
        dip: float = 90.0,
        aspect: float = 1.0,
        thickness: float = 20.0,
        epsilon: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Gets the rupture dimensions from for the given magnitude subject to the physical
        constaints.

        Args:
            epsilon:
                Number of standard deviations above or below the median to determine the
                rupture width
        """
        sof = self._get_sof(rake)
        # Get mean and standard deviation of rupture width
        mu_rw, sigma_rw, max_width, p_i = self.get_rupture_width(magnitude, dip, sof, thickness)
        # Get mean and standard deviation of rupture area
        mu_ra, sigma_ra = self.get_rupture_area(magnitude, sof, max_width, sigma_rw)
        # Apply censoring of the rupture width by the seismogenic thickness
        F_rw_max_norm = norm.cdf(log(max_width), loc=mu_rw, scale=sigma_rw)
        ncdf_epsilon = norm.cdf(epsilon)
        target = ncdf_epsilon / F_rw_max_norm
        if target > 1:
            target = ncdf_epsilon
        epsilon_rw = norm.ppf(target)
        # Retrieve the rupture width and the rupture area conditioned on the rupture width
        width = exp(mu_rw + epsilon_rw * sigma_rw)
        if width > max_width:
            width = max_width
        epsilon_ra = self.COEFFS[4][sof] * epsilon
        area = exp(mu_ra + epsilon_ra * sigma_ra)
        length = area / width
        return area, length, width

    @staticmethod
    def _get_sof(rake: float) -> str:
        """
        Determine the style of faulting: strike slip (SS), reverse (R), normal (N) or
        unknown (U)

        Args:
            rake: Rake of the rupture (in degrees from -180 to 180)

        Returns:
            Style of faulting
        """
        if rake is None:
            return "U"

        if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return "SS"
        elif rake > 0:
            # thrust/reverse
            return "R"
        else:
            # normal
            return "N"

    def get_rupture_width(
        self, magnitude: float, dip: float, sof: str, thickness: float
    ) -> Tuple[float, float, float, float]:
        """
        Returns the parameters needed to define the censored rupture width
        distribution defined in equations 8 to 14
        Args:
            magnitude: magnitude of earthquake
            dip: Dip of earthquake in degrees from 0 to 90.0
            sof: Style-of-faulting class (as string)
            thickness: Seismogenic thickness
        """
        # Gets the probability of a full width rupture
        rw_max = thickness / sin(radians(dip))
        z_i = (
            self.COEFFS[1][sof]["a0"]
            + self.COEFFS[1][sof]["a1"] * magnitude
            + self.COEFFS[1][sof]["a2"] * rw_max
        )
        # Probability of rupturing full seismogenic thickness from logistic regression
        p_i = 1.0 / (1.0 + exp(-z_i))
        # Median width from equation 16
        ln_rw = self.COEFFS[2][sof]["b0"] + self.COEFFS[2][sof]["b1"] * magnitude
        # Equations 18 - 20
        phi_rw = (log(rw_max) - ln_rw) / self.COEFFS[2][sof]["sigma"]
        phi_rw_ncdf = norm.cdf(phi_rw)
        ln_rw_trunc = ln_rw - self.COEFFS[2][sof]["sigma"] * (norm.pdf(phi_rw) / phi_rw_ncdf)
        mean_rw = p_i * log(rw_max) + (1.0 - p_i) * ln_rw_trunc
        # Equations 21 - 22
        stddev_rw = self._get_rupture_width_sigma(
            self.COEFFS[2][sof]["sigma"], phi_rw, phi_rw_ncdf, p_i
        )
        return mean_rw, stddev_rw, rw_max, p_i

    @staticmethod
    def _get_rupture_width_sigma(
        sigma: float, phi_rw: float, phi_rw_ncdf: float, p_i: float
    ) -> float:
        """
        Returns the variabiliy in the rupture width described by equations 21 and 22
        """
        denom = sqrt(2.0 * pi) * phi_rw_ncdf
        if phi_rw_ncdf >= 0.0:
            elem1 = sqrt(pi / 2.0) * (1.0 + chi2.cdf(phi_rw, 3))
        else:
            elem1 = sqrt(pi / 2.0) * (1.0 - chi2.cdf(phi_rw, 3))
        elem2 = exp(-(phi_rw ** 2)) / denom
        sigma_truncated = sqrt(((sigma ** 2.0) / denom) * (elem1 - elem2))
        return (1.0 - p_i) * sigma_truncated

    def get_rupture_area(
        self, magnitude: float, sof: str, rw_max: float, sigma_lnrw: float
    ) -> Tuple[float, float]:
        """
        Returns the rupture area conditioned upon the maximum rupture width and style of
        faulting

        Args:
            magnitude: magnitude of earthquake
            sof: Style-of-faulting class (as string)
            rw_max: Maximum rupture width (in km)
            sigma_lnrw: Standard deviation of the logarithmic rupture width

        Returns:
            ln_ra: Natural logarithm of rupture area
            sigma_ra: Standard deviation of the natural logarithm of rupture area
        """
        mw_crit = (log(rw_max) - self.COEFFS[2][sof]["b0"]) / self.COEFFS[2][sof]["b1"]
        ln_ra = self.COEFFS[3][sof]["gamma0"] + log(10.0) * magnitude
        if magnitude > mw_crit:
            # Equation 23
            ln_ra -= (log(10.0) / 4.0) * (magnitude - mw_crit)
        # Get the sigma log rupture area (equation 28)
        sigma = self.COEFFS[3][sof]["sigma"]
        sigma_ra = sqrt(
            (sigma ** 2.0)
            + (sigma_lnrw ** 2.0)
            + (2.0 * self.COEFFS[3][sof]["rho"] * sigma_lnrw * sigma)
        )
        return ln_ra, sigma_ra
