"""
Class to manage the distribution of rupture mechanisms
"""
from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Optional, Dict, Tuple, List
from openquake.hazardlib.geo import NodalPlane
from openquake.hazardlib.pmf import PMF
from shake.oq_shakemap import valid


class DIP_RANGES(Enum):
    """
    For the "global" default rupture mechanism distribution the possible ranges
    of dip values depend on the style-of-faulting. The possible range of values
    for each style-of-faulting is provided and they are assumed to be uniformly
    distributed.
    """

    R = (20.0, 45.0)
    SS = (75.0, 91.0)
    N = (45.0, 75.0)


class RuptureMechanism(object):
    """
    General class to handle the mechanism properties of the rupture, which include the
    strike, dip and rake. The mechanism properties largely constrol the dimensions of the 3D
    rupture plane that is needed for calculating the finite-source to site distances (e.g.
    Joyner-Boore distance, Rupture Distance, Rx, Ry0 etc.)

    for the majority of reported earthquakes a single rupture mechanism is not known unless
    accompanied by a 3D finite fault rupture model. In the absence of any information other
    than the hypocentre, a "global" distribution of strike, dip and rake values is assumed.
    In the case that a focal mechanism is available then the RuptureMechanism is described by
    two equiprobable planes.

    Attributes:
        mechanism: Distribution of rupture mechanisms (probability mass function) as instance
                   of :class:`openquake.hazardlib.pmf.PMF`

    """

    def __init__(self, mechanism: Optional[PMF] = None):
        if mechanism:
            assert isinstance(mechanism, PMF)
            self.mechanism = mechanism
        else:
            # Build the default distribution
            self.mechanism = self.build_mechanism_distribution()

    def __iter__(self):
        # Iterate over the mechanisms in the data set yielding the probability
        # of the nodal plane and the nodal plane itself
        for prob, nodal_plane in self.mechanism.data:
            yield prob, nodal_plane

    def __len__(self):
        # Lenth corresponds to the number of mechanisms in the distribution
        return len(self.mechanism.data)

    def sample(self, nsamples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples the focal mechanism distribution to return "nsamples" strike,
        dip and rake values

        Args:
            nsamples: Number of sample
        Returns:
            Tuple of vectors of samples strikes, dips and rakes
        """
        assert nsamples > 0
        sample_planes = self.mechanism.sample(nsamples)
        strikes = np.empty(nsamples, dtype=float)
        dips = np.empty(nsamples, dtype=float)
        rakes = np.empty(nsamples, dtype=float)
        for i, sample_plane in enumerate(sample_planes):
            strikes[i] = sample_plane.strike
            dips[i] = sample_plane.dip
            rakes[i] = sample_plane.rake
        return strikes, dips, rakes

    @classmethod
    def from_strike_dip_rake(
        cls,
        strike: Optional[float] = None,
        dip: Optional[float] = None,
        rake: Optional[float] = None,
    ) -> RuptureMechanism:
        """
        Constructs the rupture mechanism distribution from a simple strike, dip
        and rake combination, permitting None values if undefined

        Args:
            strike: Strike of fault (in decimal degrees between 0 and 360)
            dip: Dip of fault (in decimal degrees between 0 and 90)
            rake: Rake of fault (in decimal degrees between -180 and 180)
        """
        return cls(
            cls.build_mechanism_distribution(
                valid.strike(strike), valid.dip(dip), valid.rake(rake)
            )
        )

    @classmethod
    def from_focal_mechanism(cls, focal_mechanism: Dict) -> RuptureMechanism:
        """
        Constructs the focal mechanism from an evenly pair of nodal planes such as that
        of a focal mechanism

        Args:
            focal_mechanism: Dictionary containing two :class:
            `openquake.hazardlib.geo.NodalPlane` objects, each labeled as "nodal_plane_1"
            and "nodal_plane_2"
        """
        return cls(
            PMF(
                [
                    (0.5, focal_mechanism["nodal_plane_1"]),
                    (0.5, focal_mechanism["nodal_plane_2"]),
                ]
            )
        )

    @classmethod
    def from_nodal_planes(cls, nodal_planes: List, probabilities: List) -> RuptureMechanism:
        """
        Constructs the rupture mechanism distribution from a list of nodal planes and their
        associated probabilities

        Args:
            nodal_planes: Set of nodal planes as a list of dictionaries, eac containing strike,
                          dip and rake
            probabilities: List of probabilities of the nodal planes (must sum to 1)
        """
        assert len(nodal_planes) == len(
            probabilities
        ), "Number of nodal planes not equal to number of probabilities"
        assert np.isclose(
            sum(probabilities), 1.0
        ), "Probabilities do not sum to 1.0 (sum = %.6f)" % sum(probabilities)
        mechanism_distribution = []
        for prob, npl in zip(probabilities, nodal_planes):
            mechanism = valid.mechanism(npl["strike"], npl["dip"], npl["rake"])
            mechanism_distribution.append((prob, mechanism))
        return cls(PMF(mechanism_distribution))

    @staticmethod
    def build_mechanism_distribution(
        strike: Optional[float] = None,
        dip: Optional[float] = None,
        rake: Optional[float] = None,
    ) -> PMF:
        """
        Builds a mechanism distribution from a partial (or complete) characterisation of the
        rupture mechanism

        Args:
            strike: Strike of fault in decimal degrees between 0 and 360, or None
            dip: Dip of fault in decimal degrees between 0 and 90, or None
            rake: Rake of fault in decimal degrees between -180 and 180, or None

        Returns:
            Probability mass function of the mechanism distribution as an instance of the
            :class:`openquake.hazardlib.pmf.PMF`
        """
        if rake is None:
            # If rake is not defined then describe a complete range of rakes every 15 degrees
            # from -165.0 to 180.0
            rakes = np.arange(-165.0, 181.0, 15.0)
            weight_r = (1.0 / len(rakes)) * np.ones(len(rakes))
        else:
            rakes = [rake]
            weight_r = [1.0]

        if strike is None:
            # If strike is not defined then describe a complete range of strikes every 15
            # degrees between 0 and 360 (not included)
            strikes = np.arange(0.0, 359.0, 15.0)
            weight_s = (1.0 / len(strikes)) * np.ones(len(strikes))
        else:
            strikes = [strike]
            weight_s = [1.0]

        mechanisms = []
        for wght1, rake_i in zip(weight_r, rakes):
            if dip is None:
                # If dip is undefined then sample uniformly in the range
                # appropriate to the style of faulting
                if rake_i >= 45.0 and rake_i < 135.0:
                    # Reverse
                    dip_range = DIP_RANGES.R.value
                elif rake_i < -45.0 and rake_i >= -135.0:
                    # Normal
                    dip_range = DIP_RANGES.N.value
                else:
                    # Strike-slip
                    dip_range = DIP_RANGES.SS.value
                dips = np.arange(dip_range[0], dip_range[1], 5.0)
                weight_d = (1.0 / len(dips)) * np.ones(len(dips))
            else:
                dips = [dip]
                weight_d = [1.0]
            for wght2, dip_i in zip(weight_d, dips):
                for wght3, strike_i in zip(weight_s, strikes):
                    prob = wght1 * wght2 * wght3
                    mechanisms.append((prob, NodalPlane(strike_i, dip_i, rake_i)))
        return PMF(mechanisms)
