"""
Generates a set of synthetic ruptures based on a point source configuration
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import truncnorm
from openquake.hazardlib.geo import Point, Mesh
from openquake.hazardlib.geo.surface.base import BaseSurface
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture
from shake.oq_shakemap.site_model import SiteModel
from shake.oq_shakemap.earthquake import Earthquake


# The hypocentre distributions are defined from the Next Generation Attenuation
# (NGA) West 2 rupture database
NGAWEST_HYPO_DISTRIBUTIONS = {
    "All": {
        "Mean": np.array([0.46851357, 0.67460261]),
        "Sigma": np.array([0.20691535, 0.22040932]),
        "COV": np.array([[0.04293594, 0.00103987], [0.00103987, 0.04871868]]),
        "DipRange": [30.0, 90.0],
    },
    "SS": {
        "Mean": np.array([0.48545206, 0.64942746]),
        "Sigma": np.array([0.22415657, 0.21677068]),
        "COV": np.array([[0.05064814, -0.00603003], [-0.00603003, 0.04736544]]),
        "DipRange": [80.0, 90.0],
    },
    "R": {
        "Mean": np.array([0.4674859, 0.58483914]),
        "Sigma": np.array([0.16275562, 0.22017015]),
        "COV": np.array([[0.02673021, 0.0113362], [0.0113362, 0.04891558]]),
        "DipRange": [25.0, 50.0],
    },
    "N": {
        "Mean": np.array([0.50887254, 0.82404]),
        "Sigma": np.array([0.22416128, 0.13647917]),
        "COV": np.array([[0.05085368, -0.00332741], [-0.00332741, 0.01885098]]),
        "DipRange": [50.0, 80.0],
    },
}


# In case other definitions (thrust fault, undefined etc.) are used
NGAWEST_HYPO_DISTRIBUTIONS["TF"] = NGAWEST_HYPO_DISTRIBUTIONS["R"]
NGAWEST_HYPO_DISTRIBUTIONS["U"] = NGAWEST_HYPO_DISTRIBUTIONS["All"]


class FiniteRuptureSampler(object):
    """
    Module to produce a set of finite rupture samples for an event
    Process:
    1. From scaling relation and seismogenic thicknesses sample rupture
       parameters (area, length, width)
    2. Sample hypocentre location from multivariate Gaussian distribution
    3. a. If mechanism is known, use strike and dip
       b. If mechanism is not known sample strike and dip from distribution
    4. For each sample calculate Rjb, Rrup, Rx and Ry0
    5. For each distance calculate the median value
    6. For each rupture define a penalty function

    P = sum_{i=rjb, rrup, ry0, rx} (Ri - median(Ri)) ** 2.
    Take the rupture with the lowest penalty function!
    """

    def get_finite_rupture(
        self,
        nsamples: int,
        earthquake: Earthquake,
        site_model: Optional[SiteModel] = None,
        rdim: float = 0.0,
        weights: list = [0.25, 0.25, 0.25, 0.25],
        maximum_site_distance: float = 200.0,
        site_spacing: float = 0.05,
    ):
        """
        Retrieves the finite rupture that produces the distances closest to the median
        distance

        Args:
            nsamples: Number of rupture samples to use
            earthquake: Earthquake object as instance of
                        :class:`shaky.earthquake.Earthquake`
            site_model: If desired, can define the target sites to be used for the distance
                        calculation, as instance of :class:`shaky.site_model.SiteModel`
            rdim: The penalty functions can be weighted according to the source-to-site
                  distance with the weighting determined via w = 1 / (distance ^ R). rdim
                  sets the value of R
            weights: Can adjust the weight assigned to the different distance metrics. Defines
                     the list of weights for the four distance metrics: Rjb, Rrup, Rx and Ry0
            maximum_site_distance: In the case that the target sites need to be built from the
                                   bounding box, this defines the maximum distances of the
                                   target sites to define the bounding box
            site_spacing: In the case that the target sites need to be built from the
                          bounding box, this defines the site spacing

        Returns:
            central_rupture: The rupture corresponding to the `most central` as an instance of
                :class:`openquake.hazardlib.source.rupture.ParametricProbabilisticRupture`
            distances: Dictionary containing all of the possible distances calculated from
                       the rupture surface
        """
        # Sample the rupture surfaces
        (
            rupture_surfaces,
            strikes,
            dips,
            rakes,
            hypo_locs,
        ) = self.sample_rupture_surfaces(nsamples, earthquake)

        if site_model:
            target_lons = site_model["lon"]
            target_lats = site_model["lat"]
        else:
            # If no specific site model is defined then use a bounding box within 200 km of
            # the hypocentre spaced every 0.05 degrees - longer distances are less likely
            # to be significantly influenced by uncertainties in the rupture plane
            llon, llat, ulon, ulat = earthquake.get_maximum_distance_bbox(maximum_site_distance)
            if ulon < llon:
                # Happens when crosses the antimeridian
                target_lons, target_lats = np.meshgrid(
                    np.arange(llon, ulon + 360.0 + site_spacing, site_spacing),
                    np.arange(llat, ulat + site_spacing, site_spacing),
                )
                target_lons[target_lons > 180.0] -= 360.0
            else:
                target_lons, target_lats = np.meshgrid(
                    np.arange(llon, ulon + site_spacing, site_spacing),
                    np.arange(llat, ulat + site_spacing, site_spacing),
                )
                target_lons[target_lons > 180.0] -= 360.0
            target_lons = target_lons.flatten()
            target_lats = target_lats.flatten()

        # Calculate the distances
        distances, rhypo, repi = self.get_distances(
            earthquake, rupture_surfaces, target_lons, target_lats
        )
        # Get the most central rupture closest to the median finite rupture distance
        # from all finite ruptures
        central_surface, central_distances, min_loc = self.get_central_rupture(
            rupture_surfaces, distances, rhypo, rdim, weights
        )

        # Return the parametric probabilitic rupture and the corrsponding distance
        central_rupture = ParametricProbabilisticRupture(
            earthquake.mag,
            rakes[min_loc],
            None,  # No tectonic region type
            earthquake.hypocenter,
            central_surface,
            1.0,
            None,
        )

        return central_rupture, {
            "lon": target_lons,
            "lat": target_lats,
            "rjb": central_distances[:, 0],
            "rrup": central_distances[:, 1],
            "rx": central_distances[:, 2],
            "ry0": central_distances[:, 3],
            "rhypo": rhypo,
            "repi": repi,
        }

    def sample_rupture_surfaces(
        self, nsamples: int, earthquake: Earthquake
    ) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Mechanisms should be a list of Nodal plane objects

        Args:
            nsamples: Number of samples to take
            earthquake: Earthquake object as instance of
                        :class:`shaky.earthquake.Earthquake`
        Returns:
            surfaces: List of rupture surfaces as instances of the :class:
                      `openquake.hazardlib.geo.PlanarSurface
            strikes: Vector of strike values (in decimal degrees in the range 0 to 360)
            dips: Vector of dip values (in decimal degrees in the range 0 to 90)
            rakes: Vector of rake values (in decimal degrees in the range -180 to 180)
            hypo_positions: 2D array of hypocentre positions (as along-strike and down-dip)
                            positions
        """
        strikes, dips, rakes = earthquake.mechanism.sample(nsamples)
        dimensions = (
            (earthquake.area is not None)
            and (earthquake.length is not None)
            and (earthquake.width is not None)
        )
        if dimensions:
            areas = earthquake.area * np.ones(nsamples)
            lengths = earthquake.length * np.ones(nsamples)
            widths = earthquake.width * np.ones(nsamples)
        else:
            # Get the physical rupture dimensions by sampling from the magnitude scaling
            # relation
            areas, lengths, widths = self.sample_rupture_dimensions(
                nsamples, earthquake, dips, rakes
            )

        # Get hypocentres
        hypo_positions = self.sample_hypocenter_position(nsamples, self.get_sofs(rakes))

        # Build the ruptures
        surfaces = []
        for strike, dip, length, width, hypo_pos in zip(
            strikes, dips, lengths, widths, hypo_positions
        ):
            surfaces.append(
                earthquake.build_planar_surface(
                    earthquake.hypocenter,
                    strike,
                    dip,
                    length,
                    width,
                    earthquake.lsd,
                    earthquake.usd,
                    hypo_pos,
                )
            )
        return surfaces, strikes, dips, rakes, hypo_positions

    @staticmethod
    def get_sofs(rakes: np.ndarray) -> List:
        """
        Returns the style of faulting as a string given an input list of rakes
        """
        sofs = []
        for rake in rakes:
            if (rake > 45.0) and (rake <= 135.0):
                sofs.append("R")
            elif (rake > -135.0) and (rake <= -45.0):
                sofs.append("N")
            else:
                sofs.append("SS")
        return sofs

    @staticmethod
    def sample_rupture_dimensions(
        nsamples: int, earthquake: Earthquake, dips: np.ndarray, rakes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples the rupture dimension from the magnitude scaling relation accounting for
        uncertainty where it is defined

        Args:
            nsamples: Number of samples to take
            earthquake: Earthquake object as instance of
                        :class:`shaky.earthquake.Earthquake`
            dips: Vector of dip values
            rakes: Vector of rake values
        Returns:
            areas: Vector of sampled areas (in km ^ 2)
            lengths: Vector of sampled lengths (in km)
            widths: Vector of sampled widths (in km)
        """
        # Sample epsilon values from a truncated normal distribution between
        # -3 and 3 standard deviations
        msr_epsilons = truncnorm.rvs(-3.0, 3.0, loc=0.0, scale=1.0, size=nsamples)
        # Generate rupture dimensions
        lengths = np.empty(nsamples)
        widths = np.empty(nsamples)
        areas = np.empty(nsamples)
        thickness = earthquake.lsd - earthquake.usd
        for i, (dip, rake, epsilon) in enumerate(zip(dips, rakes, msr_epsilons)):
            (
                areas[i],
                lengths[i],
                widths[i],
            ) = earthquake.mag_scale_rel.get_rupture_dimensions(
                earthquake.mag,
                rake,
                dip,
                aspect=earthquake.aspect,
                thickness=thickness,
                epsilon=epsilon,
            )
        return areas, lengths, widths

    @staticmethod
    def sample_hypocenter_position(nsamples: int, sofs: List) -> np.ndarray:
        """
        Samples the hypocentral position from the pre-defined NGA West 2 hypocentre
        distributions

        Note here that the hypocentre position distributions are multivariate Gaussian, yet
        they need to be truncated within the range 0 to 1. Sampling from a truncated
        multivariate Gaussian distribution is not a trivial problem and requires a Markov Chain
        Monte Carlo sampling approach that is too complex for application here. Instead,
        invalid hypocentre positions are re-sampled repeatedly until all positions are valid

        Args:
            nsamples: Number of samples to take
            sofs: List of styles-of-faulting
        Returns:
            2D array of hypocentre position samples (along strike and down dip)
        """
        # Count the unique styles of faulting present in the sampling set
        sof_indices = {}
        sof_counter = {}
        sof_array = np.array(sofs)
        for key in NGAWEST_HYPO_DISTRIBUTIONS:
            sof_indices[key] = np.where(sof_array == key)[0]
            sof_counter[key] = len(sof_indices[key])

        hypo_pos = np.zeros([nsamples, 2])
        for key in sof_counter:
            if not sof_counter[key]:
                # Style-of-faulting not present in sample set
                continue
            mean = NGAWEST_HYPO_DISTRIBUTIONS[key]["Mean"]
            covariance = NGAWEST_HYPO_DISTRIBUTIONS[key]["COV"]
            # Draw N samples from a multivariate distribution, where N
            # is the corresponding number of samples with the required style of
            # faulting
            pos = np.random.multivariate_normal(mean, covariance, sof_counter[key])
            # Are all positions valid?
            all_valid = np.logical_and(
                np.logical_and(pos[:, 0] >= 0.05, pos[:, 0] <= 0.95),
                np.logical_and(pos[:, 1] >= 0.05, pos[:, 1] <= 0.95),
            )

            while np.sum(all_valid) < sof_counter[key]:
                # Some positions are invalid - re-sample these
                idx = np.logical_not(all_valid)
                pos[idx] = np.random.multivariate_normal(mean, covariance, sum(idx))
                all_valid = np.logical_and(
                    np.logical_and(pos[:, 0] >= 0.05, pos[:, 0] <= 0.95),
                    np.logical_and(pos[:, 1] >= 0.05, pos[:, 1] <= 0.95),
                )

            hypo_pos[sof_indices[key], :] = pos
        return hypo_pos

    @staticmethod
    def get_distances(
        earthquake: Earthquake, surfaces: List, lons: np.ndarray, lats: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the source to site distances for each rupture surface in
        the set of rupture surfaces

        Args:
            earthquake: Earthquake object as instance of
                        :class:`shaky.earthquake.Earthquake`
            lons: Vector of target longitudes for distance calculation
            lats: Vector of target latitudes for distance calculation
        Returns:
            distances: 2D array of finite fault rupture distances in km (Rjb, Rrup, Rx, Ry0)
            rhypo: hypocentral distance in km
            repi: epicentral distance in km
        """
        nsites = len(lons)
        mesh = Mesh(lons, lats)
        distances = np.zeros([len(surfaces), nsites, 4])
        for i, surf in enumerate(surfaces):
            distances[i, :, 0] = surf.get_joyner_boore_distance(mesh)
            distances[i, :, 1] = surf.get_min_distance(mesh)
            distances[i, :, 2] = surf.get_rx_distance(mesh)
            distances[i, :, 3] = surf.get_ry0_distance(mesh)
        # Include hypocentral and epicentral distance
        rhypo = Point(earthquake.lon, earthquake.lat, earthquake.depth).distance_to_mesh(mesh)
        repi = np.sqrt(rhypo ** 2.0 - earthquake.depth ** 2.0)
        return distances, rhypo, repi

    @staticmethod
    def get_central_rupture(
        rupture_surfaces: List,
        distances: np.ndarray,
        hypocentral_distances: np.ndarray,
        rdim: float = 0.0,
        weights: List = [0.25, 0.25, 0.25, 0.25],
    ) -> Tuple[BaseSurface, np.ndarray, int]:
        """
        Returns the rupture closest to the median distances
        """
        nsites = distances.shape[1]
        # If desired, it is possible to weight the distribution of distances
        # such that the weight decays with distance from the centroid, with the
        # exponent of the decay factor described by rdim (if 0 then no weighting is applied).
        site_weights = 1.0 / (hypocentral_distances ** rdim)
        site_weights = site_weights / np.sum(site_weights)

        # For each of the four different distance types retrieve the median source-to-site
        # distance for each site
        median_distances = np.zeros([4, nsites])
        for i in range(4):
            median_distances[i, :] = np.percentile(distances[:, :, i], 50, axis=0)
        # For each of the rupture surfaces determine the degree of divergence from the
        # median distance and assign a penalty that is proportional to the divergence
        penalty_function = np.zeros(distances.shape[0])
        for i in range(distances.shape[2]):
            site_penalty = np.zeros(distances.shape[0])
            for k in range(distances.shape[1]):
                site_penalty += site_weights[k] * (
                    np.sqrt((distances[:, k, i] - median_distances[i, k]) ** 2.0)
                )
            penalty_function += weights[i] * site_penalty
        # Determine which rupture gives the minimal divergence from the median and return
        # that rupture and corresponding distances
        min_loc = np.argmin(penalty_function)
        return rupture_surfaces[min_loc], distances[min_loc, :, :], min_loc
