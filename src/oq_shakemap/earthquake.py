"""
Contains the core classes to represent an earthquake
"""
from __future__ import annotations
import datetime
from typing import Tuple, Optional, List
from math import radians, sin, tan
from openquake.hazardlib.geo import Point, PlanarSurface
from openquake.hazardlib.geo.surface.base import BaseSurface
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture
from shake.oq_shakemap import valid
from shake.oq_shakemap.rupture_mechanism import RuptureMechanism
from shake.oq_shakemap.io import fetch_quakeml


class Earthquake(object):
    """
    Shakemap event class requires a minimum of an ID, longitude, latitude,
    depth and magnitude. Other attributes can be input to control the rupture
    orientation and mechanism.

    Can input a rupture geometry directly or else the rupture geometry will be
    built from the other properties input

    Attributes:
        id:
            Unique ID for the earthquake
        lon:
            Earthquake longitude in decimal degrees (-180 to 180)
        lat:
            Earthquake latitude in decimal degrees (-90, to 90)
        depth:
            Hypocentral depth (in km)
        hypocenter:
            Representation of the hypocenter as an instance of
            :class:`openquake.hazardlib.geo.Point`
        mag:
            Earthquake magnitude
        strike:
            Strike of the rupture (in degrees from north)
        dip:
            Dip of the rupture (in degrees from horizontal)
        rake:
            Rake of the earthquake rupture in decimal degrees according to the
            Aki & Richards (1980) convention (-180 to 180)
        aspect:
            Aspect ratio (length / width) of the rupture
        mag_scale_rel:
            Magnitude scaling relation as instance of the BaseScalingRelation class or None
        lsd:
            Lower seismogenic depth (in km)
        usd:
            Upper seismogenic depth (in km)
        date:
            Date of the earthquake (year, month, day) as datetime.date object
        time:
            Time of the earthquake (hour, minute, second) as datetime.time object
        surface:
            Rupture surface as instance of :class:`openquake.hazardlib.geo.BaseSurface`
        hypo_pos:
            Hypocentre location within the rupture plane as a tuple of
            (fraction of along strike length, fraction of down-dip width)
    """

    def __init__(
        self,
        id,
        lon,
        lat,
        hypo_depth,
        magnitude,
        strike=None,
        dip=None,
        rake=None,
        aspect=1.0,
        mag_scaling_relation=None,
        upper_seismogenic_depth=0.0,
        lower_seismogenic_depth=1000.0,
        surface=None,
        focal_mechanism=None,
        hypocenter_position=(0.5, 0.5),
        date=None,
        time=None,
    ):
        """
        Initialize a new Earthquake object

        Args:

        id:
            Unique ID for the earthquake
        lon:
            Earthquake longitude in decimal degrees (-180 to 180)
        lat:
            Earthquake latitude in decimal degrees (-90, to 90)
        hypo_depth:
            Hypocentral depth (in km)
        magnitude:
            Earthquake magnitude
        strike:
            Strike of the rupture (in degrees from north)
        dip:
            Dip of the rupture (in degrees from horizontal)
        rake:
            Rake of the earthquake rupture in decimal degrees according to the
            Aki & Richards (1980) convention (-180 to 180)
        aspect:
            Aspect ratio (length / width) of the rupture
        mag_scaling_relation:
            Magnitude scaling relation as instance of the BaseScalingRelation class or None.
            Defaults to :class:`shaky.magnitude_scaling_relations.PEERScalingRelation`
            when missing
        upper_seismogenic_depth:
            Upper seismogenic depth (in km)
        lower_seismogenic_depth:
            Lower seismogenic depth (in km)
        surface:
            Rupture surface as instance of :class:`openquake.hazardlib.geo.BaseSurface`, or None
        hypocenter_position:
            Hypocentre location within the rupture plane as a tuple of
            (fraction of along strike length, fraction of down-dip width)
        date:
            Date of the earthquake (year, month, day) as datetime.date object
        time:
            Time of the earthquake (hour, minute, second) as datetime.time object
        """
        self.id = id
        self.lon = valid.longitude(lon)
        self.lat = valid.latitude(lat)
        self.depth = valid.positive_float(hypo_depth, "hypocentre depth")
        self.hypocenter = Point(self.lon, self.lat, self.depth)
        self.mag = magnitude
        self.strike = valid.strike(strike)
        self.dip = valid.dip(dip)
        self.rake = valid.rake(rake)
        self.aspect = valid.positive_float(aspect, "aspect ratio")
        self.usd, self.lsd = valid.seismogenic_thickness(
            upper_seismogenic_depth, lower_seismogenic_depth
        )
        self.mag_scale_rel = valid.scaling_relation(mag_scaling_relation)
        # Date and time should be parsed as datetime.date and datetime.time
        # objects if defined, otherwise none
        assert isinstance(date, datetime.date) or date is None
        self.date = date
        assert isinstance(time, datetime.time) or time is None
        self.time = time
        assert isinstance(surface, BaseSurface) or surface is None
        self.surface = surface
        if self.surface:
            # Can calculate rupture dimensions from the surface
            self.area = self.surface.get_area()
            self.width = self.surface.get_width()
            self.length = self.area / self.width
        else:
            # Default rupture dimensions to none to none
            self.area = self.width = self.length = None
        self.hypo_pos = valid.hypocenter_position(hypocenter_position)
        self._rupture = None
        # Get a valid focal mechanism with two nodal planes
        self.focal_mechanism = valid.focal_mechanism(focal_mechanism)
        self.mechanism = self._get_mechanism()
        self.tectonic_region = None

    def _get_mechanism(self):
        """
        Defines the focal mechanism according to three different cases:
        1. A unique mechanism is defined explicitly from the strike, dip and rake
        2. A pair of equiprobable mechanisms is defined from the focal mechanism
        3. No mechanism is defined, in which case a global distribution is assumed

        Returns:
            Mechanism distribution as an instance of the
            :class:`shaky.rupture_mechanism.RuptureMechanism`
        """
        if (self.strike is not None) and (self.dip is not None) and (self.rake is not None):
            # Fixed and fully defined rupture mechanism
            return RuptureMechanism.from_strike_dip_rake(self.strike, self.dip, self.rake)
        elif self.focal_mechanism:
            # Rupture mechanism defines from nodal planes
            return RuptureMechanism.from_focal_mechanism(self.focal_mechanism)
        else:
            # Global distribution
            return RuptureMechanism()

    def __repr__(self):
        # Returns a summary string of the event, with datetime if specified
        if self.date:
            if self.time:
                datetime_string = str(datetime.datetime.combine(self.date, self.time))
            else:
                datetime_string = str(self.date)
            return "{:s} {:s} ({:.5f}E, {:.5f}N, {:.2f} km) M {:.2f}".format(
                self.id, datetime_string, self.lon, self.lat, self.depth, self.mag
            )
        else:
            return "{:s} ({:.5f}E, {:.5f}N, {:.2f} km) M {:.2f}".format(
                self.id, self.lon, self.lat, self.depth, self.mag
            )

    @classmethod
    def from_quakeml(cls, path: str) -> Earthquake:
        """
        Creates the Earthquake object from an xml file or geofon event ID

        Args:
            path: Path to QuakeML (xml) file containing GEOFON event information, or GEOFON
                  event ID (to retrieve the information from the GEOFON FDSN web service)
        """
        event = fetch_quakeml(path)
        if not event:
            raise IOError(
                "Incorrect or insufficient information to create the Earthquake "
                "object found in %s" % path
            )
        # Time is not necessarily in ISO format, so fix this for use with datetime object
        hh, mm, ss = event["time"].split(":")
        ss = "%09.6f" % float(ss.replace("Z", ""))
        event_time = ":".join([hh, mm, ss])
        d_t = datetime.datetime.fromisoformat(" ".join([event["date"], event_time]))
        # If the event has a focal mechanism then parse this into the correct format
        if event["focalmechanism"]:
            focal_mechanism = {
                "nodal_plane_1": event["focalmechanism"][0],
                "nodal_plane_2": event["focalmechanism"][1],
            }
        else:
            focal_mechanism = None
        if event["id"].endswith(".xml"):
            event["id"] = event["id"].replace(".xml", "")
        return cls(
            event["id"],
            event["origin"]["longitude"],
            event["origin"]["latitude"],
            event["origin"]["depth"],
            event["magnitude"],
            focal_mechanism=focal_mechanism,
            date=d_t.date(),
            time=d_t.time(),
        )

    def get_maximum_distance_bbox(self, max_distance: Optional[float] = None) -> List:
        """
        Defines a bounding box around the event up to a maximum distance.

        Args:
            max_distance: Maximum horizontal and vertical distance from the epicentre for the
                          bounding box to be defined. If None then a default bounding box
                          size is determined that is based on magnitude between Mw 3.0 (100 km)
                          and Mw 8.0 (1000 km)
        Returns:
            Bounding box as a list of [llon, llat, ulon, ulat]
        """
        if not max_distance:
            # Scale from 100 km (for Mw 3 or less) to 1000 km for Mw >= 8.0
            if self.mag <= 3.0:
                max_distance = 100.0
            elif self.mag >= 8.0:
                max_distance = 1000.0
            else:
                # Interpolate
                max_distance = 100.0 + (self.mag - 3.0) * (900.0 / 5.0)
        # Define the bounding box from the specified maximum distance
        north = self.hypocenter.point_at(max_distance, 0.0, 0.0)
        south = self.hypocenter.point_at(max_distance, 0.0, 180.0)
        east = self.hypocenter.point_at(max_distance, 0.0, 90.0)
        west = self.hypocenter.point_at(max_distance, 0.0, 270.0)
        return [west.longitude, south.latitude, east.longitude, north.latitude]

    @property
    def rupture(self):
        """
        If a rupture is provided then it is returned, otherwise it will build the rupture
        from the available information
        """
        if self._rupture:
            return self._rupture
        centroid = Point(self.lon, self.lat, self.depth)
        if self.surface:
            # Rupture surface has been input, so build the OpenQuake rupture object from
            # existing properties
            self._rupture = ParametricProbabilisticRupture(
                self.mag, self.rake, None, centroid, self.surface, 1.0, None
            )
            return self._rupture
        return self._rupture

    @staticmethod
    def build_planar_surface(
        centroid: Point,
        strike: float,
        dip: float,
        length: float,
        width: float,
        lsd: float,
        usd: float = 0.0,
        hypo_loc: Tuple[float, float] = (0.5, 0.5),
    ):
        """
        From a set of rupture properties returns a planar surface whose dimensions
        are constrained by the seismogenic thickness of the crust

        Args:
            centroid:
                Centroid of the rupture as instance of `class`:openquake.hazardlib.geo.Point`
            length:
                Rupture length (in km)
            width:
                Down-dip rupture width (in km)

        Returns:
            Rupture plane as instance of :class:`openquake.hazardlib.geo.PlanarSurface`
        """
        rdip = radians(dip)
        thickness = lsd - usd
        # Determine whether the upper edge of the plane would be above the Earth's surface
        updip_width = hypo_loc[1] * width
        downdip_width = (1.0 - hypo_loc[1]) * width
        updip_depth_change = updip_width * sin(rdip)
        downdip_depth_change = downdip_width * sin(rdip)
        if centroid.depth < updip_depth_change:
            # This would move the rupture above the top surface so translate
            # the rupture down until the upper depth is at the top surface
            offset = updip_depth_change - centroid.depth
            updip_depth_change = centroid.depth
            downdip_depth_change += offset
        # Now to address the case that the bottom edge exceeds the seismogenic
        # thickness
        if downdip_depth_change > (lsd - centroid.depth):
            if (updip_depth_change + downdip_depth_change) > thickness:
                # Determine excess width and translate rupture updip
                offset = (centroid.depth + downdip_depth_change) - lsd
                offset_area = length * (offset / sin(rdip))
                rw_max = thickness / sin(rdip)
                length += offset_area / rw_max
                updip_depth_change = centroid.depth
                downdip_depth_change = lsd - centroid.depth
            else:
                # This would move the rupture below the lower surface, so relocate it
                offset = (centroid.depth + downdip_depth_change) - lsd
                downdip_depth_change = lsd - centroid.depth
                updip_depth_change += offset
        if dip % 90.0:
            updip_surface_length = updip_depth_change / tan(rdip)
            downdip_surface_length = downdip_depth_change / tan(rdip)
        else:
            # Vertical rupture, so no change in surface distance
            updip_surface_length = 0.0
            downdip_surface_length = 0.0
        # Now deal with strike parts
        left_length = hypo_loc[0] * length
        right_length = (1.0 - hypo_loc[0]) * length

        # Build corner points
        downdip_dir = (strike + 90.0) % 360
        updip_dir = (strike - 90.0) % 360
        mid_left = centroid.point_at(left_length, 0.0, (strike + 180.0) % 360.0)
        mid_right = centroid.point_at(right_length, 0.0, strike)
        top_left = mid_left.point_at(updip_surface_length, -updip_depth_change, updip_dir)
        top_right = mid_right.point_at(updip_surface_length, -updip_depth_change, updip_dir)
        bottom_left = mid_left.point_at(
            downdip_surface_length, downdip_depth_change, downdip_dir
        )
        bottom_right = mid_right.point_at(
            downdip_surface_length, downdip_depth_change, downdip_dir
        )
        try:
            surface = PlanarSurface.from_corner_points(
                top_left, top_right, bottom_right, bottom_left
            )
        except Exception as e:
            # If an exception is raised then something was wrong in
            # the geometry. Return the user information to help debug
            extended_error_message = [
                "Rupture surface failed to build with the following properties:",
                "Strike: %.2f, Dip: %.2f, Length: %.2f, Width: %.2f, Hypo. Pos: %s"
                % (strike, dip, length, width, str(hypo_loc)),
            ]
            for pnt in [top_left, top_right, bottom_right, bottom_left]:
                extended_error_message.append(str(pnt))
            extended_error_message.append(str(e))
            raise ValueError("\n".join(extended_error_message))
        return surface
