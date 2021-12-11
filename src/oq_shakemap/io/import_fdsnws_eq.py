#
# Grab an event's XML file from GEOFON and convert it
# to what we need.
#
# Author: Peter L. Evans <pevans@gfz-potsdam.de>
#
# Copyright (C) 2021 Helmholtz-Zentrum Potsdam - Deutsches GeoForschungsZentrum GFZ
#
# This software is free software and comes with ABSOLUTELY NO WARRANTY.
#
# ----------------------------------------------------------------------

"""
Can run standalone:

    >>> python import_fdsnws_eq.py gfz2021ekhv

"""

import os
import sys
from typing import List, Union
import urllib.request as ul
from http.client import HTTPException
from xml.etree import ElementTree as ET
import yaml


FDSNWS_ENDPOINT: str = "http://geofon.gfz-potsdam.de/fdsnws/event/1/"
QUAKEML_NS: dict = {"ns": "http://quakeml.org/xmlns/bed/1.2"}


def _perror(msg: str) -> None:
    print(msg, file=sys.stderr)


def fetch_fm(root: ET.Element, ns: dict, preferredfmID: str) -> list:
    """
    Extract interesting properties from the focal mechanism with
    the given publicID.

    root: ElementTree root element.
    ns (dict): its namespace.
    preferredfmID (string): the focal mechanism to hunt for.

    Returns a list of one dictionary filled from each of the
    <nodalPlane1> and <nodalPlane2> components of the <nodalPlane>
    element.

    """
    for fm in root[0][0].findall("ns:focalMechanism", ns):
        if fm.attrib["publicID"] != preferredfmID:
            continue

        # We've found our guy!

        np = fm.find("ns:nodalPlanes", ns)
        # Expect there's only one <nodalPlanes>!
        if not np:
            _perror("Oops: no <nodalPlanes> object seen")
            break

        d: List[dict] = [dict(), dict()]
        for k in range(2):
            plane = np.find("ns:nodalPlane" + str(k + 1), ns)
            if plane is None:
                continue
            for child in plane:
                found = child.find("ns:value", ns)
                if found is None:
                    continue
                v = found.text
                tag = child.tag
                tag_idx = tag.index("}") + 1
                tag = tag[tag_idx:]
                try:
                    d[k][tag] = float(str(v))
                except ValueError:
                    pass
    return d


def fetch_magnitude(
    root: ET.Element, ns: dict, preferredmagnitudeID: str
) -> Union[float, None]:
    for m in root[0][0].findall("ns:magnitude", ns):
        if m.attrib["publicID"] != preferredmagnitudeID:
            continue

    child = m.find("ns:mag", ns)
    if child is None:
        return None
    value = child.find("ns:value", ns)
    if value is None:
        return None
    v = value.text
    try:
        mag = float(str(v))
    except ValueError:
        return None
    return mag


def fetch_origin(root, ns: dict, preferredoriginID: str) -> dict:
    """
    Extract interesting properties from the origin with given publicID.

    root: ElementTree root element.
    ns (dict): the XML object's namespace.
    preferredoriginID (string): the origin to hunt for.
    """
    for o in root[0][0].findall("ns:origin", ns):
        if o.attrib["publicID"] != preferredoriginID:
            continue

        d = dict()
        for child in o:
            tag = child.tag
            tag_idx = tag.index("}") + 1
            tag = tag[tag_idx:]

            if tag in ("depth", "latitude", "longitude", "time"):
                v = child.find("ns:value", ns).text
                if tag == "time":
                    d[tag] = v
                else:
                    try:
                        d[tag] = float(v)
                    except ValueError:
                        pass

        # QuakeML depths are in metres.
        d["depth"] = d["depth"] / 1000.0
    return d


def fetch_quakeml_ws(evid: str) -> str:
    """
    Query fdsnws-event web service, and return string
    to fetch_quakeml() for parsing.

    evid (string): the event ID to query for.

    """
    url = (
        FDSNWS_ENDPOINT
        + "query?"
        + "&".join(
            (
                "includeallmagnitudes=true",
                "includefocalmechanism=true",
                "eventid={}".format(evid),
            )
        )
    )

    req = ul.Request(url)
    u = ul.urlopen(req)
    if u.code != 200:
        raise HTTPException("FDSN result not returned for event %s with url:\n%s" % (evid, url))
    buf = u.read().decode("utf8")

    print("Got", len(buf), "char(s).")
    return buf


def fetch_quakeml(path: str) -> Union[dict, None]:
    """
    Prepare a dictionary holding the interesting things found by
    reading the QuakeML file served for a given event.

    If there is no focal mechanism, return None where that would otherwise go.

    path (string): Either a local file name, or a GEOFON event ID
                   such as 'gfz2044sqpr'.

    """
    if os.path.exists(path):
        with open(path) as fid:
            buf = fid.read()
    elif path.startswith("gfz") and len(path) == len("gfz2044sqpr"):
        buf = fetch_quakeml_ws(path)
    else:
        raise IOError("Not a local path or a GEOFON event ID")

    root = ET.fromstring(buf)
    ns = QUAKEML_NS

    event = root[0][0]
    if not event:
        print("Couldn't get an event!")
        return None
    if "{" + ns["ns"] + "}event" != event.tag:
        print("Got a", event.tag, "but expected an event")
        return None
    try:
        # e.g. "smi:org.gfz-potsdam.de/geofon/gfz2021ekhv"
        evid = event.attrib["publicID"].split("/")[-1]
    except AttributeError:
        _perror("Oops, couldn't get event id from " + event.attrib["publicID"])

    preferredoriginIDElem = root[0][0].find("ns:preferredOriginID", ns)
    if preferredoriginIDElem is None:
        _perror("Oops, couldn't find the preferred origin ID")
        return None
    preferredoriginID = preferredoriginIDElem.text

    preferredmagID = None
    preferredMagnitudeIDElem = root[0][0].find("ns:preferredMagnitudeID", ns)
    if preferredMagnitudeIDElem is not None:
        try:
            preferredmagID = preferredMagnitudeIDElem.text
        except AttributeError:
            pass
    preferredfmID = None
    try:
        preferredfmIDElem = root[0][0].find("ns:preferredFocalMechanismID", ns)
        if preferredfmIDElem is not None:
            preferredfmID = preferredfmIDElem.text
    except AttributeError:
        pass

    if not preferredoriginID:
        print("Oops, no preferredOriginID was found")
        return None
    origin = fetch_origin(root, ns, preferredoriginID)
    (d, t) = origin.pop("time").split("T", 2)
    # There's little point in forcing these into datetimes,
    # since writing to YAML requires they be converted back
    # to strings.  :(

    # d = datetime.date(2020, 1, 1)
    # t = datetime.time(12, 0, 30, 123000)

    focalmech = None
    if preferredfmID:
        focalmech = fetch_fm(root, ns, preferredfmID)

    # Should this be event's preferred magnitude,
    # or the preferred focal mechanism's <momentMagnitudeID>?
    # Probably they are the same thing...
    if not preferredmagID:
        print("Oops, no preferredMagnitudeID was found")
        return None
    mag = fetch_magnitude(root, ns, preferredmagID)

    return {
        "id": evid,
        "date": str(d),
        "time": str(t),
        "origin": origin,
        "magnitude": mag,
        "focalmechanism": focalmech,
        "preferred_origin_id": preferredoriginID,
        "preferred_magnitude_id": preferredmagID,
        "preferred_focalmechanism_id": preferredfmID,
    }


if __name__ == "__main__":
    try:
        evid = sys.argv[1]
    except IndexError:
        print("Usage:", sys.argv[0], "EVENTID")
        print(
            """
where EVENTID is a GEOFON event ID. These are listed at
http://geofon.gfz-potsdam.de/eqinfo/list.php
"""
        )
        sys.exit(1)

    if not evid.startswith("gfz20"):
        print("GEOFON event IDs are generally of the form 'gfz20xxnnnn'. Are you sure?")
        sys.exit(1)

    ev_info = fetch_quakeml(evid)
    if ev_info is None:
        print("Got no dictionary from QuakeML")
    outfile = evid + ".yaml"
    with open(outfile, "w") as fid:
        fid.write(yaml.safe_dump(ev_info, default_flow_style=False))
        print("Wrote to", outfile)
