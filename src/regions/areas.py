import numpy as np
from tess_stars2px import TESS_Spacecraft_Pointing_Data

RADIUS_CAMERA = np.sqrt(12.0 ** 2 + 12.0 ** 2)

_scinfo = None

def _pointing():

    global _scinfo
    if _scinfo is None:

        _scinfo = TESS_Spacecraft_Pointing_Data()

    return _scinfo

def sphere_distance(ra_1, dec_1, ra_2, dec_2):

    ra_1, dec_1, ra_2, dec_2 = map(np.radians, (ra_1, dec_1, ra_2, dec_2))

    h = (np.sin((dec_2 - dec_1) / 2) ** 2 + np.cos(dec_1) * np.cos(dec_2) * np.sin((ra_2 - ra_1) / 2) ** 2)

    final = np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))

    return final

def camera_center(sector, camera):

    sc = _pointing()
    
    inp = int(np.where(sc.sectors == sector)[0][0])

    return sc.camRa[camera - 1, inp], sc.camDec[camera - 1, inp]


def add_area(df):

    distance = np.full(len(df), np.nan)

    for (sec, cam), index in df.groupby(["sector", "camera"]).indices.items():

        cra, cdec = camera_center(sec, cam)

        distance[index] = sphere_distance(df["ra"].values[index], df["dec"].values[index], cra, cdec)

    ring = np.full(len(df), 4, dtype = int)
    #need to now separate into the separate regions of the circles based on equal distances of radius
    ring[distance < 0.75 * RADIUS_CAMERA] = 3
    ring[distance < 0.5 * RADIUS_CAMERA] = 2
    ring[distance < 0.25 * RADIUS_CAMERA] = 1

    df["cam_dist"] = distance
    df["area"] = df["camera"] * 100 + df["ccd"] * 10 + ring

    bad = ~np.isfinite(distance)
    df.loc[bad, "area"] = -1

    return df