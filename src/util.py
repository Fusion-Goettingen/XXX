import numpy as np


def llas_to_cart(llas, origin=None):
    def lla_to_cart(lla, scale):
        lat = lla[0]
        lon = lla[1]
        alt = lla[2]
        er = 6378137.0
        tx = scale * lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        tz = alt
        t = np.array([tx, ty, tz])
        return t

    if origin is not None:
        #print(origin,llas)
        llas = np.vstack((origin, llas))
        return llas_to_cart(llas)[1:]

    scale = None
    ts = np.zeros((len(llas), 3))

    for i, lla in enumerate(llas):
        lat = lla[0]
        if scale is None:
            scale = np.cos(lat * np.pi / 180.0)
        t = lla_to_cart(lla, scale)
        if origin is None:
            origin = t
        ts[i] = t - origin

    return ts


def carts_to_lla(carts, origin):
    def cart_to_lla(cart, origin):
        er = 6378137.0

        x, y, z = cart
        olat, olon, oalt = origin
        scale = np.cos(olat * np.pi / 180.0)
        x = x / scale
        y = y / scale

        lon = olon + x / er
        lat = 2 * np.arctan(np.exp(y / er)) - np.pi / 2
        alt = z

        return np.array([lat, lon, alt])

    ts = np.zeros((len(carts), 3))

    if origin is None:
        origin = carts[0]

    for i, cart in enumerate(carts):
        ts = cart_to_lla(carts[i], origin)

    return ts
