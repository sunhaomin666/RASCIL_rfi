import numpy as np

from os import path
import warnings



def rfi_impulse(fqs, lsts, rfi=None, chance=0.001, strength=20.0):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI impulses that
    are localized in time but span the frequency band.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        chance (float):
            the probability that a time bin will be assigned an RFI impulse
        strength (float): Jy
            the strength of the impulse generated in each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI'''
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi is not shape (lsts.size, fqs.size)"

    impulse_times = np.where(np.random.uniform(size=lsts.size) <= chance)[0]
    dlys = np.random.uniform(-300, 300, size=impulse_times.size)  # ns
    impulses = strength * np.array([np.exp(2j * np.pi * dly * fqs) for dly in dlys])
    if impulses.size > 0:
        rfi[impulse_times] += impulses
    return rfi



def rfi_scatter(fqs, lsts, rfi=None, chance=0.0001, strength=10, std=10):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI impulses that
    are localized in time but span the frequency band.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        chance (float): default=0.0001
            the probability that a time/freq bin will be assigned an RFI impulse
        strength (float): Jy, default=10
            the average amplitude of the spike generated in each time/freq bin
        std (float): Jy, default = 10
            the standard deviation of the amplitudes drawn for each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi shape is not (lsts.size, fqs.size)"

    rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]
    rfi.flat[rfis] += np.random.normal(strength, std) * np.exp(
        2 * np.pi * 1j * np.random.uniform(size=rfis.size)
    )
    return rfi


def rfi_dtv(fqs, lsts, rfi=None, freq_min=114, freq_max=154, width=8,
            chance=0.01, strength=10, strength_std=10):
    """
    Generate an (NTIMES, NFREQS) waterfall containing Digital TV RFI.

    DTV RFI is expected to be of uniform bandwidth (eg. 8MHz), in contiguous
    bands, in a nominal frequency range. Furthermore, it is expected to be
    short-duration, and so is implemented as randomly affecting discrete LSTS.

    There may be evidence that close times are correlated in having DTV RFI,
    and this is *not currently implemented*.

    Args:
        fqs (array-like): shape=(NFREQS,), MHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        freq_min, freq_max (float):
            the min and max frequencies of the full DTV band [MHz]
        width (float):
            Width of individual DTV bands [MHz]
        chance (float): default=0.0001
            the probability that a time/freq bin will be assigned an RFI impulse
        strength (float): Jy, default=10
            the average amplitude of the spike generated in each time/freq bin
        strength_std (float): Jy, default = 10
            the standard deviation of the amplitudes drawn for each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi shape is not (lst.size, fqs.size)"

    bands = np.arange(freq_min, freq_max, width)  # lower freq of each potential DTV band

    # ensure that only the DTV bands which overlap with the passed frequencies are kept
    bands = bands[np.logical_and(bands >= fqs.min() - width, bands <= fqs.max())]

    # if len(bands) is 0:
    #     warnings.warn("You are attempting to add DTV RFI to a visibility array whose " \
    #                   "frequencies do not overlap with any DTV band. Please ensure " \
    #                   "that you are using the correct frequencies.")

    delta_f = fqs[1] - fqs[0]

    chance = _listify(chance)
    strength_std = _listify(strength_std)
    strength = _listify(strength)

    if len(chance) == 1:
        chance *= len(bands)
    if len(strength) == 1:
        strength *= len(bands)
    if len(strength_std) == 1:
        strength_std *= len(bands)

    if len(chance) != len(bands):
        raise ValueError("chance must be float or list with len equal to number of bands")
    if len(strength) != len(bands):
        raise ValueError("strength must be float or list with len equal to number of bands")
    if len(strength_std) != len(bands):
        raise ValueError("strength_std must be float or list with len equal to number of bands")

    for band, chnc, strngth, str_std in zip(bands, chance, strength, strength_std):
        fq_ind_min = np.argwhere(band <= fqs)[0][0]
        try:
            fq_ind_max = np.argwhere(band + width <= fqs)[0][0]
        except IndexError:
            fq_ind_max = fqs.size
        this_rfi = rfi[:, fq_ind_min:min(fq_ind_max, fqs.size)]

        rfis = np.random.uniform(size=lsts.size) <= chnc
        this_rfi[rfis] += np.atleast_2d(np.random.normal(strngth, str_std, size=np.sum(rfis)) 
                          * np.exp(2 * np.pi * 1j * np.random.uniform(size=np.sum(rfis)))).T

    return rfi


def _listify(x):
    """
    Ensure a scalar/list is returned as a list.

    Gotten from https://stackoverflow.com/a/1416677/1467820
    """
    try:
        basestring
    except NameError:
        basestring = (str, bytes)

    if isinstance(x, basestring):
        return [x]
    else:
        try:
            iter(x)
        except TypeError:
            return [x]
        else:
            return list(x)


