""" Unit tests for RFI simulation

"""

import logging
import re

import astropy.units as u
import numpy
import numpy.testing
import pytest
from astropy.coordinates import SkyCoord

from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import (
    apply_beam_gain_for_low,
    calculate_station_correlation_rfi,
    simulate_rfi_block_prop,
    match_frequencies,
)
from rascil.processing_components.visibility.base import create_blockvisibility,create_blockvisibility_from_ms

log = logging.getLogger("rascil-logger")
log.setLevel(logging.WARNING)

# NCHANNELS = 1000
# NTIMES = 100

# lofar = create_blockvisibility_from_ms('/Users/sunhaomin/work/data/lofar.MS',start_chan=1,end_chan=60)[0]
# shuju = numpy.load('shuju.npy')

def setup_telescope(telescope):
    """Initialise common elements"""
    rmax = 150.0
    antskip = 1
    configuration = create_named_configuration(telescope, rmax=rmax, skip=antskip)

    # Info. for dummy BlockVisibility
    ftimes = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    if telescope == "MID":
        ffrequency = numpy.linspace(1.4e9, 1.9e9, 5)
        channel_bandwidth = numpy.array([1e8, 1e8, 1e8, 1e8, 1e8])
        phasecentre = SkyCoord(
            ra=0.0 * u.deg, dec=+30.0 * u.deg, frame="icrs", equinox="J2000"
        )

    else:
        ffrequency = numpy.linspace(1.3e8, 1.4e8, 50)
        channel_bandwidth = numpy.array([2e5]*50)
        # Set the phasecentre so as to point roughly towards Perth at transit for Low
        phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-62.8 * u.deg, frame="icrs", equinox="J2000"
        )

    polarisation_frame = PolarisationFrame("linear")
    bvis = create_blockvisibility(
        configuration,
        ftimes,
        ffrequency,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        phasecentre=phasecentre,
        weight=1.0,
    )

    return bvis


rfi_frequencies = numpy.load('chan_rfi.npy')




# print(shuju[1,1,:,1])



bvis = setup_telescope("LOW")
nants_start = len(bvis.configuration.names)

emitter_power = numpy.zeros(
    (1, len(bvis.time), nants_start, len(bvis.frequency)), dtype=complex
)

emitter_power[0,0,0,:] = numpy.load('shuju.npy')[0,0:50]


emitter_coordinates = numpy.ones((1, len(bvis.time), nants_start, 3))

emitter_coordinates[:, :, :, 0] = 0.0
emitter_coordinates[:, :, :, 1] = 20.0
emitter_coordinates[:, :, :, 2] = 600000.0




# print(emitter_power[0,0,0,:])


rfi_data = simulate_rfi_block_prop(
        bvis,
        emitter_power,
        emitter_coordinates,
        ["source1"],
        bvis.frequency.values,
        low_beam_gain=None,
        
)

import matplotlib.pyplot as plt

# plt.imshow(rfi_data.vis.data[:])
data = rfi_data.vis.data[:,0,:,0]
print(rfi_data.vis.data[:,0,:,0].shape)

imp = numpy.abs(data)
plt.imshow(imp)
plt.show()
plt.close()
