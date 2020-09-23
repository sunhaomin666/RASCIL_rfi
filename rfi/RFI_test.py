"""Simulation for RFI and Flagged by AO-Flagger
"""

import logging
import sys
import unittest
from matplotlib import pyplot as plt
import astropy.units as u
import numpy

from astropy.coordinates import SkyCoord, EarthLocation
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.simulation.configurations import create_named_configuration
from rascil.processing_components.simulation.noise import addnoise_visibility
from rascil.processing_components.visibility.base import create_visibility, create_blockvisibility, copy_visibility
import os
from rascil.sim_rfi.rfi_function import rfi_impulse ,rfi_dtv
log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

run_ms_tests = False
try:
    import casacore
    from rascil.processing_components.visibility.base import create_blockvisibility, create_blockvisibility_from_ms
    from rascil.processing_components.visibility.base import export_blockvisibility_to_ms

    run_ms_tests = True
except ImportError:
    pass


BASE_DIR=os.path.dirname(os.path.abspath(__file__))

"""Functions used to simulate RFI. Developed as part of SP-122/SIM.



The scenario is:
* There is a TV station at a remote location (e.g. Perth), emitting a broadband signal (7MHz) of known power (50kW).
* The emission from the TV station arrives at LOW stations with phase delay and attenuation. Neither of these are
well known but they are probably static.
* The RFI enters LOW stations in a sidelobe of the main beam. Calulations by Fred Dulwich indicate that this
provides attenuation of about 55 - 60dB for a source close to the horizon.
* The RFI enters each LOW station with fixed delay and zero fringe rate (assuming no e.g. ionospheric ducting)
* In tracking a source on the sky, the signal from one station is delayed and fringe-rotated to stop the fringes for one direction on the sky.
* The fringe rotation stops the fringe from a source at the phase tracking centre but phase rotates the RFI, which
now becomes time-variable.
* The correlation data are time- and frequency-averaged over a timescale appropriate for the station field of view.
This averaging decorrelates the RFI signal.
* We want to study the effects of this RFI on statistics of the images: on source and at the pole.
"""

import numpy
from astropy import constants
import astropy.units as u
from astropy.coordinates import SkyCoord

from rascil.processing_components.util.array_functions import average_chunks2
from rascil.processing_components.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn
from rascil.processing_components.visibility.base import simulate_point
# from processing_components.simulation.rfi import calculate_averaged_correlation, simulate_rfi_block

def simulate_Noise(frequency, times, power=50e3, bchan = None, echan = None, timevariable=False, frequency_variable=False):
    """ Calculate Noise sqrt(power) as a function of time and frequency

    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :param bchan: first channel number
    :param echan: last channel number
    :return: Complex array [ntimes, nchan]
    """
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    if bchan is None:
        bchan = nchan // 4
    if echan is None:
        echan = nchan // 4 + 3
    amp = power / (max(frequency) - min(frequency))
    signal = numpy.zeros(shape, dtype='complex')
    if timevariable:
        if frequency_variable:
            sshape = [ntimes, nchan // 2]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape) \
                                    + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
        else:
            sshape = [ntimes]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape) \
                                    + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
    else:
        if frequency_variable:
            sshape = [nchan // 2]
            signal[:, bchan:echan] += (numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
                                   + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape))[numpy.newaxis, ...]
        else:
            signal[:, bchan:echan] = amp

    return signal

def create_propagators(config, interferer, frequency, attenuation=1e-9):
    """ Create a set of propagators

    :return: Complex array [nants, ntimes]
    """
    nchannels = len(frequency)
    nants = len(config.data['names'])
    interferer_xyz = [interferer.geocentric[0].value, interferer.geocentric[1].value, interferer.geocentric[2].value]
    propagators = numpy.zeros([nants, nchannels], dtype='complex')
    for iant, ant_xyz in enumerate(config.xyz):
        vec = ant_xyz - interferer_xyz
        # This ignores the Earth!
        r = numpy.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        k = 2.0 * numpy.pi * frequency / constants.c.value
        propagators[iant, :] = numpy.exp(- 1.0j * k * r) / r
    return propagators * attenuation


def calculate_rfi_at_station(propagators, emitter):
    """ Calculate the rfi at each station

    :param propagators: [nstations, nchannels]
    :param emitter: [ntimes, nchannels]
    :return: Complex array [nstations, ntimes, nchannels]
    """
    rfi_at_station = emitter[:, numpy.newaxis, ...] * propagators[numpy.newaxis, ...]
    rfi_at_station[numpy.abs(rfi_at_station) < 1e-15] = 0.
    return rfi_at_station


def calculate_station_correlation_rfi(rfi_at_station):
    """ Form the correlation from the rfi at the station

    :param rfi_at_station:
    :return: Correlation(nant, nants, ntimes, nchan] in Jy
    """
    ntimes, nants, nchan = rfi_at_station.shape
    correlation = numpy.zeros([ntimes, nants, nants, nchan], dtype='complex')

    for itime in range(ntimes):
        for chan in range(nchan):
            correlation[itime, ..., chan] = numpy.outer(rfi_at_station[itime, :, chan],
                                                        numpy.conjugate(rfi_at_station[itime, :, chan]))

    correlation1 =  correlation[..., numpy.newaxis] * 1e26
    return correlation1


def calculate_averaged_correlation(correlation, channel_width, time_width):
    """ Average the correlation in time and frequency

    :param correlation: Correlation(nant, nants, ntimes, nchan]
    :param channel_width: Number of channels to average
    :param time_width: Number of integrations to average
    :return:
    """
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (channel_width, time_width))[0]


#def simulate_rfi_block(bvis, emitter_location, emitter_power=5e3, attenuation=1.0):
def simulate_rfi_block(bvis, emitter_location, emitter,attenuation=1.0):
    """ Simulate RFI block

    :param config: ARL telescope Configuration
    :param times: observation times (hour angles)
    :param frequency: frequencies
    :param phasecentre:
    :param emitter_location: EarthLocation of emitter
    :param emitter_power: Power of emitter
    :param attenuation: Attenuation to be applied to signal
    :return:
    """

    # Calculate the power spectral density of the DTV station: Watts/Hz
    # emitter = simulate_Noise(bvis.frequency, bvis.time, power=emitter_power, timevariable=False)
    # emitter = numpy.load('/home/sunhaomin/re_map/re_map/RFI/yes_rfi100001.npy')
    # emitter =emitter.reshape(500,1024)
    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency. The ad hoc attenuation
    # is set to produce signal roughly equal to noise at LOW
    propagators = create_propagators(bvis.configuration, emitter_location, frequency=bvis.frequency,
                                     attenuation=attenuation)
    # Now calculate the RFI at the stations, based on the emitter and the propagators
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)

    # Calculate the rfi correlation using the fringe rotation and the rfi at the station
    # [ntimes, nants, nants, nchan, npol]
    bvis.data['vis'][...] = calculate_station_correlation_rfi(rfi_at_station)

    ntimes, nant, _, nchan, npol = bvis.vis.shape

    # Observatory Hour angle & Declination
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-26.0 * u.deg, frame='icrs', equinox='J2000')

    # Calculate phasor needed to shift from the phasecentre to the pole
    l, m, n = skycoord_to_lmn(pole, bvis.phasecentre)
    k = numpy.array(bvis.frequency) / constants.c.to('m s^-1').value
    uvw = bvis.uvw[..., numpy.newaxis] * k
    phasor = numpy.ones([ntimes, nant, nant, nchan, npol], dtype='complex')
    for chan in range(nchan):
        phasor[:, :, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]

    # Now fill this into the BlockVisibility
    bvis.data['vis'] = bvis.data['vis'] * phasor

    return bvis

if __name__ == '__main__':
   # if run_ms_tests == False:
    #    sys.exit()

    dir = BASE_DIR+"/results"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    msoutfile = dir + "/Test_10.ms"

    # self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
    # self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
    # self.flux = numpy.array([[100.0], [100.0], [100.0], [100.0], [100.0]])
    # self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    # self.config = create_named_configuration('LOWBD2-CORE')
    # self.timecds = numpy.linspace(-300.0, 300.0, 300) * numpy.pi / 43200.0
    # s = self.config.xyz.shape[0]

    sample_freq = 97656.25
    nchannels = 1024
    frequency = 100.000e+06 + numpy.arange(nchannels) * sample_freq

    channel_bandwith = numpy.array([sample_freq for i in range(len(frequency))])

    ntimes = 500
    integration_time = 0.5
    times = numpy.arange(ntimes) * integration_time

    # Perth from Google for the moment
    # perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    observatory = EarthLocation(lon="116.76445", lat="-26.825", height=300.0)
    # observatory = EarthLocation(lon="30.76445", lat="-26.825", height=300.0)
    phasecentre = SkyCoord(ra=0 * u.deg, dec=-26.0 * u.deg, frame='icrs', equinox='J2000')

    rmax = 1000.0
    # low = create_named_configuration('LOWR3', rmax=rmax)
    low = create_named_configuration('ASKAP', rmax=rmax)
    # antskip = 33
    # low.data = low.data[::antskip]
    nants = len(low.names)

    vis = create_blockvisibility(low, times, frequency, phasecentre=phasecentre,
                                      weight=1.0, polarisation_frame=PolarisationFrame('stokesI'),
                                      channel_bandwidth=channel_bandwith)
    
    fqs = 100.000e+00 + numpy.arange(nchannels) *0.09765625
    rfi_1=rfi_dtv(fqs=fqs,lsts=times,chance=1,strength=1.5e-5,freq_min=105,freq_max=106,width=0.1,strength_std=1e-6)     
    # rfi_2=rfi_dtv(fqs=fqs,lsts=times,chance=1,strength=3e-3,freq_min=110,freq_max=113,width=0.1,strength_std=1e-4)     
    # rfi_3=rfi_dtv(fqs=fqs,lsts=times,chance=1,strength=6e-3,freq_min=160,freq_max=161,width=0.1,strength_std=1e-4)      
    # rfi_4=rfi_dtv(fqs=fqs,lsts=times,chance=1,strength=9e-3,freq_min=180,freq_max=181,width=0.1,strength_std=1e-4) 
    rfi_5=rfi_dtv(fqs=fqs,lsts=times,chance=1,strength=1.3e-5,freq_min=138,freq_max=138.5,width=0.1,strength_std=1e-6) 
    rfi_6=rfi_dtv(fqs=fqs,lsts=times,chance=1,strength=1.2e-5,freq_min=145,freq_max=156,width=0.1,strength_std=1e-6)       
    # rfi_arr=rfi_1+rfi_2+rfi_3+rfi_4+rfi_impulse(fqs=frequency,lsts=times,chance=0.01)+rfi_5+rfi_6
    rfi_arr=rfi_1+rfi_impulse(fqs=frequency,lsts=times,chance=0.01)+rfi_5+rfi_6
    vis = simulate_rfi_block(vis, emitter_location=observatory, attenuation=2e-5,emitter=rfi_arr) 
    
    vis = addnoise_visibility(vis) 
    a=vis.data['vis']
    a=a[:,0,0,:,0]
    a=numpy.abs(a)
    
    # numpy.save('/home/sunhaomin/re_map/re_map/SKA_rfi/ska_1.npy',a)
    # plt.imshow(a)
    # plt.show()
    # plt.close()
    # print(a)
    # numpy.save('/home/sunhaomin/1.npy',a)
    # print(type(vis))
    vis_list = []
    vis_list.append(vis)


    export_blockvisibility_to_ms(msoutfile, vis_list, source_name='TEST')
    print('Done.')
    