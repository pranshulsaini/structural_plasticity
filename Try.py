#neuron receiving two poisson spike trains

import numpy as np
import nest
import pylab
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def wrap_gaussian(x, mu, sig):
    if mu == 0.5:
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    elif mu > 0.5:
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        mu_new = mu - 1.0
        mark = int(abs(1600*(mu - 0.5)))
        gauss1 = np.exp(-np.power(x - mu_new, 2.) / (2 * np.power(sig, 2.)))
        for i in range(0, mark):
            gauss[i]= gauss1[i]
    else:
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        mu_new = mu + 1.0
        mark = int(abs(1600 - 1600 * abs((mu - 0.5))))
        gauss1 = np.exp(-np.power(x - mu_new, 2.) / (2 * np.power(sig, 2.)))
        for i in range(mark, 1600):
            gauss[i] = gauss1[i]

    return gauss






def func(x, mu, sigma, a):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def rmse(predictions, targets):  #root mean square error
    return np.sqrt(((predictions - targets) ** 2).mean())

firing_rate = np.zeros(1600)

xdata = np.linspace(0, 1599, 1600) / 1600
sigma_input = 0.0833
mu_input = 0.0
bg_rate1 = 20. * wrap_gaussian(xdata, mu_input, sigma_input)
bg_rate2 = 20. * wrap_gaussian(xdata, mu_input+0.1, sigma_input)
bg_rate3 = 20. * wrap_gaussian(xdata, mu_input+0.3, sigma_input)
bg_rate4 = 20. * wrap_gaussian(xdata, mu_input-0.2, sigma_input)

pylab.plot(xdata, bg_rate1)
pylab.plot(xdata, bg_rate2)
pylab.plot(xdata, bg_rate3)
pylab.plot(xdata, bg_rate4)
pylab.show()

# dmm = nest.GetStatus(multimeter)
# pylab.figure(1)
# pylab.plot(dmm[0]['events']['times'], dmm[0]['events']['V_m'] )
# dSD = nest.GetStatus(spikedetector, keys = 'events')
# pylab.figure(2)
# pylab.plot(dSD[0]['times'], dSD[0]['senders'], ".")
# #pylab.show()    # to show the graph



# print (nest.GetStatus(spikedetector, keys = 'events'))
# print ('The number of spikes are',(len(nest.GetStatus(spikedetector, keys = 'events')[2]['senders'])))
#
# conn = nest.GetConnections(neuron1, neuron2)
# print nest.GetStatus(conn)[6]['weight']





