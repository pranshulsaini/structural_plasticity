#no structural plasticity in this code. Only stdp synapses. The output should stabilise to input gaussian pattern after some time

import nest
import math
import cmath
import numpy
import matplotlib.pyplot as pl
import sys
from scipy.optimize import curve_fit
import numpy as np
import pylab

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def func(x, mu, sigma, a):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def rmse(predictions, targets):  #root mean square error
    return np.sqrt(((predictions - targets) ** 2).mean())

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


def middlesort(foo):

    foo = numpy.sort(foo)
    length = len(foo)
    l = length
    fooo = np.zeros(l)
    index = l / 2

    for i in range(index):
        fooo[index - 1] = foo[l - 1]
        index = index - 1
        l = l - 2

    l = length
    index = l / 2

    for i in range(length - index):
        fooo[index] = foo[l - 2]
        index = index + 1
        l = l - 2
    return fooo


class stdp_class:
    def __init__(self):

        # simulated time (ms)
        self.t_sim = 1000*250.0    # This training case. 15000 examples, each for 250ms. It would 200000 in test case
        # simulation step (ms).
        self.dt = 0.5
        #record time
        self.record_interval = 1000.

        #input change
        self.input_change_interval = 250.0

        #input population
        self.number_input_neurons = 1600

        #computation population
        self.number_excitatory_neurons = 1600
        self.number_inhibitory_neurons = 400

        # rate of background Poisson input. I have to make it a guassian over 1600 units
        self.xdata = np.linspace(0, 1599, 1600) / 1600
        self.sigma_input = 1./6.
        self.mu_input = 0.5
        self.bg_rate = 20. * wrap_gaussian(self.xdata, self.mu_input, self.sigma_input)

        # will contain firing rate of excitatory neurons with time
        self.spikes = np.zeros(1600)
        self.old_spikes = np.zeros(1600)
        self.new_spikes = np.zeros(1600)
        self.firing_rate = np.zeros(1600)
        self.y_fit = np.zeros(1600)

        # connections
        self.wt_inp_e = np.zeros((self.number_excitatory_neurons, self.number_input_neurons))
        self.wt_inp_i = np.zeros((self.number_inhibitory_neurons, self.number_input_neurons))
        self.wt_e_e = np.zeros((self.number_excitatory_neurons, self.number_excitatory_neurons))
        self.wt_e_i = np.zeros((self.number_inhibitory_neurons, self.number_excitatory_neurons))
        self.wt_i_e = np.zeros((self.number_excitatory_neurons, self.number_inhibitory_neurons))
        self.wt_i_i = np.zeros((self.number_inhibitory_neurons, self.number_inhibitory_neurons))

        #will contain the noise
        self.noise = []

        #counter to execute initial condition for mean only once
        self.counter = 0
        self.mu0 = []

        # excitatory and inhibitory leaky integrate-and-fire neurons.
        self.neuron_model = 'iaf_psc_alpha'  # I have to keep simple iaf neurons

        self.model_params_ex = {'tau_m': 20.0,  # membrane time constant (ms)
                             #'tau_syn': 0.5,  # excitatory synaptic time constant (ms)
                             't_ref': 2.0,  # absolute refractory period (ms)
                             'E_L': -65.0,  # resting membrane potential (mV)
                             'V_th': -52.0,  # spike threshold (mV)
                             'C_m': 250.0,  # membrane capacitance (pF)
                             'V_reset': -65.0,  # reset potential (mV)
                             'tau_minus': 30.0 # used for stdp
                             }

        self.model_params_in = {'tau_m': 10.0,  # membrane time constant (ms)
                                #'tau_syn': 0.5,  # excitatory synaptic time constant (ms)
                                't_ref': 2.0,  # absolute refractory period (ms)
                                'E_L': -60.0,  # resting membrane potential (mV)
                                'V_th': -40.0,  # spike threshold (mV)
                                'C_m': 250.0,  # membrane capacitance (pF)
                                'V_reset': -45.0,  # reset potential (mV)
                                'tau_minus': 30.0  # used for stdp
                                }

        self.nodes_inp = None
        self.nodes_e = None
        self.nodes_i = None
        self.mean_ca_e = []
        self.mean_ca_i = []
        self.total_connections_e = []
        self.total_connections_i = []


    def prepare_simulation(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus(
            {
                'resolution': self.dt
            }
        )


    def create_nodes(self):
        self.nodes_inp = nest.Create(self.neuron_model, self.number_input_neurons)
        self.nodes_e = nest.Create(self.neuron_model, self.number_excitatory_neurons, params = self.model_params_ex)
        self.nodes_i = nest.Create(self.neuron_model, self.number_inhibitory_neurons, params = self.model_params_in)


    def connections(self):
        poisson = nest.Create('poisson_generator', 1600)   # we have to create many poisson generators because firing rate to each input neuron has to be different
        nest.SetStatus(poisson, "rate", self.bg_rate)  # I hope this notation works
        nest.Connect(poisson, self.nodes_inp, "one_to_one", syn_spec = {'weight': 1500}) # default syapses is static_synapse, where weight = 1. Now, it is 1500


        # I have now made random connections in the beginning. I have created weight matrices for that

        # input population to excitatory neurons.
        #self.wt_inp_e = 30. * np.random.rand(1600,1600)   # weights have to be between 0 and 1
        nest.Connect(self.nodes_inp, self.nodes_e, conn_spec = {'rule': 'fixed_total_number', 'N': 1600*160}, syn_spec = {'model': 'stdp_synapse', 'weight': {'distribution': 'uniform', 'low': 0.0, 'high': 1.0} , 'delay': 1.0, 'tau_plus': 15})

        #input population to inhibitory neurons.
        #self.wt_inp_i = 30. *(1./5)*np.random.rand(400,1600)  # weights have to be between 0 and 0.2
        nest.Connect(self.nodes_inp, self.nodes_i,conn_spec = {'rule': 'fixed_total_number', 'N': 400*160}, syn_spec = {'model': 'stdp_synapse', 'weight': {'distribution': 'uniform', 'low': 0.0, 'high': 0.2} , 'delay': 1.0, 'tau_plus': 15})

        #recurrent connections from excitatory neurons on itself.
        #self.wt_e_e = 30. * (1. / 5) * np.random.rand(1600,1600)  # weights have to be between 0 and 0.2
        #for i in range(1600):
        #    self.wt_e_e[i][i] = 0.0   #prevent from connecting a neuron to itself in case it has randomly been assigned
        nest.Connect(self.nodes_e, self.nodes_e, conn_spec = {'rule': 'fixed_total_number', 'N': 1600*160,'autapses': False }, syn_spec = {'model': 'stdp_synapse', 'weight': {'distribution': 'uniform', 'low': 0.0, 'high': 0.2}, 'delay': 1.0, 'tau_plus': 15, 'Wmax': 0.5})

        # connections from excitatory population to inhibitory population.
        #self.wt_e_i = 30. *(1. / 5) * np.random.rand(400, 1600)  # weights have to be between 0 and 0.2
        nest.Connect(self.nodes_e, self.nodes_i, conn_spec = {'rule': 'fixed_total_number', 'N': 400*160}, syn_spec = {'model': 'stdp_synapse', 'weight': {'distribution': 'uniform', 'low': 0.0, 'high': 0.2}, 'delay': 1.0, 'tau_plus': 15})

        # connections from inhibitory population to excitatory population.
        #self.wt_i_e =  30. * np.random.rand(1600,400)  # weights have to be between 0 and 1
        nest.Connect(self.nodes_i, self.nodes_e, conn_spec = {'rule': 'fixed_total_number', 'N': 1600*40}, syn_spec = {'model': 'stdp_synapse', 'weight': {'distribution': 'uniform', 'low': 0.0, 'high': 1.0} , 'delay': 1.0, 'tau_plus': 15, 'alpha':0.12, 'Wmax': 1000.})

        # connections from inhibitory population to inhibitory population.
        #self.wt_i_i = 30. * (1 / 2.5) * np.random.rand(400,400)  # weights have to be between 0 and 0.4
        #for j in range(0, 400):
        #    self.wt_i_i[j][j] = 0.0  # prevent from connecting a neuron to itself in case it has randomly been assigned
        nest.Connect(self.nodes_i, self.nodes_i, conn_spec = {'rule': 'fixed_total_number', 'N': 400*40,'autapses': False }, syn_spec = {'model': 'stdp_synapse', 'weight': {'distribution': 'uniform', 'low': 0.0, 'high': 0.4} , 'delay': 1.0, 'tau_plus': 15})



        # to record the spikes from the excitaory neurosn of computation population.
        self.spikedetector = nest.Create("spike_detector", 1600, params={"withgid": True, "withtime": True, 'precise_times': True})
        nest.Connect(self.nodes_e, self.spikedetector, 'one_to_one')


    def record_spikes(self):
        for i in range(1600):
            self.spikes[i] = len(nest.GetStatus(self.spikedetector, keys="events")[i]['senders'])
        self.new_spikes = self.spikes
        self.spikes = self.spikes - self.old_spikes
        print ('The total number of new spikes of the population are', np.sum(self.spikes))
        self.firing_rate = self.spikes/(self.record_interval * (10**-3))
        print ('')
        self.old_spikes = self.new_spikes
        #print ('The firing rate is:', self.firing_rate)


    def noise_estimation(self):
        self.firing_rate = middlesort(self.firing_rate)
        #popt, pcov = curve_fit(func, self.xdata, self.firing_rate)
        #self.y_fit = func(self.xdata, *popt)
        noise = rmse(self.firing_rate, self.bg_rate)
        print ('The noise is:', noise)
        return noise


    def simulate(self):
        if nest.NumProcesses() > 1:
            sys.exit("For simplicity, this example only works for a single process.")

        print("Starting simulation")

        #sim_steps = numpy.arange(0, self.t_sim, self.record_interval)
        sim_steps = numpy.arange(0, self.t_sim, self.input_change_interval)  # for training phase

        for i, step in enumerate(sim_steps):
            nest.Simulate(self.input_change_interval)  # simulates the network for the time in the arguments. The argument here is for the training phase
            self.bg_rate = np.random.rand()   # this is for training examples
            self.record_spikes()
            self.noise.append(self.noise_estimation())
            if i % 20 == 0:  # happens every 20*1000 ms = 20s
                print("Progress: " + str(i / 2) + "%")
            stdp_case.plot_data1()
            stdp_case.plot_data2()
            stdp_case.plot_data3()
        print("Simulation finished successfully")


    def plot_data1(self):
        fig, ax1 = pl.subplots()
        #ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Number of iterations")
        ax1.set_ylabel("Noise value")
        ax1.plot(self.noise, 'm', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdpNoise.eps', format='eps')


    def plot_data2(self):
        fig, ax2 = pl.subplots()
        ax2.set_xlabel("Number of neurons")
        ax2.set_ylabel("Firing rate")
        ax2.plot(self.firing_rate, 'b', label='Noise', linewidth=2.0, linestyle='--')
        ax2.plot(self.bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdpfiringrate.eps', format='eps')

    def plot_data3(self):
        fig, ax3 = pl.subplots()
        ax3.set_xlabel("Number of neurons")
        ax3.set_ylabel("Fitted Firing Rate")
        ax3.plot(self.y_fit, 'b', label='Noise', linewidth=2.0, linestyle='--')
        ax3.plot(self.bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdpfittedfiringrate.eps', format='eps')


if __name__ == '__main__':
    stdp_case = stdp_class()
    # Prepare simulation
    stdp_case.prepare_simulation()
    stdp_case.create_nodes()
    stdp_case.connections()
    # Start simulation
    stdp_case.simulate()
    stdp_case.plot_data1()
    stdp_case.plot_data2()
    stdp_case.plot_data3()

    # creating a population of inhibitory and excitatory neurons
