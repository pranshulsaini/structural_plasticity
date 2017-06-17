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
import brian2
from brian2 import *

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
        self.t_sim = 1000*250.0    # This training case. 1000 examples, each for 250ms. It would 200000 in test case
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
        self.nodes_inp = None
        self.nodes_e = None
        self.nodes_i = None

        self.v_rest_e = -65 * mV
        self.v_rest_i = -60 * mV
        self.v_reset_e = -65. * mV
        self.v_reset_i = -45. * mV
        self.v_thresh_e = -52. * mV
        self.v_thresh_i = -40. * mV
        self.refrac_e = 5. * ms
        self.refrac_i = 2. * ms



        #I have not put wt_inp_i = 0.15 and wt_e_e = 0.03

        self.tc_pre_ee = 20 * ms
        self.tc_post_ee = 40 * ms
        self.tc_pre_ie = 20 * ms
        self.tc_post_ie = 20 *ms
        self.nu_pre_ee = 0.0005  # learning rate -- decrease for exc->exc
        self.nu_post_ee = 0.0025  # learning rate -- increase for exc->exc
        self.nu_ie = 0.005  # learning rate -- for inh->exc
        self.alpha_ie = 3 * Hz * self.tc_post_ie * 2  # controls the firing rate
        self.wmax_ee = 0.5
        self.wmax_ie = 1000.
        self.exp_pre_ee = 0.2
        self.exp_post_ee = self.exp_pre_ee

        self.delay_inp_e = (10 * ms - 0 * ms) * np.random.random() + 0 * ms
        self.delay_inp_i = (5 * ms - 0 * ms) * np.random.random() + 0 * ms
        self.delay_e_e = (5 * ms - 0 * ms) * np.random.random() + 0 * ms
        self.delay_e_i = (2 * ms - 0 * ms) * np.random.random() + 0 * ms
        self.delay_i_e = ((1 * ms - 0 * ms) * np.random.random() + 0 * ms
        self.delay_i_i = (5 * ms - 0 * ms) * np.random.random() + 0 * ms

        self.neuron_eqs_e = '''
                dv/dt = ((self.v_rest_e-v) + (I_synE+I_synI) / nS) / (20*ms)  : volt
                I_synE = ge * nS *         -v                           : amp
                I_synI = gi * nS * (-85.*mV-v)                          : amp
                dge/dt = -ge/(5.0*ms)                                   : 1
                dgi/dt = -gi/(10.0*ms)                                  : 1
                '''

        self.neuron_eqs_i = '''
                dv/dt = ((self.v_rest_i-v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
                I_synE = ge * nS * -v                           : amp
                I_synI = gi * nS * (-85.*mV-v)                          : amp
                dge/dt = -ge/(5.0*ms)                                   : 1
                dgi/dt = -gi/(10.0*ms)                                  : 1
                '''

        self.eqs_stdp_ee = '''
                    w : 1  
                    postbefore                        : 1.0
                    dpre/dt   =   -pre/(self.tc_pre_ee)       : 1.0 (event-driven)
                    dpost/dt = -post/(self.tc_post_ee)     : 1.0 (event-driven)
                    w = clip(w+post, 0, self.wmax_ee)
                    '''
        self.eqs_stdp_ie = '''
                    w : 1
                    dpre/dt   =  -pre/(self.tc_pre_ie)        : 1.0 (event-driven)
                    dpost/dt  = -post/(self.tc_post_ie)       : 1.0 (event-driven)
                    w = clip(w+post, 0, self.wmax_ie)
                    '''

        self.eqs_STDP_pre_ee = '''
                        pre = 1.
                        w -= self.nu_pre_ee * post * w**self.exp_pre_ee
                        '''
        self.eqs_STDP_post_ee = '''
                    postbefore = post
                    w += self.nu_post_ee * pre * postbefore * (wmax_ee - w)**exp_post_ee
                    post = 1.
                    '''

        self.eqs_STDP_pre_ie ='''
                    pre += 1.
                    w += self.nu_ie * (post-self.alpha_ie)
                    '''

        self.eqs_STDP_post_ie ='''
                    post += 1.
                    w += self.nu_ie * pre
                    '''


    def prepare_simulation(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus(
            {
                'resolution': self.dt
            }
        )


    def create_nodes(self):
        self.nodes_inp = PoissonGroup(self.number_input_neurons, self.bg_rate)
        self.nodes_e = NeuronGroup(self.number_excitatory_neurons, self.neuron_eqs_e, threshold= self.v_thresh_e, refractory= self.refrac_e, reset= self.v_reset_e, compile = True, freeze = True)
        self.nodes_i = NeuronGroup(self.number_inhibitory_neurons, self.neuron_eqs_i, threshold= self.v_thresh_i, refractory= self.refrac_i, reset= self.v_reset_i, compile= True, freeze= True)

    def connections(self):
        # we have to chose only 10% random connections in the beginning. I have created weight matrices for that

        # input population to excitatory neurons.



        for j in range(0,1600):
            pos_inp_e =  np.random.randint(0,1600,160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_inp_e:
                self.wt_inp_e[i][j] = np.random.random()   # weights have to be between 0 and 1
        #'weight': self.wt_inp_e , 'delay': 1.0, 'tau_plus': 15, 'Wmax':1.0})
        S = Synapses(self.nodes_inp, self.nodes_inp, model = self.eqs_stdp_ee, on_pre = self.eqs_STDP_pre_ee )
        S.connect()  # all to all connections except to itself
        S.delay = self.delay_inp_e  #It needs to be input as an array, not a matrix
        S.w = self.wt_inp_e         # It needs to be input as an array, not a matrix

        #input population to inhibitory neurons.

        for j in range(0, 1600):
            pos_inp_i = np.random.randint(0, 400,40)  # assigns random 40 positions between [0 and 400). Acts as a column of connection matrix
            for i in pos_inp_i:
                self.wt_inp_i[i][j] = (1./5)*np.random.random()  # weights have to be between 0 and 0.2
        nest.Connect(self.nodes_inp, self.nodes_i,conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_inp_i , 'delay': 1.0, 'tau_plus': 15, 'Wmax':0.2})


        #recurrent connections from excitatory neurons on itself.

        for j in range(0, 1600):
            pos_e_e = np.random.randint(0, 1600,160)
            for i in pos_e_e:
                self.wt_e_e[i][j] = 200. * (1. / 5) * np.random.random()  # weights have to be between 0 and 0.2
            self.wt_e_e[j][j] = 0.0   #prevent from connecting a neuron to itself in case it has randomly been assigned
        nest.Connect(self.nodes_e, self.nodes_e, conn_spec = {'rule': 'all_to_all','autapses': False }, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_e_e , 'delay': 1.0, 'tau_plus': 15, 'Wmax':0.2})


        # connections from excitatory population to inhibitory population.

        for j in range(0, 1600):
            pos_e_i = np.random.randint(0, 400, 40)
            for i in pos_e_i:
                self.wt_e_i[i][j] = 200. * (1. / 5) * np.random.random()  # weights have to be between 0 and 0.2
        nest.Connect(self.nodes_e, self.nodes_i, conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_e_i , 'delay': 1.0, 'tau_plus': 15, 'Wmax':0.2})

        # connections from inhibitory population to excitatory population.

        for j in range(0, 400):
            pos_i_e =  np.random.randint(0, 1600, 160)
            for i in pos_i_e:
                self.wt_i_e[i][j] = 200. * np.random.random()  # weights have to be between 0 and 1
        nest.Connect(self.nodes_i, self.nodes_e, conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_i_e , 'delay': 1.0, 'tau_plus': 15, 'Wmax':0.2})

        # connections from inhibitory population to inhibitory population.

        for j in range(0, 400):
            pos_i_i =  np.random.randint(0, 400, 40)
            for i in pos_i_i:
                self.wt_i_i[i][j] = 200. * (1 / 2.5) * np.random.random()  # weights have to be between 0 and 0.4
            self.wt_i_i[j][j] = 0.0  # prevent from connecting a neuron to itself in case it has randomly been assigned
        nest.Connect(self.nodes_i, self.nodes_i, conn_spec = {'rule': 'all_to_all','autapses': False }, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_i_i , 'delay': 1.0, 'tau_plus': 15, 'Wmax':0.4})

        # to record the spikes from the excitaory neurosn of computation population.
        self.spikedetector = nest.Create("spike_detector", 1600, params={"withgid": True, "withtime": True})
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
