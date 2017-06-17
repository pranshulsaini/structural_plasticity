#The basic structural plasticity in this code. The output should stabilise to input gaussian pattern after some time
# The weights have been multiplied by 500 because the firing rate was coming out to be zero otherwise
# After removing the synapse with the weight less than 30, I am not adding any random synapse because I am not dealing with 10%
#neurons here, but 100 %. This deletion might be seen as pruining as well

import nest
import math
import cmath
import numpy
import matplotlib.pyplot as pl
import sys
from scipy.optimize import curve_fit
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def func(x, mu, sigma, a, o_noise):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) + o_noise

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
        self.t_sim = 200000.0
        # simulation step (ms).
        self.dt = 0.5
        #record time
        self.record_interval = 1000.   # this much time would let the input population catch up the poisson generator


        #input population
        self.number_input_neurons = 1600

        #computation population
        self.number_excitatory_neurons = 1600
        self.number_inhibitory_neurons = 400

        # rate of background Poisson input. I have to make it a guassian over 1600 units
        self.xdata = np.linspace(0, 1599, 1600) / 1600
        self.sigma_input = 0.0833
        self.mu_input = 0.5
        self.bg_rate = 20. * gaussian(self.xdata, self.mu_input, self.sigma_input)

        # will contain firing rate of excitatory neurons with time
        self.spikes = np.zeros(1600)
        self.old_spikes = np.zeros(1600)
        self.new_spikes = np.zeros(1600)
        self.firing_rate = np.zeros(1600)
        self.y_fit = np.zeros(1600)

        #weight threshold to delete the synapse
        self.weight_threshold = 0.03

        #will contain the noise
        self.noise = []

        #counter to execute initial condition for mean only once
        self.counter = 0
        self.mu0 = []

        # excitatory and inhibitory leaky integrate-and-fire neurons.
        self.neuron_model = 'iaf_neuron'  # I have to keep simple iaf neurons

        self.model_params_ex = {'tau_m': 20.0,  # membrane time constant (ms)
                             'tau_syn': 0.5,  # excitatory synaptic time constant (ms)
                             't_ref': 2.0,  # absolute refractory period (ms)
                             'E_L': -65.0,  # resting membrane potential (mV)
                             'V_th': -52.0,  # spike threshold (mV)
                             'C_m': 250.0,  # membrane capacitance (pF)
                             'V_reset': -65.0,  # reset potential (mV)
                             'tau_minus': 30.0 # used for stdp
                             }

        self.model_params_in = {'tau_m': 10.0,  # membrane time constant (ms)
                                'tau_syn': 0.5,  # excitatory synaptic time constant (ms)
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

        #connections
        self.wt_inp_e = np.zeros((self.number_excitatory_neurons, self.number_input_neurons))
        self.wt_inp_i = np.zeros((self.number_inhibitory_neurons, self.number_input_neurons))
        self.wt_e_e = np.zeros((self.number_excitatory_neurons, self.number_excitatory_neurons))
        self.wt_e_i = np.zeros((self.number_inhibitory_neurons, self.number_excitatory_neurons))
        self.wt_i_e = np.zeros((self.number_excitatory_neurons, self.number_inhibitory_neurons))
        self.wt_i_i = np.zeros((self.number_inhibitory_neurons, self.number_inhibitory_neurons))


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
        poisson = nest.Create('poisson_generator', 1600)
        nest.SetStatus(poisson, "rate", self.bg_rate)  # I hope this notation works
        nest.Connect(poisson, self.nodes_inp, "one_to_one", syn_spec = {'weight': 1500.}) # default syapses is static_synapse, where weight = 1.  # 1500 weight makes sure that input population produce almost the same firing rate as the poisson input


        # we have to chose only 10% random connections in the beginning. I have created weight matrices for that

        # input population to excitatory neurons.

        for j in range(0,1600):
            pos_inp_e =  np.random.randint(0,1600,160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_inp_e:
                self.wt_inp_e[i][j] = 200. * np.random.random()   # weights have to be between 0 and 1
        nest.Connect(self.nodes_inp, self.nodes_e, conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_inp_e , 'delay': 1.0, 'tau_plus': 15, 'Wmax':1.0})


        #input population to inhibitory neurons.

        for j in range(0, 1600):
            pos_inp_i = np.random.randint(0, 400,40)  # assigns random 40 positions between [0 and 400). Acts as a column of connection matrix
            for i in pos_inp_i:
                self.wt_inp_i[i][j] = 200. * (1./5)*np.random.random()  # weights have to be between 0 and 0.2
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
        self.old_spikes = self.new_spikes
        for i in range(1600):
            self.spikes[i] = len(nest.GetStatus(self.spikedetector, keys="events")[i]['senders'])
        #print ('The number of spikes of different neurons are', self.spikes)


        self.new_spikes = self.spikes
        self.spikes = self.spikes - self.old_spikes
        print ('The total number of new spikes of the population are', np.sum(self.spikes))
        self.firing_rate = self.spikes/(self.record_interval * (10**-3))
        #print ('')
        #print ('The firing rate is:', self.firing_rate)

    def noise_estimation(self):

        self.firing_rate = middlesort(self.firing_rate)
        # popt, pcov = curve_fit(func, self.xdata, self.firing_rate)
        # self.y_fit = func(self.xdata, *popt)
        noise = rmse(self.firing_rate, self.bg_rate)
        print ('The noise is:', noise)
        return noise


    def update_synapses(self):
        # input population to excitatory neurons.

        for j in range(0, 1600):
            for i in range(0,1600):
                if self.wt_inp_e[i][j] < 30.:
                    self.wt_inp_e[i][j] = 0.00
                    pos_add = np.random.randint(0, 1600)  # assigns random position between [0 and 1600)
                    self.wt_inp_e[pos_add][j] = 200. * np.random.random()  # weights have to be between 0 and 1

        # input population to inhibitory neurons.

        for j in range(0, 1600):
            for i in range(0, 400):
                if self.wt_inp_i[i][j] < 30.:
                    self.wt_inp_i[i][j] = 0.00
                    pos_add = np.random.randint(0, 400)  # assigns random position between [0 and 400)
                    self.wt_inp_i[pos_add][j] = 200. * (1./5)* np.random.random()  # weights have to be between 0 and 0.2

        # recurrent connections from excitatory neurons on itself.

        for j in range(0, 1600):
            for i in range(0, 1600):
                if self.wt_e_e[i][j] < 30.:
                    self.wt_e_e[i][j] = 0.00
                    pos_add = np.random.randint(0, 1600)  # assigns random position between [0 and 400)
                    self.wt_e_e[pos_add][j] = 200. *(1./5)* np.random.random()  # weights have to be between 0 and 0.2
            self.wt_e_e[j][j] = 0.0  # a neuron should not be connected to itself in this updating process

        # connections from excitatory population to inhibitory population.

        for j in range(0, 1600):
            for i in range(0, 400):
                if self.wt_e_i[i][j] < 30.:
                    self.wt_e_i[i][j] = 0.00
                    pos_add = np.random.randint(0, 400)  # assigns random position between [0 and 400)
                    self.wt_e_i[pos_add][j] = 200. *(1./5)* np.random.random()  # weights have to be between 0 and 0.2


        # connections from inhibitory population to excitatory population.

        for j in range(0, 400):
            for i in range(0, 1600):
                if self.wt_i_e[i][j] < 30.:
                    self.wt_i_e[i][j] = 0.00
                    pos_add = np.random.randint(0, 1600)  # assigns random position between [0 and 400)
                    self.wt_i_e[pos_add][j] =  200. *np.random.random()  # weights have to be between 0 and 1


        # connections from inhibitory population to inhibitory population.

        for j in range(0, 400):
            for i in range(0, 400):
                if self.wt_i_i[i][j] < 30.:
                    self.wt_i_i[i][j] = 0.00
                    pos_add = np.random.randint(0, 400)  # assigns random position between [0 and 400)
                    self.wt_i_i[pos_add][j] = 200. *(1./2.5)* np.random.random()  # weights have to be between 0 and 0.4
            self.wt_i_i[j][j] = 0.0 # a neuron should not be connected to itself in this updating process


    def simulate(self):
        if nest.NumProcesses() > 1:
            sys.exit("For simplicity, this example only works for a single process.")

        print("Starting simulation")

        sim_steps = numpy.arange(0, self.t_sim, self.record_interval)

        for i, step in enumerate(sim_steps):
            nest.Simulate(self.record_interval)  # simulates the network for the time in the arguments
            self.record_spikes()
            self.noise.append(self.noise_estimation())
            self.update_synapses()
            if i % 20 == 0:  # happens every 20*1000 ms = 20s
                print("Progress: " + str(i / 2) + "%")
        print("Simulation finished successfully")



    def plot_data1(self):
        fig, ax1 = pl.subplots()
        #ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Number of iterations")
        ax1.set_ylabel("Noise value")
        ax1.plot(self.noise, 'm', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdp_basic_Noise_new.eps', format='eps')

    def plot_data2(self):
        fig, ax2 = pl.subplots()
        ax2.set_xlabel("Number of neurons")
        ax2.set_ylabel("Firing rate")
        ax2.plot(self.firing_rate, 'b', label='Noise', linewidth=2.0, linestyle='--')
        ax2.plot(self.bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdp_basic_firingrate.eps', format='eps')

    def plot_data3(self):
        fig, ax3 = pl.subplots()
        ax3.set_xlabel("Number of neurons")
        ax3.set_ylabel("Fitted Firing Rate")
        ax3.plot(self.y_fit, 'b', label='Noise', linewidth=2.0, linestyle='--')
        ax3.plot(self.bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdp_basic_fittedfiringrate.eps', format='eps')


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
