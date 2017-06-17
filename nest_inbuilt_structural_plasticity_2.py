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
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) + max(0, o_noise)


def rmse(predictions, targets):  #root mean square error
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



class StructralPlasticityClass:
    def __init__(self):

        # simulated time (ms)
        self.t_sim = 200000.0
        # simulation step (ms).
        self.dt = 0.5
        #input population
        self.number_input_neurons = 1600

        #computation population
        self.number_excitatory_neurons = 1600
        self.number_inhibitory_neurons = 400

        # Structural_plasticity properties
        self.update_interval = 1000
        self.record_interval = 1000.0

        # rate of background Poisson input. I have to make it a guassian over 1600 units
        self.xdata = np.linspace(0, 1599, 1600) / 1600
        self.sigma_input = 0.0833
        self.mu_input = 0.5
        self.bg_rate = 20 * gaussian(self.xdata, self.mu_input, self.sigma_input)

        # will contain firing rate of excitatory neurons with time
        self.spikes = np.zeros(1600)
        self.old_spikes = np.zeros(1600)
        self.new_spikes = np.zeros(1600)
        self.firing_rate = np.zeros(1600)

        # connections
        self.wt_inp_e = np.zeros((self.number_excitatory_neurons, self.number_input_neurons))
        self.wt_inp_i = np.zeros((self.number_inhibitory_neurons, self.number_input_neurons))
        self.wt_e_e = np.zeros((self.number_excitatory_neurons, self.number_excitatory_neurons))
        self.wt_e_i = np.zeros((self.number_inhibitory_neurons, self.number_excitatory_neurons))
        self.wt_i_e = np.zeros((self.number_excitatory_neurons, self.number_inhibitory_neurons))
        self.wt_i_i = np.zeros((self.number_inhibitory_neurons, self.number_inhibitory_neurons))

        # will contain the noise
        self.noise = []

        # counter to execute initial condition for mean only once
        self.counter = 0
        self.mu0 = []

        # excitatory and inhibitory leaky integrate-and-fire neurons.
        self.neuron_model = 'iaf_neuron'  # I have to keep simple iaf neurons

        # Excitatory synaptic elements of excitatory neurons
        self.growth_curve_e = {
            'growth_curve': "gaussian",
            'growth_rate': 0.001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Ca2+
            'eps': 0.04,  # Ca2+
        }

        #Excitatory synaptic elements of input neurons
        self.growth_curve_inp = {
            'growth_curve': "gaussian",
            'growth_rate': 0.000001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Ca2+
            'eps': 0.04,  # Ca2+. It was chosen after I saw that Ca is becoming stagnant at this value
        }

        self.model_params_ex = {'tau_m': 20.0,  # membrane time constant (ms)
                                'tau_syn': 0.5,  # excitatory synaptic time constant (ms)
                                't_ref': 2.0,  # absolute refractory period (ms)
                                'E_L': -65.0,  # resting membrane potential (mV)
                                'V_th': -52.0,  # spike threshold (mV)
                                'C_m': 250.0,  # membrane capacitance (pF)
                                'V_reset': -65.0,  # reset potential (mV)
                                'tau_minus': 30.0  # used for stdp
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
        self.mean_ca_inp = []
        self.total_connections_e = []
        self.total_connections_inp = []


    def prepare_simulation(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus(
            {
                'resolution': self.dt
            }
        )

        nest.SetStructuralPlasticityStatus({
            'structural_plasticity_update_interval': self.update_interval,
        })

        nest.CopyModel('stdp_synapse', 'synapse_inp_ex')
        nest.SetDefaults('synapse_inp_ex', {'weight': 500. * np.random.random(), 'delay': 1.0, "tau_plus": 15.0})

        nest.CopyModel('stdp_synapse', 'synapse_ex_ex')
        nest.SetDefaults('synapse_ex_ex', {'weight': 500. * 0.2 * np.random.random(), 'delay': 1.0, "tau_plus": 15.0})

        #the structural plascticity happens only at excitatory synapses. I don't have to connect neurons for such synapses
        nest.SetStructuralPlasticityStatus({
            'structural_plasticity_synapses': {
                'synapse_inp_ex': {
                    'model': 'synapse_inp_ex',
                    'post_synaptic_element': 'Den_ex',
                    'pre_synaptic_element': 'Axon_inp',
                },
                'synapse_ex_ex': {
                    'model': 'synapse_ex_ex',
                    'post_synaptic_element': 'Den_ex',
                    'pre_synaptic_element': 'Axon_ex',
                },
            }
        })

    def create_nodes(self):
        synaptic_elements_inp = {
            'Axon_inp': self.growth_curve_inp,
        }

        synaptic_elements_e = {
            'Den_ex': self.growth_curve_e,
            'Axon_ex': self.growth_curve_e,
        }


        self.nodes_inp = nest.Create('iaf_neuron', self.number_input_neurons)

        self.nodes_e = nest.Create('iaf_neuron', self.number_excitatory_neurons, params = self.model_params_ex)

        self.nodes_i = nest.Create('iaf_neuron', self.number_inhibitory_neurons, params = self.model_params_in)

        # tau_minus is needed for stdp synapses
        nest.SetStatus(self.nodes_e, {'synaptic_elements': synaptic_elements_e})
        nest.SetStatus(self.nodes_inp, {'synaptic_elements': synaptic_elements_inp})


    def connections(self):
        poisson = nest.Create('poisson_generator', 1600)   # we have to create many poisson generators because firing rate to each input neuron has to be different
        nest.SetStatus(poisson, "rate", self.bg_rate)  # I hope this notation works
        nest.Connect(poisson, self.nodes_inp, "one_to_one", syn_spec = {'weight': 1500})


        # we have to chose only 10% random connections in the beginning. I have created weight matrices for that

        #input population to excitatory neurons. Structual enabled
        for j in range(0,1600):
            pos_inp_e = np.random.randint(0,1600,160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_inp_e:
                self.wt_inp_e[i][j] = 500. * np.random.random()   # weights have to be between 0 and 1
        nest.Connect(self.nodes_inp, self.nodes_e, conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'synapse_inp_ex', 'weight': self.wt_inp_e , 'delay': 1.0, 'tau_plus': 15})


        #input population to inhibitory neurons. Simple stdp
        for j in range(0, 1600):
            pos_inp_i = np.random.randint(0, 400,40)  # assigns random 40 positions between [0 and 400). Acts as a column of connection matrix
            for i in pos_inp_i:
                self.wt_inp_i[i][j] = 500. * (1/5)*np.random.random()  # weights have to be between 0 and 0.2
        nest.Connect(self.nodes_inp, self.nodes_i,conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_inp_i , 'delay': 1.0, 'tau_plus': 15})


        #recurrent connections from excitatory neurons on itself. Structural
        for j in range(0, 1600):
            pos_e_e = np.random.randint(0, 1600,160)
            for i in pos_e_e:
                self.wt_e_e[i][j] = 500. * (1 / 5) * np.random.random()  # weights have to be between 0 and 0.2
            self.wt_e_e[j][j] = 0.0   #prevent from connecting a neuron to itself in case it has randomly been assigned
        nest.Connect(self.nodes_e, self.nodes_e, conn_spec = {'rule': 'all_to_all','autapses': False }, syn_spec = {'model': 'synapse_ex_ex', 'weight': self.wt_e_e , 'delay': 1.0, 'tau_plus': 15})


        # connections from excitatory population to inhibitory population. Simple stdp
        for j in range(0, 1600):
            pos_e_i = np.random.randint(0, 400, 40)
            for i in pos_e_i:
                self.wt_e_i[i][j] = 500. * (1 / 5) * np.random.random()  # weights have to be between 0 and 0.2
        nest.Connect(self.nodes_e, self.nodes_i, conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_e_i , 'delay': 1.0, 'tau_plus': 15})

        # connections from inhibitory population to excitatory population. Simple stdp
        for j in range(0, 400):
            pos_i_e = np.random.randint(0, 1600, 160)
            for i in pos_i_e:
                self.wt_i_e[i][j] = 500. * np.random.random()  # weights have to be between 0 and 1
        nest.Connect(self.nodes_i, self.nodes_e, conn_spec = {'rule': 'all_to_all'}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_i_e , 'delay': 1.0, 'tau_plus': 15})

        # connections from inhibitory population to inhibitory population. Simple stdp
        for j in range(0, 400):
            pos_i_i = np.random.randint(0, 400, 40)
            for i in pos_i_i:
                self.wt_i_i[i][j] = 500. * (1 / 2.5) * np.random.random()  # weights have to be between 0 and 0.4
            self.wt_i_i[j][j] = 0.0  # prevent from connecting a neuron to itself in case it has randomly been assigned
        nest.Connect(self.nodes_i, self.nodes_i, conn_spec = {'rule': 'all_to_all','autapses': False}, syn_spec = {'model': 'stdp_synapse', 'weight': self.wt_i_i , 'delay': 1.0, 'tau_plus': 15})

        # to record the spikes from the excitaory neurosn of computation population.
        self.spikedetector = nest.Create("spike_detector", 1600, params={"withgid": True, "withtime": True})
        nest.Connect(self.nodes_e, self.spikedetector, 'one_to_one')

    #it would only record the connections formed by synaptic elements, not the fixed connections
    def record_connectivity(self):
        syn_elems_e = nest.GetStatus(self.nodes_e, 'synaptic_elements')
        syn_elems_inp = nest.GetStatus(self.nodes_inp, 'synaptic_elements')
        self.total_connections_e.append(sum(neuron['Axon_ex']['z_connected'] for neuron in syn_elems_e))
        self.total_connections_inp.append(sum(neuron['Axon_inp']['z_connected'] for neuron in syn_elems_inp))

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

    def record_ca(self):
        ca_e = nest.GetStatus(self.nodes_e, 'Ca'),  # Calcium concentration
        self.mean_ca_e.append(numpy.mean(ca_e))

        ca_inp = nest.GetStatus(self.nodes_inp, 'Ca'),  # Calcium concentration
        self.mean_ca_inp.append(numpy.mean(ca_inp))

        print ("The input mean Ca is", numpy.mean(ca_inp), "and excitatory mean Ca is:", numpy.mean(ca_e))


    def noise_estimation(self):
        self.firing_rate = middlesort(self.firing_rate)
        # popt, pcov = curve_fit(func, self.xdata, self.firing_rate)
        # self.y_fit = func(self.xdata, *popt)
        noise = rmse(self.firing_rate, self.bg_rate)
        print ('The noise is:', noise)
        return noise


    def simulate(self):
        if nest.NumProcesses() > 1:
            sys.exit("For simplicity, this example only works for a single process.")
        nest.EnableStructuralPlasticity()
        print("Starting simulation")
        sim_steps = numpy.arange(0, self.t_sim, self.record_interval)


        for i, step in enumerate(sim_steps):
            nest.Simulate(self.record_interval)
            self.record_ca()
            self.record_connectivity()
            self.record_spikes()
            self.noise.append(self.noise_estimation())

            if i % 20 == 0:
                print("Progress: " + str(i / 2) + "%")

        print("Simulation finished successfully")


    def plot_data(self):
        fig, ax1 = pl.subplots()
        ax1.axhline(self.growth_curve_e['eps'], linewidth=4.0, color='#9999FF')
        ax1.plot(self.mean_ca_e, 'b', label='Ca Concentration Excitatory Neurons', linewidth=2.0)
        ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Time in [s]")
        ax1.set_ylabel("Ca concentration")
        ax2 = ax1.twinx()
        ax2.plot(self.total_connections_e, 'm', label='Excitatory connections', linewidth=2.0, linestyle='--')
        ax2.plot(self.total_connections_inp, 'k', label='Inhibitory connections', linewidth=2.0, linestyle='--')
        ax2.set_ylim([0, 2500])
        ax2.set_ylabel("Connections")
        ax1.legend(loc=1)
        ax2.legend(loc=4)
        pl.savefig('StructuralPlasticityExample.eps', format='eps')

    def plot_data1(self):
        fig, ax1 = pl.subplots()
        #ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Number of iterations")
        ax1.set_ylabel("Noise value")
        ax1.plot(self.noise, 'm', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdp+nest+SP+Noise.eps', format='eps')


    def plot_data2(self):
        fig, ax2 = pl.subplots()
        ax2.set_xlabel("Number of neurons")
        ax2.set_ylabel("Firing rate")
        ax2.plot(self.firing_rate, 'b', label='Noise', linewidth=2.0, linestyle='--')
        ax2.plot(self.bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
        pl.savefig('stdp+nest+SP+firingrate.eps', format='eps')
    #
    # def plot_data3(self):
    #     fig, ax3 = pl.subplots()
    #     ax3.set_xlabel("Number of neurons")
    #     ax3.set_ylabel("Fitted Firing Rate")
    #     ax3.plot(self.y_fit, 'b', label='Noise', linewidth=2.0, linestyle='--')
    #     ax3.plot(self.bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
    #     pl.savefig('stdp+nest+SP+fittedfiringrate.eps', format='eps')

if __name__ == '__main__':
    SP = StructralPlasticityClass()
    # Prepare simulation
    SP.prepare_simulation()
    SP.create_nodes()
    SP.connections()
    # Start simulation
    SP.simulate()
    SP.plot_data()
    SP.plot_data1()
    SP.plot_data2()
    SP.plot_data3()
    # creating a population of inhibitory and excitatory neurons
