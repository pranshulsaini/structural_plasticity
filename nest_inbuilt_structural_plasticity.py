# nest structural plasticity example

import nest
import numpy
import matplotlib.pyplot as pl
import sys


class StructuralPlasticityExample:
    def __init__(self):
        #similation time (in ms)
        self.t_sim = 200000.0
        #simulation step (ms)
        self.dt = 0.1
        self.number_ex_neurons = 800
        self.number_in_neurons = 200

        #Structural plasticity properties
        self.update_interval = 1000    # to update the connectivity
        self.record_interval = 1000.0    # to record calcium concentration

        # rate of background poisson input
        self.bg_rate = 10000.0     # what is background here?
        self.neuron_model = 'iaf_psc_exp'

        #Here we define growth curves which is the growth of synaptic elements with the concentration of Ca

        #excitatory synaptic elements of excitatory neurons
        self.growth_curve_e_e = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0001, # elements/s   (nu)
            'continuous': False,
            'eta': 0.0, # min level of Ca concentration to generate synapses
            'eps': 0.05 # desired Ca concentration
        }

        # Inhibitory synaptic elements of excitatory neurons
        self.growth_curve_e_i = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Ca2+
            'eps': self.growth_curve_e_e['eps'],  # Ca2+
        }

        # Excitatory synaptic elements of inhibitory neurons
        self.growth_curve_i_e = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0004,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Ca2+
            'eps': 0.2,  # Ca2+
        }

        # Inhibitory synaptic elements of inhibitory neurons
        self.growth_curve_i_i = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Ca2+
            'eps': self.growth_curve_i_e['eps']  # Ca2+
        }


        #Now we specify the neuron model

        self.model_params = {'tau_m': 10.0,  # membrane time constant
                             'tau_syn_ex': 0.5,  # excitatory synaptic time constant (ms)
                             'tau_syn_in': 0.5,  # inhibitory synaptic time constant (ms)
                             't_ref': 2.0,  # absolute refractory period (ms)
                             'E_L': -65.0,  # resting membrane potential (mV)
                             'V_th': -50.0,  # spike threshold (mV)
                             'C_m': 250.0,  # membrane capacitance (pF)
                             'V_reset': -65.0,  # reset potential (mV)
        }

        self.nodes_e = None
        self.nodes_i = None
        self.mean_ca_e = []
        self.mean_ca_i = []
        self.total_connections_e = []
        self.total_connections_i = []

        #Initialize the variable for post synaptic currents.
        self.psc_e = 585.0  #caused by excitatory synapse for 1 mV post synaptic potential
        self.psc_i = -585.0  #caused by inhibitory synapse for -1 mV post synaptic potential
        self.psc_ext = 6.2  # caused by external synapse for 0.11 mV post synaptic potential


    #Assigning the growth curves to the corresponding synaptic elements
    def create_nodes(self):
        nest.synaptic_elements = {
            'Den_ex': self.growth_curve_e_e,
            'Den_in': self.growth_curve_e_i,
            'Axon_ex': self.growth_curve_e_e,
        }

        nest.synaptic_elements_i = {
            'Den_ex': self.growth_curve_i_e,
            'Den_in': self.growth_curve_i_i,
            'Axon_in': self.growth_curve_i_i,
        }


    def prepare_simulation(self):
        nest.ResetKernel()   #what does it do?
        nest.set_verbosity('M_ERROR')  # what does it do?
        nest.SetKernelStatus({'resolution': self.dt})
        nest.SetStructuralPlasticityStatus({'structural_plasticity_update_interval': self.update_interval})
        nest.CopyModel('stdp_synapse', 'synapse_ex')
        nest.SetDefaults('synapse_ex', {'delay': 1.0,"tau_plus": 15.0 })
        nest.CopyModel('stdp_synapse', 'synapse_in')
        nest.SetDefaults('synapse_in', {'delay': 1.0, "tau_plus": 15.0})
        nest.SetStructuralPlasticityStatus({
                'structural_plasticity_synapses':{

                    'synapse_ex':{
                        'model': 'synapse_ex',
                        'post_synaptic_element': 'Den_ex',
                        'pre_synaptic_element': 'Axon_ex',
                    },

                    'synapse_in':{
                        'model': 'synapse_in',
                        'post_synaptic_element': 'Den_in',
                        'pre_synaptic_element': 'Axon_in',
                    },

                }
        })



    #Creating a population with 80% excitatory neurons and 20% inhibitory neurons


        self.nodes_e = nest.Create('iaf_neuron', self.number_ex_neurons, {'synaptic_elements': nest.synaptic_elements})
        self.nodes_i = nest.Create('iaf_neuron', self.number_in_neurons, {'synaptic_elements': nest.synaptic_elements_i})

        nest.SetStatus(self.nodes_e, {'synaptic_elements': nest.synaptic_elements, "tau_minus": 30.0})  # what is the need?
        nest.SetStatus(self.nodes_i, {'synaptic_elements': nest.synaptic_elements_i, "tau_minus": 30.0})  #what is the need?

        wt_e_i = numpy.zeros = ((self.number_in_neurons, self.number_ex_neurons))
        for j in range(0, 800):
            pos_e_i = numpy.random.randint(0, 200,
                                          100)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_e_i:
                wt_e_i[i][j] = numpy.random.random()  # weights have to be between 0 and 1
        nest.Connect(self.nodes_e, self.nodes_i, {'weight': wt_e_i, 'delay': 1.0})

        wt_e_e = numpy.zeros = ((self.number_ex_neurons, self.number_ex_neurons))
        for j in range(0, 800):
            pos_e_e = numpy.random.randint(0, 800, 400)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_e_e:
                wt_e_e[i][j] = numpy.random.random()  # weights have to be between 0 and 1
        nest.Connect(self.nodes_e, self.nodes_i, {'weight': wt_e_e, 'delay': 1.0})

        wt_i_e = numpy.zeros = ((self.number_ex_neurons, self.number_in_neurons))
        for j in range(0, 200):
            pos_i_e = numpy.random.randint(0, 800, 400)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_i_e:
                wt_i_e[i][j] = numpy.random.random()  # weights have to be between 0 and 1
        nest.Connect(self.nodes_e, self.nodes_i, {'weight': wt_i_e, 'delay': 1.0})

        wt_i_i = numpy.zeros = ((self.number_in_neurons, self.number_in_neurons))
        for j in range(0, 200):
            pos_i_i = numpy.random.randint(0, 200, 100)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
            for i in pos_i_i:
                wt_i_i[i][j] = numpy.random.random()  # weights have to be between 0 and 1
        nest.Connect(self.nodes_e, self.nodes_i, {'weight': wt_i_i, 'delay': 1.0})



    def connect_ext_input(self):
        noise = nest.Create('poisson_generator')
        nest.SetStatus(noise, {'rate':self.bg_rate})
        nest.Connect(noise, self.nodes_e, 'all_to_all', {'weight':self.psc_ext, 'delay':1.0}) # would it send the spikes to all
        #neurons at the same time?
        nest.Connect(noise, self.nodes_i, 'all_to_all', {'weight': self.psc_ext, 'delay': 1.0})


    def record_ca(self):
        ca_e = nest.GetStatus(self.nodes_e,'Ca') # calcium concentration
        self.mean_ca_e.append(numpy.mean(ca_e))

        ca_i = nest.GetStatus(self.nodes_i, 'Ca'),  # Calcium concentration
        self.mean_ca_i.append(numpy.mean(ca_i))


    #record the connectivity
    #the total excitatory connections are equal to connected excitatory presynaptic elements
    def record_connectivity(self):
        syn_elems_e = nest.GetStatus(self.nodes_e, 'synaptic_elements')
        syn_elems_i = nest.GetStatus(self.nodes_i, 'synaptic_elements')
        self.total_connections_e.append(sum(neuron['Axon_ex']['z_connected'] for neuron in syn_elems_e))
        self.total_connections_i.append(sum(neuron['Axon_in']['z_connected'] for neuron in syn_elems_i))

    #a function to plot the recorded values at the end of the simulation.

    def plot_data(self):
        fig, ax1 = pl.subplots()
        ax1.axhline(self.growth_curve_e_e['eps'], linewidth=4.0, color='#9999FF')
        ax1.plot(self.mean_ca_e, 'b', label='Ca Concentration Excitatory Neurons', linewidth=2.0)
        ax1.axhline(self.growth_curve_i_e['eps'], linewidth=4.0, color='#FF9999')
        ax1.plot(self.mean_ca_i, 'r', label='Ca Concentration Inhibitory Neurons', linewidth=2.0)
        ax1.set_ylim([0, 0.275])
        ax1.set_xlabel("Time in [s]")
        ax1.set_ylabel("Ca concentration")
        ax2 = ax1.twinx()
        ax2.plot(self.total_connections_e, 'm', label='Excitatory connections', linewidth=2.0, linestyle='--')
        ax2.plot(self.total_connections_i, 'k', label='Inhibitory connections', linewidth=2.0, linestyle='--')
        ax2.set_ylim([0, 2500])
        ax2.set_ylabel("Connections")
        ax1.legend(loc=1)
        ax2.legend(loc=4)
        pl.savefig('StructuralPlasticityExample.eps', format='eps')



    #performing the simulation
    def simulate(self):
        if nest.NumProcesses() > 1:
            sys.exit("For simplicity, this example only works for a single process.")
        nest.EnableStructuralPlasticity()
        print("Starting simulation")
        sim_steps = numpy.arange(0, self.t_sim, self.record_interval)
        for i, step in enumerate(sim_steps):
            nest.Simulate(self.record_interval)   #simulates the network for the time in the arguments
            self.record_ca()
            self.record_connectivity()
            if i % 20 == 0:       # happens every 20*1000 ms = 20s
                print("Progress: " + str(i / 2) + "%")
        print("Simulation finished successfully")



if __name__ == '__main__':
    example = StructuralPlasticityExample()
    example.create_nodes()
    # Prepare simulation
    example.prepare_simulation()
    example.connect_ext_input()
    # Start simulation
    example.simulate()
    example.plot_data()







