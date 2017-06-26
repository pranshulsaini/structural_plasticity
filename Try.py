import math
import cmath
import matplotlib.pyplot as pl
import sys
from scipy.optimize import curve_fit
import numpy as np
import pylab
from brian2 import *




#############FUNCTIONS#######################

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


def middlesort(foo): # I don't rememmber now where it will be used
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


def simulate(t_sim, record_interval, input_change_interval, xdata, sigma_input,bg_rate, firing_rate, spikes, spikedetector,new_spikes, old_spikes, noise):
    print("Starting simulation")

    sim_steps = numpy.arange(0, t_sim, record_interval)  # for training phase
    j = 0
    for i, step in enumerate(sim_steps):
        run(input_change_interval)  # simulates the network for the time in the arguments. The argument here is for the training phase
        record_spikes(spikes, spikedetector, new_spikes, old_spikes, firing_rate, record_interval)
        noise.append(noise_estimation(firing_rate, bg_rate))
        if i % 20 == 0:  # happens every 20*1000 ms = 20s
            print("Progress: " + str(i / 2) + "%")

        if (j == 9):         # happens after 10 * 25 ms = 250  ms        Input is changed now
            bg_rate = 20. * wrap_gaussian(xdata, np.random.random(), sigma_input )
            plot_data1(step, noise)
            plot_data2(step, firing_rate, bg_rate)
            j = -1   # because 1 will be added to it in the next line

        j = j +1
    print("Simulation finished successfully")


def record_spikes(spikes, spikedetector, new_spikes, old_spikes, firing_rate, record_interval):
    for i in range(1600):
        spikes[i] = len(spikedetector.spike_trains()[i])   # stores the times of the spikes of neuron with index i

    new_spikes = spikes
    spikes = spikes - old_spikes
    firing_rate = spikes/(record_interval * (10**-3))
    print('The average firing rate in', record_interval, 'ms of the population are',(np.sum(firing_rate))/160)
    print ('')
    #print('The Poisson input is',spikedetector.spike_trains()[80])
    old_spikes = new_spikes
    #print ('The firing rate is:', self.firing_rate)


def noise_estimation(firing_rate, bg_rate):
    firing_rate = middlesort(firing_rate)
    #popt, pcov = curve_fit(func, self.xdata, self.firing_rate)
    #self.y_fit = func(self.xdata, *popt)
    noise = rmse(firing_rate, bg_rate)
    print ('The noise is:', noise)
    return noise





def plot_data1(step,noise):
    fig, ax1 = pl.subplots()
    #ax1.set_ylim([0, 0.275])
    ax1.set_xlabel("Number of iterations")
    ax1.set_ylabel("Noise value")
    ax1.plot(noise, 'm', label='Noise', linewidth=2.0, linestyle='--')
    pl.savefig('stdpNoise'+str(step) +'ms.eps', format='eps')


def plot_data2(step, firing_rate, bg_rate):
    fig, ax2 = pl.subplots()
    ax2.set_xlabel("Number of neurons")
    ax2.set_ylabel("Firing rate")
    ax2.plot(firing_rate, 'b', label='Noise', linewidth=2.0, linestyle='--')
    ax2.plot(bg_rate, 'r', label='Noise', linewidth=2.0, linestyle='--')
    pl.savefig('stdpfiringrate'+str(step) +'ms.eps', format='eps')





###################################FUNCTIONS END###################



# simulated time (ms)
t_sim = 1000*250.0 *ms    # This training case. 1000 examples, each for 250ms. It would 200000 in test case
# simulation step (ms).
dt = 0.5 * ms
#record time
record_interval = 1000. *ms  # let's take 10 recordings per example

#input change
input_change_interval = 250.0 * ms

#input population
number_input_neurons = 1600

#computation population
number_excitatory_neurons = 1600
number_inhibitory_neurons = 400

# rate of background Poisson input. I have to make it a guassian over 1600 units
xdata = np.linspace(0, 1599, 1600) / 1600
sigma_input = 1./6.
mu_input = 0.5
bg_rate = 20. * wrap_gaussian(xdata, mu_input, sigma_input) *Hz

# will contain firing rate of excitatory neurons with time
spikes = np.zeros(1600)
old_spikes = np.zeros(1600)
new_spikes = np.zeros(1600)
firing_rate = np.zeros(1600)
y_fit = np.zeros(1600)



#will contain the noise
noise = []

#counter to execute initial condition for mean only once
counter = 0
mu0 = []



v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms


#Below are the variables to be used in the equations. I have reduced self.___ because that was causing problems
#I have not put wt_inp_i = 0.15 and wt_e_e = 0.03
v_rest_e = -65 * mV  # using self is causing problems in equations
v_rest_i = -60 * mV
tc_pre_ee = 20 * ms
tc_post_ee = 40 * ms
tc_pre_ie = 20 * ms
tc_post_ie = 20 *ms
nu_pre_ee = 0.0005  # learning rate -- decrease for exc->exc
nu_post_ee = 0.0025  # learning rate -- increase for exc->exc
nu_ie = 0.005  # learning rate -- for inh->exc
alpha_ie = 3 * Hz * tc_post_ie * 2  # controls the firing rate
wmax_ee = 0.5
wmax_ie = 1000.
exp_pre_ee = 0.2
exp_post_ee = exp_pre_ee


neuron_eqs_e ="""
        dv/dt = ((v_rest_e-v) + (I_synE+I_synI) / nS) / (20*ms)  : volt
        dge/dt = -ge/(5.0*ms)                                   : 1
        dgi/dt = -gi/(10.0*ms)                                  : 1
        I_synE = ge * nS * -v                                   : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        """

neuron_eqs_i = """
        dv/dt = ((v_rest_i-v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        dge/dt = -ge/(5.0*ms)                                   : 1
        dgi/dt = -gi/(10.0*ms)                                  : 1
        I_synE = ge * nS * -v                                   : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        """

eqs_stdp_ee = """
        w : 1  
        postbefore                        : 1
        dpre/dt   =   -pre/(tc_pre_ee)       : 1 (event-driven)
        dpost/dt = -post/(tc_post_ee)     : 1 (event-driven)
        """
eqs_stdp_ie = """
        w : 1
        dpre/dt   =  -pre/(tc_pre_ie)        : 1 (event-driven)
        dpost/dt  = -post/(tc_post_ie)       : 1 (event-driven)
        """
# added the conductance change and weight clipping below
eqs_STDP_pre_ee = """
        ge = ge + w
        pre = 1.
        w -= nu_pre_ee * post * w**exp_pre_ee
        w = clip(w, 0, wmax_ee)
        """

eqs_STDP_post_ee = """
        postbefore = post
        w += nu_post_ee * pre * postbefore * (wmax_ee - w)**exp_post_ee
        post = 1.
        w = clip(w, 0, wmax_ee)
        """
#The equations below were not mentioned in the paper. Robin also does not know the reference
eqs_STDP_pre_ie ="""
        gi =  gi + w
        pre += 1.
        w += nu_ie * (post-alpha_ie)
        w = clip(w, 0, wmax_ie)
        """

eqs_STDP_post_ie ="""
        post += 1.
        w += nu_ie * pre
        w = clip(w, 0, wmax_ie)
        """


defaultclock.dt = dt


#creation of nodes
nodes_inp = PoissonGroup(number_input_neurons, rates = bg_rate)
nodes_e = NeuronGroup(number_excitatory_neurons, neuron_eqs_e, threshold= 'v>v_thresh_e', refractory= refrac_e, reset= 'v = v_reset_i')
nodes_i = NeuronGroup(number_inhibitory_neurons, neuron_eqs_i, threshold= 'v>v_thresh_i', refractory= refrac_i, reset= 'v = v_reset_i')

#connections. we have to chose only 10% random connections in the beginning. I have created weight matrices for that

# input population to excitatory neurons.
S_inp_e = Synapses(nodes_inp, nodes_e, model=eqs_stdp_ee, on_pre=eqs_STDP_pre_ee, on_post = eqs_STDP_post_ee)
S_inp_e.connect()  # all to all connections except to itself
for j in range(0,1600):
    pos_inp_e = np.random.randint(0,1600,160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
    delay_inp_e = (10 * ms - 0 * ms) * np.random.random(160) + 0 * ms
    wt_inp_e = np.random.rand(160)        # unformly distributed random numbers between 0 and 1
    k = 0
    for i in pos_inp_e:
        S_inp_e.w[i,j] = wt_inp_e[k]   # weights have to be between 0 and 1
        S_inp_e.delay[i,j] =  delay_inp_e[k]
        k = k+1

#input population to inhibitory neurons.
S_inp_i = Synapses(nodes_inp, nodes_i, model=eqs_stdp_ee, on_pre=eqs_STDP_pre_ee,  on_post = eqs_STDP_post_ee)
S_inp_i.connect()  # all to all connections except to itself
for j in range(0, 400):
    pos_inp_i = np.random.randint(0, 1600, 160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
    delay_inp_i = (5 * ms - 0 * ms) * np.random.random(160) + 0 * ms
    wt_inp_i = (1./5)*np.random.rand(160)  # unformly distributed random numbers between 0 and 0.2
    k = 0
    for i in pos_inp_i:
        S_inp_i.w[i,j] = wt_inp_i[k]  # weights have to be between 0 and 1
        S_inp_i.delay[i,j] = delay_inp_i[k]
        k = k + 1


#recurrent connections from excitatory neurons on itself.
S_e_e = Synapses(nodes_e, nodes_e, model=eqs_stdp_ee, on_pre=eqs_STDP_pre_ee,  on_post = eqs_STDP_post_ee)
S_e_e.connect('i !=j')  # all to all connections except to itself, Still I can go upon [1599,1599] but just can't change the diagonal elements
for j in range(0, 1600):
    pos_e_e = np.random.randint(0, 1600, 160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
    delay_e_e = (5 * ms - 0 * ms) * np.random.random(160) + 0 * ms
    wt_e_e = (1./5)*np.random.rand(160)  # unformly distributed random numbers between 0 and 1
    k = 0
    for i in pos_e_e:
        if (i == j):  # to avoid the recurrent connections. There will be no error but one weight will get waste
            i = i +1
        S_e_e.w[i,j] = wt_e_e[k]  # weights have to be between 0 and 1
        S_e_e.delay[i,j] = delay_e_e[k]
        k = k + 1


# connections from excitatory population to inhibitory population.
S_e_i = Synapses(nodes_e, nodes_i, model=eqs_stdp_ee, on_pre=eqs_STDP_pre_ee,  on_post = eqs_STDP_post_ee)
S_e_i.connect()  # all to all connections
for j in range(0, 400):
    pos_e_i = np.random.randint(0, 1600, 160)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
    delay_e_i = (2 * ms - 0 * ms) * np.random.random(160) + 0 * ms
    wt_e_i = (1./5)*np.random.rand(160)  # unformly distributed random numbers between 0 and 1
    k = 0
    for i in pos_e_i:
        S_e_i.w[i,j] = wt_e_i[k]  # weights have to be between 0 and 1
        S_e_i.delay[i,j] = delay_e_i[k]
        k = k + 1

# connections from inhibitory population to excitatory population.
S_i_e = Synapses(nodes_i, nodes_e, model=eqs_stdp_ie, on_pre=eqs_STDP_pre_ie,  on_post = eqs_STDP_post_ie)
S_i_e.connect()  # all to all connections
for j in range(0, 1600):
    pos_i_e = np.random.randint(0, 400, 40)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
    delay_i_e = (1 * ms - 0 * ms) * np.random.random(40) + 0 * ms
    wt_i_e = np.random.rand(40)  # unformly distributed random numbers between 0 and 1
    k = 0
    for i in pos_i_e:
        S_i_e.w[i,j] = wt_i_e[k]  # weights have to be between 0 and 1
        S_i_e.delay[i,j] = delay_i_e[k]

 # connections from inhibitory population to inhibitory population.
S_i_i = Synapses(nodes_i, nodes_i, model=eqs_stdp_ie, on_pre= eqs_STDP_pre_ie,  on_post = eqs_STDP_post_ie)
S_i_i.connect('i !=j')  # all to all connections
for j in range(0, 400):
    pos_i_i = np.random.randint(0, 400, 40)  # assigns random 160 positions between [0 and 1599). Acts as a column of connection matrix
    delay_i_i = (5 * ms - 0 * ms) * np.random.random(40) + 0 * ms
    wt_i_i =  (1 / 2.5) * np.random.rand(40)  # unformly distributed random numbers between 0 and 1
    k = 0
    for i in pos_i_i:
        if (i == j):
            i = i +1
        S_i_e.w[i,j] = wt_i_i[k]  # weights have to be between 0 and 1
        S_i_e.delay[i,j] = delay_i_i[k]

# to record the spikes from the excitaory neurosn of computation population.
spikedetector = SpikeMonitor(nodes_e)


# Start simulation
simulate(t_sim, record_interval, input_change_interval, xdata, sigma_input,bg_rate, firing_rate, spikes, spikedetector,new_spikes, old_spikes, noise)


#plotting
step = 'end'
plot_data1(step, noise)
plot_data2(step, firing_rate, bg_rate)


