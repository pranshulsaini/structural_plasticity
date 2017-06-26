def wrap_gaussian(x, mu, sig):
    if mu == 0.5:
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    elif mu > 0.5:
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        mu_new = mu - 1.0
        mark = int(abs(160*(mu - 0.5)))
        gauss1 = np.exp(-np.power(x - mu_new, 2.) / (2 * np.power(sig, 2.)))
        for i in range(0, mark):
            gauss[i]= gauss1[i]
    else:
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        mu_new = mu + 1.0
        mark = int(abs(160 - 160 * abs((mu - 0.5))))
        gauss1 = np.exp(-np.power(x - mu_new, 2.) / (2 * np.power(sig, 2.)))
        for i in range(mark, 160):
            gauss[i] = gauss1[i]

    return gauss


from brian2 import *

set_device('cpp_standalone', directory='STDP_standalone')

N = 160
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
xdata = np.linspace(0, 159, 160) / 160
sigma_input = 1./6.
mu_input = 0.5
bg_rate = 20. * wrap_gaussian(xdata,mu_input, sigma_input) *Hz

#F = 15*Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

input = PoissonGroup(N, rates=bg_rate)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='linear')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S.connect()
S.w = 'rand() * gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)

run(1*second, report='text')


print ('The spikes are',s_mon.spike_trains()[80])
subplot(311)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(312)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(313)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
show()

