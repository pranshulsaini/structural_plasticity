'''
Created on 05.11.2012

@author: peter
'''
#------------------------------------------------------------------------------ 
# imports and brian options
#------------------------------------------------------------------------------ 

import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging

import brian as b
from brian import *
defaul_clock_time = 0.5 * b.ms
# from brian.globalprefs import *
b.globalprefs.set_global_preferences( 
                        defaultclock = b.Clock(dt = defaul_clock_time), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave=True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimisations are turned on 
                        #- if you need IEEE guaranteed results, turn this switch off.
                        useweave_linear_diffeq = False,  # Whether to use weave C++ acceleration for the solution of linear differential 
                        #equations. Note that on some platforms, typically older ones, this is faster and on some platforms, 
                        #typically new ones, this is actually slower.
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenreset = True,  # Whether or not to use experimental code generation support on resets. 
                        #Typically slower due to weave overheads, so usually leave this off.
                        usecodegenthreshold = True,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                        openmp = True,  # Whether or not to use OpenMP pragmas in generated C code. 
                        #If supported on your compiler (gcc 4.2+) it will use multiple CPUs and can run substantially faster.
                        magic_useframes = True,  # Defines whether or not the magic functions should search for objects 
                        #defined only in the calling frame or if they should find all objects defined in any frame. 
                        #This should be set to False if you are using Brian from an interactive shell like IDLE or IPython 
                        #where each command has its own frame, otherwise set it to True.
                       ) 
 
 
import numpy as np
import matplotlib
import matplotlib.cm as cm
import time
import scipy 
import scipy.sparse
import os
import brian.experimental.realtime_monitor as rltmMon
# import brian.experimental.cuda.gpucodegen as gpu

import structural_plasticity_evaluate

#------------------------------------------------------------------------------ 
# helper functions
#------------------------------------------------------------------------------     
def create_topo_input(n_e, pop_val, activation_function = None):
#         print 'input', input, 'pop_val', pop_val
    if activation_function == None:
        activation_function = gaussian_1D
    center_ID = int(pop_val*n_e)
    topo_coords = {}
    for i in xrange(n_e):
        pos = 1. * float(i)/n_e
        topo_coords[i] = (0.5,pos)
    center_coords = topo_coords[center_ID]
    dists = np.zeros(n_e)
    
    for i in xrange(n_e):
        coords = topo_coords[i]
        deltaX = abs(coords[0]-center_coords[0])
        deltaY = abs(coords[1]-center_coords[1])
        if deltaX > 0.5: deltaX=1.0-deltaX  # silent assumption: topo is defined in unit square (and fills it)
        if deltaY > 0.5: deltaY=1.0-deltaY  # silent assumption: topo is defined in unit square (and fills it)
        squared_dist = deltaX ** 2  + deltaY  ** 2
        dists[i] = squared_dist
    dists_Ids = zip(dists, range(n_e))
    dists_Ids.sort()
    unused_sorted_dists, dist_sorted_ids = zip(*dists_Ids)
    activity = np.zeros(n_e)
    for i,idx in enumerate(dist_sorted_ids):
        activity[idx] = activation_function(float(i)/n_e)
#        print "Integral over input activity: %f"%np.sum(activity)
    return activity


def compute_pop_vector(pop_array):
    size = len(pop_array)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(pop_array * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos


def gaussian_1D(x):
    return gaussian_peak * (np.exp(-0.5 * (x / gaussian_sigma)**2))


def get_matrix_from_file(filename):
    if filename[-3-4]=='e':
        n_src = n_e
    else:
        n_src = n_i
    if filename[-1-4]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(filename)
    value_arr = np.zeros((n_src, n_tgt))
    value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    
#     print value_arr
#     print filename, n_src, n_tgt
#     figure()
#     im2 = imshow(value_arr, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#     cbar2 = colorbar(im2)
#     title(filename)
#     show()
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for conn_name in connections:
        conn_matrix = connections[conn_name][:]
        conn_list_sparse = ([(i,j[0],j[1]) for i in xrange(conn_matrix.shape[0]) for j in zip(conn_matrix.rowj[i],conn_matrix.rowdata[i])])
    #     print len(conn_list_sparse)
        np.save(target_path + conn_name + ending, conn_list_sparse)


def normalize_weights():
    print 'normalize weights...',
    for conn_name in connections:
        if conn_name[1] == 'e' and conn_name[3] == 'e':
            if conn_name[0] == conn_name[2]:   # ==> recurrent connection
                factor = weight['ee']
            else:   # ==> input connection
                factor = weight['ee_input']
                    
            connection = connections[conn_name][:]
            
            w_pre = np.zeros((n_e, n_e))
            w_post = np.zeros((n_e, n_e))
            w_end = np.zeros((n_e, n_e))
            for i in xrange(n_e):#
                rowi = connection.rowdata[i]
                rowMean = np.mean(rowi)
                w_pre[i, connection.rowj[i]] = rowi
                connection.rowdata[i] *= factor/rowMean
                w_post[i, connection.rowj[i]] = connection.rowdata[i]
                
            col_means = np.sum(w_post, axis = 0)
            col_factors = factor/col_means
            col_data_entries = [len(connection.coldataindices[j]) for j in xrange(n_e)]
#             print conn_name, col_means, col_factors, col_data_entries
            
            for j in xrange(n_e):#
                connection[:,j] *= col_factors[j]*col_data_entries[j]
                
                w_end[j, connection.rowj[j]] = connection.rowdata[j]
                
#             if conn_name == 'AeAe':
#                 figure()
#                 im2 = imshow(w_pre, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#                 cbar2 = colorbar(im2)
#                 title(conn_name + ' pre')
#                 figure()
#                 im2 = imshow(w_post, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#                 cbar2 = colorbar(im2)
#                 title(conn_name + ' post')
#                 figure()
#                 im2 = imshow(w_end, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#                 cbar2 = colorbar(im2)
#                 title(conn_name + ' end')
#                 show()
    print('done')

            
def apply_structural_plasticity_basic(conn_name):
    conn_type = conn_name[1] + conn_name[3]
    mean_weight = weight[conn_type]
    connection = connections[conn_name][:]
    weight_matrix =  np.zeros((n_e, n_e))
    for i in xrange(n_e):#
        weight_matrix[i, connection.rowj[i]] = connection.rowdata[i]
    
    #------------------------------------------------------------------------------ 
    # remove weak synapses and add as many new ones
    #------------------------------------------------------------------------------ 
    valid_entries = np.where(weight_matrix >= mean_weight*synapse_removal_thres)
    num_all_entries = np.count_nonzero(weight_matrix)
    num_invalid_entries = num_all_entries - len(valid_entries[0])
    new_weight_matrix = np.zeros(weight_matrix.shape)
    for i, j in zip(valid_entries[0], valid_entries[1]):
        new_weight_matrix[i, j] = weight_matrix[i, j]
    num_replaced_entries = 0
    while num_replaced_entries < num_invalid_entries:
        idx = (np.int32(np.random.rand()*weight_matrix.shape[0]), np.int32(np.random.rand()*weight_matrix.shape[1]))
        if not (weight_matrix[idx]):
            new_weight_matrix[idx] = mean_weight*synpase_reset_value
            num_replaced_entries += 1
            
#     print weight_matrix
#     print new_weight_matrix
#     print weight_matrix - new_weight_matrix
    print 'nnz:',num_all_entries,'inval:', num_invalid_entries
            
    #------------------------------------------------------------------------------ 
    # create new connections and STDP methods
    #------------------------------------------------------------------------------ 
    net.remove(connections[conn_name], STDP_methods[conn_name])
    del connections[conn_name]
    del STDP_methods[conn_name]
    if conn_name[0:2] in input_groups:
        src_group = input_groups[conn_name[0:2]]
    else:
        src_group = neuron_groups[conn_name[0:2]]
        
    sparse_weight_matrix = scipy.sparse.lil_matrix(new_weight_matrix)
        
    connections[conn_name] = Connection(src_group, neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])
    connections[conn_name].connect(src_group, neuron_groups[conn_name[2:4]], sparse_weight_matrix, delay=delay[conn_type])
    net.add(connections[conn_name])

    STDP_methods[conn_name] =  b.STDP(connections[conn_name], eqs=eqs_stdp_ee, pre = eqs_STDP_pre_ee, 
                                           post = eqs_STDP_post_ee, wmin=0., wmax= wmax_ee)#, clock = b.Clock(defaul_clock_time))
    net.add(STDP_methods[conn_name])

#     if conn_name == 'XeAe':
#         connection = connections[conn_name][:]
#         w_post =  np.zeros((n_e, n_e))
#         for i in xrange(n_e):#
#             w_post[i, connection.rowj[i]] = connection.rowdata[i]
#         b.figure()
#         im2 = b.imshow(weight_matrix, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#         cbar2 = b.colorbar(im2)
#         b.title(conn_name + ' initial')
#         b.figure()
#         im2 = b.imshow(new_weight_matrix, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#         cbar2 = b.colorbar(im2)
#         b.title(conn_name + ' after structure change')
#         b.figure()
#         im2 = b.imshow(w_post, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
#         cbar2 = b.colorbar(im2)
#         b.title(conn_name + ' end')
#         b.show()
    return new_weight_matrix


def apply_structural_plasticity_bookkeeping(conn_name):
    conn_type = conn_name[1] + conn_name[3]
    mean_weight = weight[conn_type]
    connection = connections[conn_name][:]
    weight_matrix =  np.zeros((n_e, n_e))
    for i in xrange(n_e):#
        weight_matrix[i, connection.rowj[i]] = connection.rowdata[i]
    
    #------------------------------------------------------------------------------ 
    # update bookkeeping matrix, remove weak synapses and add as many new ones
    #------------------------------------------------------------------------------ 
    valid_entries = np.where(weight_matrix >= mean_weight*weak_threshold)
    weak_entries = np.where((weight_matrix > 0) & (weight_matrix < mean_weight*weak_threshold))
    invalid_entries = [] #Below threshold for too long
    critical_entries = [] #Below threshold but not yet long enough
    #Update entries for currently weak entries
    print("weak entries: "+str(len(weak_entries[0])))
    for i, j in zip(weak_entries[0], weak_entries[1]):
        weak_matrix[conn_name][i,j] += 2 #Since we decay with 1 in the next step, we add 2 (TODO: this should be solved better)
        if weak_matrix[conn_name][i,j] > weak_clock_threshold:
            invalid_entries.append((i,j))
            weak_matrix[conn_name][i,j] = 0 #Since this synapse will be removed, we can set the bookkeeping entry to 0
        else:
            critical_entries.append((i,j))
    print("critical entries:" +str(len(critical_entries)))
    print("invalid entries:" + str(len(invalid_entries)))
    critical_entries = np.asarray(critical_entries).transpose()
    #slow decay of all bookkeeping values
    weak_nonzero = np.nonzero(weak_matrix[conn_name])
    for i, j in zip(weak_nonzero[0], weak_nonzero[1]):
        weak_matrix[conn_name][i,j] -= 1
    
    num_all_entries = np.count_nonzero(weight_matrix)
    num_invalid_entries = len(invalid_entries)
    new_weight_matrix = np.zeros(weight_matrix.shape)
    for i, j in zip(valid_entries[0], valid_entries[1]):
        new_weight_matrix[i, j] = weight_matrix[i, j]
    if len(critical_entries) > 0:
        for i, j in zip(critical_entries[0], critical_entries[1]):
            new_weight_matrix[i,j] = weight_matrix[i,j]
    num_replaced_entries = 0
    while num_replaced_entries < num_invalid_entries:
        idx = (np.int32(np.random.rand()*weight_matrix.shape[0]), np.int32(np.random.rand()*weight_matrix.shape[1]))
        if not (weight_matrix[idx]):
            new_weight_matrix[idx] = mean_weight*synpase_reset_value
            num_replaced_entries += 1
            
#     print weight_matrix
#     print new_weight_matrix
#     print weight_matrix - new_weight_matrix
    print 'nnz:',num_all_entries,'inval:', num_invalid_entries
            
    #------------------------------------------------------------------------------ 
    # create new connections and STDP methods
    #------------------------------------------------------------------------------ 
    net.remove(connections[conn_name], STDP_methods[conn_name])
    del connections[conn_name]
    del STDP_methods[conn_name]
    if conn_name[0:2] in input_groups:
        src_group = input_groups[conn_name[0:2]]
    else:
        src_group = neuron_groups[conn_name[0:2]]
        
    sparse_weight_matrix = scipy.sparse.lil_matrix(new_weight_matrix)
        
    connections[conn_name] = Connection(src_group, neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])
    connections[conn_name].connect(src_group, neuron_groups[conn_name[2:4]], sparse_weight_matrix, delay=delay[conn_type])
    net.add(connections[conn_name])

    STDP_methods[conn_name] =  b.STDP(connections[conn_name], eqs=eqs_stdp_ee, pre = eqs_STDP_pre_ee, 
                                           post = eqs_STDP_post_ee, wmin=0., wmax= wmax_ee)#, clock = b.Clock(defaul_clock_time))
    net.add(STDP_methods[conn_name])

    return new_weight_matrix




            

def apply_structural_plasticity(conn_name):
    if structural_algorithm == 'basic':
        apply_structural_plasticity_basic(conn_name)
    elif structural_algorithm == 'bookkeeping':
        apply_structural_plasticity_bookkeeping(conn_name)


            
#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
# data_path = '../data/structural_plasticity/'
data_path = os.getcwd()+'/data/structural_plasticity/'
weight_path = data_path +    'learned_weights/'#     'sorted_weights/'#   'random/'#      'weights/'#
target_path = data_path + 'weights/'    
n_e = 1600
n_i = n_e/4
single_example_time =  0.25*b.second #runtime # 
num_examples = 1000
resting_time = 0.0*b.second
runtime = num_examples*(single_example_time+resting_time)
weight_normalization_interval = 20
structural_plasticity_interval = 30 #100 for basic

test_mode = False 
if test_mode:
    record_spikes = True
else:
    record_spikes = False
# record_spikes = True

v_rest_e = -65*b.mV 
v_rest_i = -60*b.mV 
v_reset_e = -65.*b.mV
v_reset_i = -45.*b.mV
v_thresh_e = -52.*b.mV
v_thresh_i = -40.*b.mV
refrac_e = 5.*b.ms
refrac_i = 2.*b.ms

input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
inter_pop_connection_names = []
input_conn_names = ['ee_input', 'ei_input']
recurrent_conn_names = ['ee', 'ei', 'ie', 'ii']
inter_conn_names = ['ee_input', 'ei_input']

conn_structure = 'sparse' # 'dense' 
weight = {}
delay = {}

weight['ee_input'] = 0.15 # 0.10
weight['ee'] = 0.03

delay['ee_input'] = (0*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,5*b.ms)
delay['ee'] = (0*b.ms,5*b.ms)
delay['ei'] = (0*b.ms,2*b.ms)
delay['ie'] = (0*b.ms,1*b.ms)
delay['ii'] = (0*b.ms,2*b.ms)

if test_mode:
    ee_STDP_on = False # True # 
else:
    ee_STDP_on = True
tc_pre_ee = 20*b.ms
tc_post_1_ee = 40*b.ms
TCpost2EE = 40*b.ms
tc_pre_ie = 20*b.ms
tc_post_ie = 20*b.ms
nu_pre_ee =  0.0005      # learning rate -- decrease for exc->exc
nu_post_ee = 0.0025      # learning rate -- increase for exc->exc
nu_ie =      0.005       # learning rate -- for inh->exc
alpha_ie = 3*b.Hz*tc_post_ie*2    # controls the firing rate
wmax_ee = 0.5
wmax_ie = 1000.
exp_pre_ee = 0.2
exp_post_ee = exp_pre_ee

#Parameters for structural plasticity
structural_algorithm = 'bookkeeping'
synapse_removal_thres = 1/20.
synpase_reset_value = 1/20.

weak_threshold = 1/10.
weak_clock_threshold = 3 #How many times may a synapse be under the threshold
weak_matrix = {}
if structural_algorithm == 'bookkeeping':
    weak_matrix['XeAe'] = scipy.sparse.lil_matrix((n_e,n_e),dtype='i')
    weak_matrix['AeAe'] = scipy.sparse.lil_matrix((n_e,n_e),dtype='i')




gaussian_peak = 20
gaussian_sigma = 1./6.

neuron_eqs_e = '''
        dv/dt = ((v_rest_e-v) + (I_synE+I_synI) / nS) / (20*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(5.0*ms)                                   : 1
        dgi/dt = -gi/(10.0*ms)                                  : 1
        '''

neuron_eqs_i = '''
        dv/dt = ((v_rest_i-v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(5.0*ms)                                   : 1
        dgi/dt = -gi/(10.0*ms)                                  : 1
        '''
        
eqs_stdp_ee = '''
            post2before                        : 1.0
            dpre/dt   =   -pre/(tc_pre_ee)       : 1.0
            dpost1/dt = -post1/(tc_post_1_ee)     : 1.0
            dpost2/dt = -post2/(TCpost2EE)     : 1.0
            '''
eqs_stdp_ie = '''
            dpre/dt   =  -pre/(tc_pre_ie)        : 1.0
            dpost/dt  = -post/(tc_post_ie)       : 1.0
            '''
            
eqs_STDP_pre_ee = 'pre = 1.; w -= nu_pre_ee * post1 * w**exp_pre_ee'
eqs_STDP_post_ee = 'post2before = post2; w += nu_post_ee * pre * post2before * (wmax_ee - w)**exp_post_ee; post1 = 1.; post2 = 1.'

eqs_STDP_pre_ie = 'pre += 1.; w += nu_ie * (post-alpha_ie)'
eqs_STDP_post_ie = 'post += 1.; w += nu_ie * pre'

neuron_groups = {}
input_groups = {}
connections = {}
STDP_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
state_monitors = {}
result_monitor = np.zeros((num_examples,3))

net = b.Network()

#Set parameters for the evaluation of the weights
structural_plasticity_evaluate.set_parameters(n_e)



#------------------------------------------------------------------------------ 
# create network populations and recurrent connections
#------------------------------------------------------------------------------ 
neuron_groups['e'] = b.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= v_reset_e, 
                 compile = True, freeze = True)
neuron_groups['i'] = b.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, 
                 compile = True, freeze = True)

for name in population_names:
    print 'create neuron group', name
    
    neuron_groups[name+'e'] = neuron_groups['e'].subgroup(n_e)
    neuron_groups[name+'i'] = neuron_groups['i'].subgroup(n_i)
    neuron_groups[name+'e'].v = v_rest_e
    neuron_groups[name+'i'].v = v_rest_i
    net.add(neuron_groups[name+'e'], neuron_groups[name+'i'])
    
    print 'create monitors for', name
    rate_monitors[name+'e'] = b.PopulationRateMonitor(neuron_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
    rate_monitors[name+'i'] = b.PopulationRateMonitor(neuron_groups[name+'i'], bin = (single_example_time+resting_time)/b.second)
    net.add(rate_monitors[name+'e'], rate_monitors[name+'i'])
    
    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'])
#         spike_monitors[name+'i'] = SpikeMonitor(neuron_groups[name+'i'])
        net.add(spike_monitors[name+'e'])
#     state_monitors[name+'e'] = MultiStateMonitor(neuron_groups[name+'e'], ['v', 'ge'], record=[0])
#     state_monitors[name+'i'] = MultiStateMonitor(neuron_groups[name+'i'], ['v', 'ge'], record=[0])

#------------------------------------------------------------------------------ 
# create input populations
#------------------------------------------------------------------------------ 
pop_values = [0] * len(input_population_names)
for i,name in enumerate(input_population_names):
    print 'create input group', name
    input_groups[name+'e'] = PoissonGroup(n_e, 0)
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
#     spike_monitors[name+'e'] = SpikeMonitor(input_groups[name+'e'])
    net.add(input_groups[name+'e'], rate_monitors[name+'e'])


#------------------------------------------------------------------------------ 
# create connections from input populations to network populations
#------------------------------------------------------------------------------ 
for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for conn_type in input_conn_names:
        conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
        weight_matrix = get_matrix_from_file(weight_path+conn_name+'.npy')
        weight_matrix = scipy.sparse.lil_matrix(weight_matrix)
        connections[conn_name] = Connection(input_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])
        connections[conn_name].connect(input_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], weight_matrix, delay=delay[conn_type])
        net.add(connections[conn_name])
    
    if ee_STDP_on:
        STDP_methods[name[0]+'e'+name[1]+'e'] =  b.STDP(connections[name[0]+'e'+name[1]+'e'], eqs=eqs_stdp_ee, pre = eqs_STDP_pre_ee, 
                                               post = eqs_STDP_post_ee, wmin=0., wmax= wmax_ee)
        net.add(STDP_methods[name[0]+'e'+name[1]+'e'])


#------------------------------------------------------------------------------ 
# create recurrent connections
#------------------------------------------------------------------------------ 
for name in population_names:
    print 'create recurrent connections for population', name
    for conn_type in recurrent_conn_names:
        conn_name = name+conn_type[0]+name+conn_type[1]
        weight_matrix = get_matrix_from_file(weight_path +conn_name+'.npy')
        weight_matrix = scipy.sparse.lil_matrix(weight_matrix)
        connections[conn_name] = Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])#, delay=delay[conn_type])
        connections[conn_name].connect(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], weight_matrix, delay=delay[conn_type])
        net.add(connections[conn_name])
            
    print 'create STDP for', name
    if ee_STDP_on:
        STDP_methods[name+'e'+name+'e'] =  b.STDP(connections[name+'e'+name+'e'], eqs=eqs_stdp_ee, pre = eqs_STDP_pre_ee, 
                                               post = eqs_STDP_post_ee, wmin=0., wmax= wmax_ee)
        net.add(STDP_methods[name+'e'+name+'e'])
    if not test_mode:
        STDP_methods[name+'i'+name+'e'] = b.STDP(connections[name+'i'+name+'e'], eqs=eqs_stdp_ie, pre = eqs_STDP_pre_ie, 
                                              post = eqs_STDP_post_ie, wmin=0., wmax= wmax_ie)
        net.add(STDP_methods[name+'i'+name+'e'])


#------------------------------------------------------------------------------ 
# create connections between populations
#------------------------------------------------------------------------------ 
for name in inter_pop_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for conn_type in inter_conn_names:
        conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
        weight_matrix = get_matrix_from_file(weight_path+conn_name+'.npy')
        weight_matrix = scipy.sparse.lil_matrix(weight_matrix)
        connections[conn_name] = Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])
        connections[conn_name].connect(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], weight_matrix, delay=delay[conn_type])
        net.add(connections[conn_name])
        
    if ee_STDP_on:
        STDP_methods[name[0]+'e'+name[1]+'e'] =  b.STDP(connections[name[0]+'e'+name[1]+'e'], eqs=eqs_stdp_ee, pre = eqs_STDP_pre_ee, 
                                               post = eqs_STDP_post_ee, wmin=0., wmax= wmax_ee)
        net.add(STDP_methods[name[0]+'e'+name[1]+'e'])
    
if record_spikes:
    b.figure()
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
    # subplot(212)
    # raster_plot(spike_monitors['Hi'], refresh=1000*ms, showlast=1000*ms)

# realTimeMonitor = None
# realTimeMonitor = rltmMon.RealtimeConnectionMonitor(connections['XeAe'], cmap=cm.get_cmap('gist_ncar'), 
#                                                     wmin=0, wmax=wmax_ee, clock = b.Clock(1000*b.ms))


#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
start = time.time()
savetime = 0 #time used to save connections
for j in xrange(int(num_examples)):
    remaining_time = runtime - j*(single_example_time+resting_time)
    
    if resting_time:
        for i,name in enumerate(input_population_names):
            rates = np.ones(n_e)  * 0
            input_groups[name+'e'].rate = rates
        b.run(resting_time, report='text')
        
#     print 'set new rates of the inputs'
    pop_values = [0] * len(input_population_names)
    for i,name in enumerate(input_population_names):
        pop_values[i] = np.random.rand();
        rates = create_topo_input(n_e, pop_values[i])
        input_groups[name+'e'].rate = rates
            
    if j%structural_plasticity_interval == 0 and not test_mode:
        normalize_weights()
        apply_structural_plasticity('XeAe')
        apply_structural_plasticity('AeAe')
        normalize_weights()
    elif j%weight_normalization_interval == 0 and not test_mode:
        print 'run number:', j+1, 'of', int(num_examples), ', remaining time:', remaining_time, 's'
        print ('Estimated remaining real time ' + str((time.time()-savetime-start)/j*(num_examples-j)))
        normalize_weights()
            
    net.run(single_example_time)#, report='text')
        
    tmptime = time.time() #For estimate, subtract connection save times
    if not test_mode:
        if num_examples <= 2000:
            if j%500 == 0:
                save_connections(str(j))
                structural_plasticity_evaluate.evaluate(target_path, 'XeAe', str(j))
        else:
            if j%1000 == 0:
                save_connections(str(j))
                structural_plasticity_evaluate.evaluate(target_path, 'XeAe', str(j))
    savetime += time.time()-tmptime
    
end = time.time()
print 'time needed to simulate:', end - start


#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'

if test_mode:
    np.savetxt(data_path + 'activity/resultPopVecs' + str(num_examples) + '.txt', result_monitor)
else:
    save_connections(str(j))
    normalize_weights()
    save_connections()
    structural_plasticity_evaluate.evaluate(target_path, 'XeAe', '')


#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    b.figure()
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('rates of population ' + name)
    
if spike_monitors:
    b.figure()
    for i, name in enumerate(spike_monitors):
        b.subplot(len(spike_monitors), 1, i)
        b.raster_plot(spike_monitors[name])
        b.title('spikes of population ' + name)
        if name=='Ce':
            timePoints = np.linspace(0+(single_example_time+resting_time)/(2*b.second)*1000, runtime/b.second*1000-(single_example_time+resting_time)/(2*b.second)*1000, num_examples)
            b.plot(timePoints, result_monitor[:,0]*n_e, 'g')
            b.plot(timePoints, result_monitor[:,1]*n_e, 'r')

if state_monitors:
    b.figure()
    for i, name in enumerate(state_monitors):
        b.plot(state_monitors[name].times/b.second, state_monitors[name]['v'][0], label = name + ' v 0')
        b.legend()
        b.title('membrane voltages of population ' + name)
    
    b.figure()
    for i, name in enumerate(state_monitors):
        b.plot(state_monitors[name].times/b.second, state_monitors[name]['ge'][0], label = name + ' v 0')
        b.legend()
        b.title('conductances of population ' + name)

plot_weights = [
                'XeAe', 
#                 'XeAi', 
                'AeAe', 
#                 'AeAi', 
                'AiAe', 
#                 'AiAi', 
               ]
for name in plot_weights:
    b.figure()
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('own2',['#f4f4f4', '#000000'])
    my_cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('own2',['#000000', '#f4f4f4'])
    if name[1]=='e':
        n_src = n_e
    else:
        n_src = n_i
    if name[3]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    w_post = np.zeros((n_src, n_tgt))
    conn_matrix = connections[name][:]
    for i in xrange(n_src):
        w_post[i, conn_matrix.rowj[i]] = conn_matrix.rowdata[i]
    im2 = b.imshow(w_post, interpolation="nearest", vmin = 0, cmap=cm.get_cmap('gist_ncar')) #my_cmap
    cbar2 = b.colorbar(im2)
    b.title('weights of connection' + name)
    
    


b.ioff()
b.show()










