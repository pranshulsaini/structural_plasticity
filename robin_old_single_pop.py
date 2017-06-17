'''
Based on structural plasticity 
created on 05.11.2012
author: peter
Modified on 23.2.2015
@author: Robin Spiess
'''
#------------------------------------------------------------------------------ 
# imports and brian options
#------------------------------------------------------------------------------ 


import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging

import brian as b
from brian import *
from brian.neurongroup import NeuronGroup
defaul_clock_time = 0.5 * b.ms
# from brian.globalprefs import *
b.globalprefs.set_global_preferences( 
                        defaultclock = b.Clock(dt = defaul_clock_time), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave=True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-O3 -ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimisations are turned on 
                        #- if you need IEEE guaranteed results, turn this switch off.
                        useweave_linear_diffeq = False,  # Whether to use weave C++ acceleration for the solution of linear differential 
                        #equations. Note that on some platforms, typically older ones, this is faster and on some platforms, 
                        #typically new ones, this is actually slower.
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenreset = False,  # Whether or not to use experimental code generation support on resets. 
                        #Typically slower due to weave overheads, so usually leave this off.
                        usecodegenthreshold = True,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                        openmp = False,  # Whether or not to use OpenMP pragmas in generated C code. 
                        #If supported on your compiler (gcc 4.2+) it will use multiple CPUs and can run substantially faster.
                        magic_useframes = True,  # Defines whether or not the magic functions should search for objects 
                        #defined only in the calling frame or if they should find all objects defined in any frame. 
                        #This should be set to False if you are using Brian from an interactive shell like IDLE or IPython 
                        #where each command has its own frame, otherwise set it to True.
                       ) 
 
 
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib import rcParams # Used e.g. for spiking activity plots
import time
import scipy 
import scipy.sparse
import scipy.ndimage.filters
from scipy.optimize import curve_fit
import itertools
import os #for path
import sys #for argument in case a script is used
import heapq #For n smallest element
import bisect
import random
import datetime #For timestamp in testmode output
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


#Precompute for slight speedup
complex_unit_roots_1600 = np.array([np.exp(1j*(2*np.pi/1600)*cur_pos) for cur_pos in xrange(1600)])
def compute_pop_vector(pop_array):
    size = len(pop_array)
    cur_pos = 0.
    if size == 1600:
        cur_pos = (np.angle(np.sum(pop_array * complex_unit_roots_1600)) % (2*np.pi)) / (2*np.pi) 
    else:
        print('Complex unit roots not precomputed, for size '+str(size))
        complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
        cur_pos = (np.angle(np.sum(pop_array * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos


def gaussian_1D(x):
    return gaussian_peak * (np.exp(-0.5 * (x / gaussian_sigma)**2))

def gauss_with_offset(x,a,mu,offset,sigma):
    #Compute the noise offset by fitting this function to spikes
    if a is None:
        a = 1/(sigma*np.sqrt(2*np.pi))
    position = x-mu
    position[position > 0.5] = 1-position[position > 0.5]
    position[position < -0.5] = 1+position[position < -0.5]
    return a * (np.exp(-( ((position)**2) / (2*(sigma**2)) ))) + max(0,offset)



def get_matrix_from_file(filename,ending=''):
    if filename[-3]=='e':
        n_src = n_e
    else:
        n_src = n_i
    if filename[-1]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(filename+ending+'.npy')
    value_arr = np.zeros((n_src, n_tgt))
    #e.g. Weak matrices might be empty
    if readout != []:
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
    if 'bookkeeping' in structural_algorithm:
        for weak_name in weak_matrix_names:
            weakcoo = weak_matrix[weak_name].tocoo()
            weak_list = ([(i,j,v) for (i,j,v) in itertools.izip(weakcoo.row, weakcoo.col, weakcoo.data)])
            np.save(target_path + 'weak_matrix_' + weak_name + ending, weak_list)
    #Save target_nnz if the decay was used
    if nnz_decay != 1:
        f = open(target_path+'parameter'+ending+'.txt','w')
        for name in struct_conn_names:
            f.write(name + ','+str(target_nnz[name])+"\n")
        f.close()
            


def normalize_weights():
    print 'normalize weights using means...',
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
                #Originially used the np.mean (therefore rowMean)
                rowMean = np.mean(rowi)
#                 rowMean = np.sum(rowi)
                w_pre[i, connection.rowj[i]] = rowi
                connection.rowdata[i] *= factor/rowMean
                w_post[i, connection.rowj[i]] = connection.rowdata[i]
            
            #Either use sum and multiply with the data entries or compute the mean directly
            col_means = np.sum(w_post, axis = 0)
#             col_means = np.ones(n_e)
#             for i in xrange(n_e):
#                 col_means[i] = np.mean(connection[:,i])
            col_factors = factor/col_means
            col_data_entries = [len(connection.coldataindices[j]) for j in xrange(n_e)]
            
            for j in xrange(n_e):#
                connection[:,j] *= col_factors[j]*col_data_entries[j]
                w_end[j, connection.rowj[j]] = connection.rowdata[j]
    print('done')

            
def apply_structural_plasticity_basic(conn_name):
    conn_type = conn_name[1] + conn_name[3]
    if conn_name[0]==conn_name[2]:
        mean_weight = weight['ee']
    elif conn_name[0] in input_population_names:
        mean_weight = weight['ee_input']
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
        if not (weight_matrix[idx]) and not (conn_name == 'AeAe' and idx[0] == idx[1]):
            new_weight_matrix[idx] = mean_weight*synapse_reset_value
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



def apply_structural_plasticity_spatial_convolve(conn_name):
    #Same as bookkeeping_forced algorithm, but also considers other synapses when placing new ones
    print('Convolution for ' + conn_name)
    conn_type = conn_name[1] + conn_name[3]
    if conn_name[0]==conn_name[2]:
        mean_weight = weight['ee']
    elif conn_name[0] in input_population_names:
        mean_weight = weight['ee_input']
    connection = connections[conn_name][:]

    weight_matrix =  np.zeros((n_e, n_e))
    for i in xrange(n_e):
        weight_matrix[i, connection.rowj[i]] = connection.rowdata[i]
    
    #------------------------------------------------------------------------------ 
    # update bookkeeping matrix, remove weak synapses and add as many new ones
    #------------------------------------------------------------------------------ 
    valid_entries = np.where(weight_matrix >= mean_weight*weak_threshold)
    weak_entries = np.where((weight_matrix > 0) & (weight_matrix < mean_weight*weak_threshold))
    critical_entries = [] #Below threshold but not yet long enough
    num_invalid_entries = 0
#     print(mean_weight*weak_threshold, [np.min(weight_matrix[weight_matrix[:,i]>0,i]) for i in range(weight_matrix.shape[1])])
    #Update entries for currently weak entries
    
    print ("weak entries: "+str(len(weak_entries[0]))+'\t'),
    
    for i, j in zip(weak_entries[0], weak_entries[1]):
        if weak_matrix[conn_name][i,j] >= 0: #Only consider those who are not negative (negative is period of grace)
            weak_matrix[conn_name][i,j] += 2 #Since we decay with 1 in the next step, we add 2. This is not elegant but fast (less checks).
            
        if weak_matrix[conn_name][i,j] > weak_clock_threshold:
            #Deletion is implicit, they are just not added to the new weight matrix
            num_invalid_entries += 1
            weak_matrix[conn_name][i,j] = 0 #Since this synapse will be removed, we can set the bookkeeping entry to 0
        else:
            critical_entries.append((i,j)) #Also Synapses within period of grace are critical (otherwise they would not be added to new weight matrix)
    
    #Critical entries will be kept in new weight matrix
    print("critical entries:" +str(len(critical_entries))+'\t'),
    #Invalid entries will be replaced
    print("invalid entries:" + str(num_invalid_entries))
    critical_entries = np.asarray(critical_entries).transpose() #To go from an array of pairs to a pair of arrays
    #slow decay of all bookkeeping values towards 0
    weak_nonzero = np.nonzero(weak_matrix[conn_name])
    for i, j in zip(weak_nonzero[0], weak_nonzero[1]):
        weak_matrix[conn_name][i,j] -= np.sign(weak_matrix[conn_name][i,j])
        
    
    #Create new Weight matrix, consisting of valid entries + critical entries + newly generated
    num_all_entries = np.count_nonzero(weight_matrix)
    new_weight_matrix = np.zeros(weight_matrix.shape)
    #add Valid entries
    for i, j in zip(valid_entries[0], valid_entries[1]):
        new_weight_matrix[i, j] = weight_matrix[i, j]
    #add Critical entries
    if len(critical_entries) > 0:
        for i, j in zip(critical_entries[0], critical_entries[1]):
            new_weight_matrix[i,j] = weight_matrix[i,j]
    
    #Now check each column for the number of nonzero synapses.
    #Add randomly new ones or delete the weakest to fit the targeted nnz values
    target_nnz[conn_name] = nnz_decay * target_nnz[conn_name]
    target_mean_nnz = int(np.ceil(target_nnz[conn_name]/new_weight_matrix.shape[1]))
    
    #Build Spatial-Weight matrix
    #We want to increase the weight according to the rows 
    #but later we will consider each column 
    #(to keep the input roughly constant so that no neuron dies of starvation)
    
#     plt.figure()
#     im = plt.imshow(new_weight_matrix)
#     plt.colorbar(im)
#     plt.show()
     
    st_time = time.time()
    
    #Give more weight to strong synapses
    spatial_weights = new_weight_matrix/mean_weight
    spatial_weights = np.square(spatial_weights)
    spatial_weights /= np.max(spatial_weights)
    

    # Show the current connection matrix
#     rcParams.update({'xtick.labelsize' : 18})
#     rcParams.update({'ytick.labelsize' : 18})
#     plt.figure(figsize=(10,8))
#     plt.gcf().subplots_adjust(bottom=0.1)
#     plt.title("Connection matrix",fontsize=28)
#     plt.xlabel("Target neuron",fontsize=26)
#     plt.ylabel("Source neuron",fontsize=26)
#     im = plt.imshow(spatial_weights[0:201,0:201],vmin=0.0,vmax=1.0)
#     plt.colorbar(im)
#     plt.show()
    
    spatial_weights *= spatial_factor
    
#     spatial_weights = scipy.ndimage.filters.uniform_filter(spatial_weights,
#                                                         spatial_locality, 
#                                                         mode='wrap')
    if structural_algorithm == 'bookkeeping_spatial_gauss':
        spatial_weights = scipy.ndimage.filters.gaussian_filter(spatial_weights, 
                                              spatial_convolve_sigma, 
                                              mode='wrap')
    else:
        #Custom filter (is sharper than Gaussian)
        height = spatial_convolve_sigma[0]
        width = spatial_convolve_sigma[1]
        filter_kernel = np.zeros((1+2*height,1+2*width))
        for i in range(filter_kernel.shape[0]):
            for j in range(filter_kernel.shape[1]):
                filter_kernel[i,j] = (height - abs(i-height))/height + (width - abs(j-width))/width
        filter_kernel /= np.sum(filter_kernel) #Normalize the filter
#         filter_kernel = np.square(filter_kernel)
        
#         print(filter_kernel)
        spatial_weights = scipy.ndimage.filters.convolve(spatial_weights,
                                                         filter_kernel, 
                                                         mode='wrap')

    spatial_weights[np.where(new_weight_matrix>0)] = 0
    
    spatial_weights = np.power(2,spatial_weights)
    
    #Delete possibility of recursive weights
    if (conn_name[0]==conn_name[2]):
        np.fill_diagonal(spatial_weights,0.,wrap=False)   

    
    #Apply cutoff
    spatial_weights[np.where(spatial_weights > spatial_cutoff)] = spatial_cutoff
    
    # THE FOLLOWING IS ONLY FOR PLOTTING!!!
#     spatial_weights /= spatial_weights[0:201,0:201].max()

    print('Spatial weights created after ' + str(time.time()-st_time))
    
    # Show the probability distribution for new synapses
#     plt.figure(figsize=(10,8))
#     plt.gcf().subplots_adjust(bottom=0.1)
#     plt.title("Probability distribution",fontsize=28)
#     plt.xlabel("Target neuron",fontsize=26)
#     plt.ylabel("Source neuron",fontsize=26)
#     im = plt.imshow(spatial_weights[0:201,0:201],vmin=0.0,vmax=1.0)
#     plt.colorbar(im)
#     plt.show()
    
    
    for col in range(new_weight_matrix.shape[1]):
        num_nonzero = np.count_nonzero(new_weight_matrix[:,col])
        if num_nonzero < target_mean_nnz:
            #Not enough synapses, build spatial weights
            #take a slice of the spatial_weight matrix
            cur_spatial_weights = spatial_weights[:,col]
            #Convert weights into running sum
            for i in range(1,len(cur_spatial_weights[:])):
                cur_spatial_weights[i] += cur_spatial_weights[i-1]
            
            
            while num_nonzero < target_mean_nnz:
                rnd = random.random()*cur_spatial_weights[-1]
                row = bisect.bisect_right(cur_spatial_weights,rnd)
                idx = (row, col)
                while (new_weight_matrix[idx]) or (conn_name[0] == conn_name[2] and idx[0] == idx[1]):
#                     print('hit already existing weight: ',idx,cur_spatial_weights[row]-cur_spatial_weights[row-1],weight_matrix[idx],new_weight_matrix[idx])
                    rnd = random.random()*cur_spatial_weights[-1]
                    row = bisect.bisect_right(cur_spatial_weights,rnd)
                    idx = (row, col)
#                     print(idx,cur_spatial_weights[row]-cur_spatial_weights[row-1],weight_matrix[idx],new_weight_matrix[idx])
                new_weight_matrix[idx] = mean_weight*weak_reset_value
                weak_matrix[conn_name][idx] = weak_clock_reset
                num_nonzero += 1
        elif num_nonzero > target_mean_nnz:
            while num_nonzero > target_mean_nnz:
                nnz_indices = np.nonzero(new_weight_matrix[:,col])
                minind = np.argmin(new_weight_matrix[nnz_indices[0],col])
                new_weight_matrix[nnz_indices[0][minind],col] = 0
                weak_matrix[conn_name][minind,col] = 0
                num_nonzero -= 1
        
                
    num_new_all_entries = np.count_nonzero(new_weight_matrix)

    print 'oldnnz:',num_all_entries, 'newnnz:',num_new_all_entries,'inval:', num_invalid_entries, 'target_nnz:', target_nnz[conn_name] 
            
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











def apply_structural_plasticity_spatial_row(conn_name):
    #Same as bookkeeping_forced algorithm, but also considers other synapses when placing new ones
    conn_type = conn_name[1] + conn_name[3]
    if conn_name[0]==conn_name[2]:
        mean_weight = weight['ee']
    elif conn_name[0] in input_population_names:
        mean_weight = weight['ee_input']
    connection = connections[conn_name][:]

    weight_matrix =  np.zeros((n_e, n_e))
    for i in xrange(n_e):
        weight_matrix[i, connection.rowj[i]] = connection.rowdata[i]
    
    #------------------------------------------------------------------------------ 
    # update bookkeeping matrix, remove weak synapses and add as many new ones
    #------------------------------------------------------------------------------ 
    valid_entries = np.where(weight_matrix >= mean_weight*weak_threshold)
    weak_entries = np.where((weight_matrix > 0) & (weight_matrix < mean_weight*weak_threshold))
    critical_entries = [] #Below threshold but not yet long enough
    num_invalid_entries = 0
#     print(mean_weight*weak_threshold, [np.min(weight_matrix[weight_matrix[:,i]>0,i]) for i in range(weight_matrix.shape[1])])
    #Update entries for currently weak entries
    
    print ("weak entries: "+str(len(weak_entries[0]))+'\t'),
    
    for i, j in zip(weak_entries[0], weak_entries[1]):
        if weak_matrix[conn_name][i,j] >= 0: #Only consider those who are not negative (negative is period of grace)
            weak_matrix[conn_name][i,j] += 2 #Since we decay with 1 in the next step, we add 2. This is not elegant but fast (less checks).
            
        if weak_matrix[conn_name][i,j] > weak_clock_threshold:
            #Deletion is implicit, they are just not added to the new weight matrix
            num_invalid_entries += 1
            weak_matrix[conn_name][i,j] = 0 #Since this synapse will be removed, we can set the bookkeeping entry to 0
        else:
            critical_entries.append((i,j)) #Also Synapses within period of grace are critical (otherwise they would not be added to new weight matrix)
    
    #Critical entries will be kept in new weight matrix
    print("critical entries:" +str(len(critical_entries))+'\t'),
    #Invalid entries will be replaced
    print("invalid entries:" + str(num_invalid_entries))
    critical_entries = np.asarray(critical_entries).transpose() #To go from an array of pairs to a pair of arrays
    #slow decay of all bookkeeping values towards 0
    weak_nonzero = np.nonzero(weak_matrix[conn_name])
    for i, j in zip(weak_nonzero[0], weak_nonzero[1]):
        weak_matrix[conn_name][i,j] -= np.sign(weak_matrix[conn_name][i,j])
        
    
    #Create new Weight matrix, consisting of valid entries + critical entries + newly generated
    num_all_entries = np.count_nonzero(weight_matrix)
    new_weight_matrix = np.zeros(weight_matrix.shape)
    #add Valid entries
    for i, j in zip(valid_entries[0], valid_entries[1]):
        new_weight_matrix[i, j] = weight_matrix[i, j]
    #add Critical entries
    if len(critical_entries) > 0:
        for i, j in zip(critical_entries[0], critical_entries[1]):
            new_weight_matrix[i,j] = weight_matrix[i,j]
    
    #Now check each column for the number of nonzero synapses.
    #Add randomly new ones or delete the weakest to fit the targeted nnz values
    target_nnz[conn_name] = nnz_decay * target_nnz[conn_name]
    target_mean_nnz = int(np.ceil(target_nnz[conn_name]/new_weight_matrix.shape[1]))
    
    #Build Spatial-Weight matrix
    #We want to increase the weight according to the rows 
    #but later we will consider each column 
    #(to keep the input roughly constant so that no neuron dies of starvation)
    spatial_weights = np.ones((n_e,n_e))
    
    st_time = time.time()
    
    #Precompute factors (can save ~1-2 sec for 'hat' function)
    precomp_factor = np.zeros(spatial_locality*2+1)
    for i in range(0,spatial_locality*2+1):
        if spatial_method == 'hat':
            precomp_factor[i] = spatial_factor * (1 - (float(abs(i-spatial_locality))/spatial_locality))
        else:
            precomp_factor[i] = spatial_factor
    
    
    
    for row in range(n_e):
        for col in range(n_e):
            if new_weight_matrix[row,col] > 0:
                spatial_weights[row,col] = 0 #We can't place a new synapse here (already non-zero)
                offset = col-spatial_locality
                w = new_weight_matrix[row,col]/mean_weight
                if spatial_operator=='mul':
                    for i in range(max(0,offset),min(n_e,col+spatial_locality+1)):
                        spatial_weights[row,i] *= (1 + (w * precomp_factor[i-offset] ))
#                         print(row,col,abs(i-col),w*precomp_factor[i-offset],precomp_factor[i-offset],new_weight_matrix[row,col],w)
                else:
                    for i in range(max(0,offset),min(n_e,col+spatial_locality+1)):
                        if spatial_weights[row,i] > 0:
                            #Add factor, If it is 0, it is occupied
                            spatial_weights[row,i] += (w * precomp_factor[i-offset] )
    
    #Delete possibility recursive weights
    if (conn_name[0]==conn_name[2]):
        np.fill_diagonal(spatial_weights,0.,wrap=False)   
    #Apply cutoff
    spatial_weights[np.where(spatial_weights > spatial_cutoff)] = spatial_cutoff

    print('Spatial weights created after ' + str(time.time()-st_time))
    
    
    plt.figure()
    im = plt.imshow(spatial_weights[0:201,0:201])
    plt.colorbar(im)
    plt.show()
    
    
    for col in range(new_weight_matrix.shape[1]):
        num_nonzero = np.count_nonzero(new_weight_matrix[:,col])
        if num_nonzero < target_mean_nnz:
            #Not enough synapses, build spatial weights
            #take a slice of the spatial_weight matrix
            cur_spatial_weights = spatial_weights[:,col]
            #Convert weights into running sum
            for i in range(1,len(cur_spatial_weights[:])):
                cur_spatial_weights[i] += cur_spatial_weights[i-1]
            
            
            while num_nonzero < target_mean_nnz:
                rnd = random.random()*cur_spatial_weights[-1]
                row = bisect.bisect_right(cur_spatial_weights,rnd)
                idx = (row, col)
                while (new_weight_matrix[idx]) or (conn_name[0] == conn_name[2] and idx[0] == idx[1]):
#                     print('hit already existing weight: ',idx,cur_spatial_weights[row]-cur_spatial_weights[row-1],weight_matrix[idx],new_weight_matrix[idx])
                    rnd = random.random()*cur_spatial_weights[-1]
                    row = bisect.bisect_right(cur_spatial_weights,rnd)
                    idx = (row, col)
#                     print(idx,cur_spatial_weights[row]-cur_spatial_weights[row-1],weight_matrix[idx],new_weight_matrix[idx])
                new_weight_matrix[idx] = mean_weight*weak_reset_value
                weak_matrix[conn_name][idx] = weak_clock_reset
                num_nonzero += 1
        elif num_nonzero > target_mean_nnz:
            while num_nonzero > target_mean_nnz:
                nnz_indices = np.nonzero(new_weight_matrix[:,col])
                minind = np.argmin(new_weight_matrix[nnz_indices[0],col])
                new_weight_matrix[nnz_indices[0][minind],col] = 0
                weak_matrix[conn_name][minind,col] = 0
                num_nonzero -= 1
        
                
    num_new_all_entries = np.count_nonzero(new_weight_matrix)

    print 'oldnnz:',num_all_entries, 'newnnz:',num_new_all_entries,'inval:', num_invalid_entries, 'target_nnz:', target_nnz[conn_name] 
            
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












def apply_structural_plasticity_bookkeeping(conn_name):
    #Same as bookkeeping algorithm, but forces number of synapses in all columns constant.
    #Decay works by decreasing the target number of synapses per column
    conn_type = conn_name[1] + conn_name[3]
    if conn_name[0]==conn_name[2]:
        mean_weight = weight['ee']
    elif conn_name[0] in input_population_names:
        mean_weight = weight['ee_input']
    connection = connections[conn_name][:]

    weight_matrix =  np.zeros((n_e, n_e))
    for i in xrange(n_e):#
        weight_matrix[i, connection.rowj[i]] = connection.rowdata[i]
    
    #------------------------------------------------------------------------------ 
    # update bookkeeping matrix, remove weak synapses and add as many new ones
    #------------------------------------------------------------------------------ 
    valid_entries = np.where(weight_matrix >= mean_weight*weak_threshold)
    weak_entries = np.where((weight_matrix > 0) & (weight_matrix < mean_weight*weak_threshold))
    critical_entries = [] #Below threshold but not yet long enough
    num_invalid_entries = 0
#     print(mean_weight*weak_threshold, [np.min(weight_matrix[weight_matrix[:,i]>0,i]) for i in range(weight_matrix.shape[1])])
    #Update entries for currently weak entries
    print ("weak entries: "+str(len(weak_entries[0]))+'\t'),
    for i, j in zip(weak_entries[0], weak_entries[1]):
        if weak_matrix[conn_name][i,j] >= 0: #Only consider those who are not negative (negative is period of grace)
            weak_matrix[conn_name][i,j] += 2 #Since we decay with 1 in the next step, we add 2
            
        if weak_matrix[conn_name][i,j] > weak_clock_threshold:
            #Deletion is implicit, they are just not added to the new weight matrix
            num_invalid_entries += 1
            weak_matrix[conn_name][i,j] = 0 #Since this synapse will be removed, we can set the bookkeeping entry to 0
        else:
            critical_entries.append((i,j)) #Also Synapses within period of grace are critical (otherwise they would not be added to new weight matrix)
    #Critical entries will be kept in new weight matrix
    print("critical entries:" +str(len(critical_entries))+'\t'),
    #Invalid entries will be replaced
    print("invalid entries:" + str(num_invalid_entries))
    critical_entries = np.asarray(critical_entries).transpose() #To go from an array of pairs to a pair of arrays
    #slow decay of all bookkeeping values towards 0
    weak_nonzero = np.nonzero(weak_matrix[conn_name])
    for i, j in zip(weak_nonzero[0], weak_nonzero[1]):
        weak_matrix[conn_name][i,j] -= np.sign(weak_matrix[conn_name][i,j])
        
    
    #Create new Weight matrix, consisting of valid entries + critical entries + newly generated
    num_all_entries = np.count_nonzero(weight_matrix)
    new_weight_matrix = np.zeros(weight_matrix.shape)
    #add Valid entries
    for i, j in zip(valid_entries[0], valid_entries[1]):
        new_weight_matrix[i, j] = weight_matrix[i, j]
    #add Critical entries
    if len(critical_entries) > 0:
        for i, j in zip(critical_entries[0], critical_entries[1]):
            new_weight_matrix[i,j] = weight_matrix[i,j]
    
    
    #Now check each column for the number of nonzero synapses.
    #Add randomly new ones or delete the weakest to fit the targeted nnz values
    target_nnz[conn_name] = nnz_decay * target_nnz[conn_name]
    target_mean_nnz = int(np.ceil(target_nnz[conn_name]/new_weight_matrix.shape[1]))
    
    for col in range(new_weight_matrix.shape[1]):
        num_nonzero = np.count_nonzero(new_weight_matrix[:,col])
        if num_nonzero < target_mean_nnz:
            while num_nonzero < target_mean_nnz:
                idx = (np.int32(np.random.rand()*weight_matrix.shape[0]), col)
                while (new_weight_matrix[idx]) or (conn_name == 'AeAe' and idx[0] == idx[1]):
                    idx = (np.int32(np.random.rand()*weight_matrix.shape[0]), col)
                new_weight_matrix[idx] = mean_weight*weak_reset_value
                weak_matrix[conn_name][idx] = weak_clock_reset
                num_nonzero += 1
        elif num_nonzero > target_mean_nnz:
#         if num_nonzero > target_mean_nnz:
#             nnz_indices = np.nonzero(new_weight_matrix)
#             cutval = heapq.nsmallest(num_nonzero-target_mean_nnz,new_weight_matrix[nnz_indices])
#             cutval = cutval[-1]
#             indices = np.where((new_weight_matrix > 0) & (new_weight_matrix <= cutval))
#             for i,j in zip(indices[0],indices[1]):
#                 new_weight_matrix[i,j] = 0
#                 weak_matrix[conn_name][i,j] = 0
            while num_nonzero > target_mean_nnz:
                nnz_indices = np.nonzero(new_weight_matrix[:,col])
                minind = np.argmin(new_weight_matrix[nnz_indices[0],col])
                new_weight_matrix[nnz_indices[0][minind],col] = 0
                weak_matrix[conn_name][minind,col] = 0
                num_nonzero -= 1
        
            
                
    num_new_all_entries = np.count_nonzero(new_weight_matrix)

    print 'oldnnz:',num_all_entries, 'newnnz:',num_new_all_entries,'inval:', num_invalid_entries, 'target_nnz:', target_nnz[conn_name] 
            
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
    if structural_algorithm == 'none':
        return
    elif structural_algorithm == 'basic':
        apply_structural_plasticity_basic(conn_name)
    elif structural_algorithm == 'bookkeeping':
        apply_structural_plasticity_bookkeeping(conn_name)
    elif structural_algorithm == 'bookkeeping_spatial_row':
        apply_structural_plasticity_spatial_row(conn_name)
    elif structural_algorithm in ['bookkeeping_spatial_gauss','bookkeeping_spatial_custom']:
        apply_structural_plasticity_spatial_convolve(conn_name)    


            
#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
# data_path = '../data/structural_plasticity/'
data_path = os.getcwd()
weight_path = data_path
target_path = data_path
src_ending = '' #Ending for weights to be loaded (iteration_offset is used for the saved weights)
iteration_offset = 0 #Offset for target ending & scoring iteration

# Read arguments: (1) src_ending, (2) weight_path, (3) target_path, (4) num_examples,
#                 (5) test_mode, (6) structural_plasticity_algorithm, (7) pruning 
if len(sys.argv) > 1:
    src_ending = str(sys.argv[1])
    if src_ending == 'empty':
        src_ending = ''
    print('Loading weights with ending = ' + src_ending)
if len(sys.argv) > 2:
    weight_path = data_path + str(sys.argv[2])
    print('Loading from '+weight_path)
if len(sys.argv) > 3:
    target_path = data_path + str(sys.argv[3])
    print('Target path '+target_path)

n_e = 1600
n_i = n_e/4

single_example_time =  0.25*b.second # How long a single example is shown to the network
resting_time = 0.01*b.second #0 for training, 0.01 for testing
num_examples = 1000 #How many iterations should be trained or tested
if len(sys.argv) > 4:
    num_examples = int(sys.argv[4])
runtime = num_examples*(single_example_time+resting_time)

weight_normalization_interval = 20 #20 default
structural_plasticity_interval = 50 #100 for basic, 50 for bookkeeping
save_interval = 2500 #Save connections after this interval

save_simple_evaluation = False # A simple evaluation of the connection matrices. Higher score if stronger synapses are on the diagonal.
evaluate_interval = 250


test_mode = True #For test_mode = True set resting_time > 0
if len(sys.argv) > 5:
    test_mode = (sys.argv[5].lower() == 'true')
if test_mode:
    record_spikes = False#True #No plots, so that testmode can be batch processed
    ee_STDP_on = False
else:
    record_spikes = False
    ee_STDP_on = True
show_spike_bars = False #For testMode only. Shows spike response after each example

#Load parameters from previous run? (only used for the continuation of a run)
weak_import = False #Load weak matrices or not
load_parameter = False #Load the target_nnz

#Parameters for structural plasticity
# Possible algorithms: 'none', 'basic', 'bookkeeping', 'bookkeeping_spatial_row',
#                      'bookkeeping_spatial_gauss', 'bookkeeping_spatial_custom'
structural_algorithm = 'bookkeeping_spatial_gauss'
if len(sys.argv) > 6:
    structural_algorithm = sys.argv[6]
struct_conn_names = ['XeAe','AeAe']





#e.g. After 30000 iterations have only 'First Argument' of non-zero synapses left
#decay/pruning is applied when structural plasticity algorithm is executed
pruning = False
if len(sys.argv) > 7:
    pruning = (sys.argv[7].lower() == 'true')
if pruning:
    nnz_decay = np.power(1/2.,1./(30000/structural_plasticity_interval))
else:
    nnz_decay = 1 
target_nnz = {} #Target number of non zero entries, to fill up 0s due to STDP and to simulate decay of global #synapses
for name in struct_conn_names:
    target_nnz[name] = 0 # Create entries in dictionary, true values are set when matrices are loaded


#parameters for 'basic'
synapse_removal_thres = 1/20.
synapse_reset_value = 1/20.

#parameters for 'bookkeeping***' algorithms
weak_threshold = 1/10. #synapse is weak if it is smaller than mean*weak_threshold
weak_reset_value = 1/10. #1/20#1/25. #Strength of new synapse
weak_clock_threshold = 2#3 #How many times may a synapse be under the threshold
weak_clock_reset = -3#-5 #Period of grace for new synapses
weak_matrix = {} #The matrices which are used for the monitoring of the synapses
weak_matrix_names = ['XeAe','AeAe'] #This is most likely equal to struct_conn_names
if 'bookkeeping' in structural_algorithm:
    weak_matrix['XeAe'] = scipy.sparse.lil_matrix((n_e,n_e),dtype='i')
    weak_matrix['AeAe'] = scipy.sparse.lil_matrix((n_e,n_e),dtype='i')


#Parameters for 'spatial' algorithm
spatial_locality = 1#15 #1
spatial_factor = 250#1.5#250.
spatial_cutoff = n_e #Maximum weight, to prevent massive oversampling (allows higher parameters). Choice is arbitrary
spatial_operator = 'mul' #'mul' or 'add'
spatial_method = 'hat' #'hat' or 'const'
spatial_convolve_sigma = [5.,10.] #The sigma for the convolution filters (vertical, horizontal)



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

weight['ee_input'] = 0.15 #0.15        # 0.10
weight['ee'] = 0.03 #0.03

#Old delays
# delay['ee_input'] = (0*b.ms,10*b.ms)
# delay['ei_input'] = (0*b.ms,5*b.ms)
# delay['ee'] = (0*b.ms,5*b.ms)
# delay['ei'] = (0*b.ms,2*b.ms)
# delay['ie'] = (0*b.ms,1*b.ms)
# delay['ii'] = (0*b.ms,2*b.ms)

#New delays
delay['ee_input'] = (1*b.ms,10*b.ms)
delay['ei_input'] = (0*b.ms,3*b.ms)
delay['ee'] = (1*b.ms,5*b.ms)
delay['ei'] = (0*b.ms,1*b.ms)
delay['ie'] = (0*b.ms,1*b.ms)
delay['ii'] = (0*b.ms,1*b.ms)


tc_pre_ee = 20*b.ms
tc_post_1_ee = 40*b.ms
TCpost2EE = 40*b.ms
tc_pre_ie = 20*b.ms
tc_post_ie = 20*b.ms
nu_pre_ee =  0.0005      # learning rate -- decrease for exc->exc
nu_post_ee = 0.0025      # learning rate -- increase for exc->exc
nu_ie =      0.005       # learning rate -- for inh->exc
if structural_algorithm == 'none':
    nu_ie = 0.01
alpha_ie = 3*b.Hz*tc_post_ie*2    # controls the firing rate
wmax_ee = 0.5
wmax_ie = 1000.
exp_pre_ee = 0.2
exp_post_ee = exp_pre_ee




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
result_monitor = np.zeros((num_examples,6))

net = b.Network()

#Set parameters for the evaluation of the weights
structural_plasticity_evaluate.set_parameters(n_e)



#------------------------------------------------------------------------------ 
# create network populations and recurrent connections
#------------------------------------------------------------------------------ 
for name in population_names:
    print 'create neuron group', name

    neuron_groups[name+'e'] = b.NeuronGroup(n_e, neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= v_reset_e, 
                 compile = True, freeze = True)
    neuron_groups[name+'i'] = b.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, 
                 compile = True, freeze = True)

#     neuron_groups[name+'e'] = neuron_groups['e'].subgroup(n_e)
#     neuron_groups[name+'i'] = neuron_groups['i'].subgroup(n_i)
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
        weight_matrix = get_matrix_from_file(weight_path+conn_name,src_ending)
        weight_matrix = scipy.sparse.lil_matrix(weight_matrix)
        connections[conn_name] = Connection(input_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])
        connections[conn_name].connect(input_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], weight_matrix, delay=delay[conn_type])
        net.add(connections[conn_name])
        if conn_name in struct_conn_names and not load_parameter:
            target_nnz[conn_name] = len(np.nonzero(weight_matrix)[0])
    
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
        weight_matrix = get_matrix_from_file(weight_path +conn_name,src_ending)
        weight_matrix = scipy.sparse.lil_matrix(weight_matrix)
        connections[conn_name] = Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0], delay=True, max_delay=delay[conn_type][1])#, delay=delay[conn_type])
        connections[conn_name].connect(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]], weight_matrix, delay=delay[conn_type])
        net.add(connections[conn_name])
        if conn_name in struct_conn_names and not load_parameter:
            #Save current nonzero elements
            target_nnz[conn_name] = len(np.nonzero(weight_matrix)[0])
                
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
        weight_matrix = get_matrix_from_file(weight_path+conn_name,src_ending)
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
    #b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
    # subplot(212)
    # raster_plot(spike_monitors['Hi'], refresh=1000*ms, showlast=1000*ms)

# realTimeMonitor = None
# realTimeMonitor = rltmMon.RealtimeConnectionMonitor(connections['XeAe'], cmap=cm.get_cmap('gist_ncar'), 
#                                                     wmin=0, wmax=wmax_ee, clock = b.Clock(1000*b.ms))

#Spike Counter for Test Mode
spike_counter = {}
spike_counter_input = {}
if test_mode:
    spike_counter = b.SpikeCounter(neuron_groups['Ae'])
    net.add(spike_counter)
    spike_counter_input = b.SpikeCounter(input_groups['Xe'])
    net.add(spike_counter_input)

#------------------------------------------------------------------------------ 
# Load weak matrices for 'bookkeeping' algorithm
# And load parameter for decay
#------------------------------------------------------------------------------
if not test_mode:
    if ('bookkeeping' in structural_algorithm) and weak_import:
        for weak_name in weak_matrix_names:
            print('import weak matrix '+weak_name)
            weak_matrix_loaded = get_matrix_from_file(weight_path + 'weak_matrix_' + weak_name,src_ending)
            weak_matrix[weak_name] = scipy.sparse.lil_matrix(weak_matrix_loaded)
    
    if load_parameter:
        data = np.genfromtxt(weight_path + 'parameter'+src_ending+'.txt',dtype=None,delimiter=',')
        print 'Load parameter for target_nnz ', [i[0] for i in data],
        for d in data:
            target_nnz[d[0]] = float(d[1])
            print target_nnz[d[0]],
        print ''

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
previous_spike_count_input = np.zeros(n_e)
start = time.time()
savetime = 0 #time used to save connections
for j in xrange(int(num_examples)):
    remaining_time = runtime - j*(single_example_time+resting_time)
    
    if resting_time:
        for i,name in enumerate(input_population_names):
            rates = np.ones(n_e)  * 0
            input_groups[name+'e'].rate = rates
        net.run(resting_time)
        
    if test_mode:
        previous_spike_count = np.copy(spike_counter.count[:])
        previous_spike_count_input = np.copy(spike_counter_input.count[:])

#     print 'set new rates of the inputs'
    pop_values = [0] * len(input_population_names)
    testing_pop_values = [0.25,0.75,0.5,0.8,0.3]
    for i,name in enumerate(input_population_names):
        pop_values[i] = np.random.rand()#testing_pop_values[j%len(testing_pop_values)]
        rates = create_topo_input(n_e, pop_values[i])
        input_groups[name+'e'].rate = rates
        if test_mode:
            result_monitor[j,1] = pop_values[i]
            
    if j%structural_plasticity_interval == 0 and not test_mode:
        normalize_weights()
        apply_structural_plasticity('XeAe')
        apply_structural_plasticity('AeAe')
        normalize_weights()
    elif j%weight_normalization_interval == 0:
        print 'run number:', j+1, 'of', int(num_examples), ', remaining time:', remaining_time, 's'
        #print ('Estimated remaining real time ' + str((time.time()-savetime-start)/max(j,1)*(num_examples-j)))
        normalize_weights()
           
    net.run(single_example_time)#, report='text')
    
    tmptime = time.time() #For estimate, subtract connection save times
    if not test_mode:
        if save_simple_evaluation and j%evaluate_interval == 0:
            #Sorted evaluation
            order = {}
            for conn_name in struct_conn_names:
                conn_matrix = connections[conn_name][:]
                conn_list_sparse = ([(i,k[0],k[1]) for i in xrange(conn_matrix.shape[0]) for k in zip(conn_matrix.rowj[i],conn_matrix.rowdata[i])])
    #             weight_list = {}
    #             weight_list['XeAe'] = ([(i,j[0],j[1]) for i in xrange(connections['XeAe'][:].shape[0]) for j in zip(connections['XeAe'][:].rowj[i],connections['XeAe'][:].rowdata[i])])
                order = structural_plasticity_evaluate.evaluate(target_path, 
                                        conn_name, str(j + iteration_offset),
                                        order=order,tgtPath=target_path,weightList=conn_list_sparse)
            
            #Unsorted evaluation, used especially for spatial algorithm
            order = {}
            for name in population_names:
                order[name] = range(0,1600) #No sorting required
            for conn_name in struct_conn_names:
                conn_matrix = connections[conn_name][:]
                conn_list_sparse = ([(i,k[0],k[1]) for i in xrange(conn_matrix.shape[0]) for k in zip(conn_matrix.rowj[i],conn_matrix.rowdata[i])])
                structural_plasticity_evaluate.evaluate(target_path, 
                                        conn_name, str(j + iteration_offset),
                                        order=order,tgtPath=target_path+'unsorted_',weightList=conn_list_sparse)
            
        if j%save_interval == 0:
            save_connections(str(iteration_offset+j))
    else:
        current_spike_count = np.asarray(spike_counter.count[:]) - previous_spike_count
        result_monitor[j,0] = compute_pop_vector(current_spike_count)
        difference = np.abs(result_monitor[j,0] - result_monitor[j,1])
        if difference > 0.5:
            difference = 1-difference
        result_monitor[j,2] = difference
        
        #compute sample standard deviation
        stddev = 0.
        for i in range(0,n_e):
            dif = np.abs(result_monitor[j,0] - float(i)/n_e)
            if dif > 0.5:
                dif = 1-dif
            stddev += current_spike_count[i]*dif*dif
        stddev /= np.sum(current_spike_count)
        stddev = np.sqrt(stddev)
        
        result_monitor[j,3] = stddev
        
        input_sigma = 1.0/12.0;
        xx = np.arange(0.0,n_e)/float(n_e)
        popt, pcov = curve_fit(gauss_with_offset, xx, current_spike_count, p0=[1.0/(input_sigma*np.sqrt(2*np.pi)), result_monitor[j,0], 0, input_sigma])
        #Set offset to at least 0
        popt[2] = max(0,popt[2])
        result_monitor[j,4] = popt[2]
        result_monitor[j,5] = popt[3]
        
        if j%10 == 0:
            if record_spikes:
                plt.figure()
                plt.xlim((j-5)*1000*(single_example_time+resting_time),j*1000*(single_example_time+resting_time))
                b.raster_plot(spike_monitors['Ae'])
            #print(j,np.mean(result_monitor[0:j+1,2]),result_monitor[j,0],result_monitor[j,1],result_monitor[j,2],result_monitor[j,3])
        if show_spike_bars:
            plt.gcf().subplots_adjust(bottom=0.15)
            rcParams.update({'xtick.labelsize' : 18})
            rcParams.update({'ytick.labelsize' : 18})
            plt.figure(figsize=(8,5),dpi=120)
            plt.title('Spike activity',fontsize=20)
            plt.xlabel('Neuron',fontsize=20)
            plt.ylabel('Spikes',fontsize=20)
            plt.bar(np.arange(0,1600),current_spike_count,width=1.0,linewidth=0,color='blue')
            trueGauss = input_groups['Xe'].rate/gaussian_peak*np.max(current_spike_count)
            plt.plot(np.arange(0,1600),trueGauss,color='black')
            plt.bar(pop_values[0]*1600-4,np.max(current_spike_count),width=8.0,linewidth=0,color='black')
            plt.plot(np.arange(0,1600), gauss_with_offset(xx, *popt))
#             plt.bar(result_monitor[j,0]*n_e-5,np.max(current_spike_count),width=10.0,linewidth=0,color='red')
#             plt.bar(result_monitor[j,0]*n_e, 0.03*np.max(current_spike_count),gaussian_sigma/2*n_e,linewidth=0,color='yellow')
#             plt.bar(result_monitor[j,0]*n_e, 0.015*np.max(current_spike_count), stddev*n_e,linewidth=0,color='red')
            print('Noise offset: ',popt[2])
            print('Standard deviation: ',popt[3])

            plt.show()
            
            np.save(target_path+'spikecount'+str(j),current_spike_count)
            
            #Draw the input activity
            if False:
                plt.figure(figsize=(8,5),dpi=120)
                plt.title('Input Neurons Activity')
                plt.xlabel('Neuron')
                plt.ylabel('Spikes')
                current_spike_count_input = np.asarray(spike_counter_input.count[:]) - previous_spike_count_input
                plt.bar(np.arange(0,1600),current_spike_count_input,width=1.0,linewidth=0,color='blue')
                trueGauss = trueGauss/np.max(current_spike_count)*np.max(current_spike_count_input)
                plt.plot(np.arange(0,1600),trueGauss,color='red')
                plt.bar(result_monitor[j,1]*n_e-2.5,np.max(current_spike_count_input),width=5.0,linewidth=0,color='red')
                plt.show()
            
            
    savetime += time.time()-tmptime
    
end = time.time()
print 'Weights from:',weight_path
print 'time needed to simulate:', end - start


#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'

if test_mode:
    print('Mean error: ' + str(np.mean(result_monitor[:,2])))
    print('Mean variance: ' + str(np.mean(result_monitor[:,3])))
    print('Mean noise offset: '+str(np.mean(result_monitor[:,4])))
    np.savetxt(target_path + 'pop_vector/resultPopVecs' + src_ending + '_' + str(num_examples) + '.txt', result_monitor)
    
    f = open(target_path+'testmode_A'+'.txt','a')
    f.write(str(datetime.datetime.now().replace(microsecond=0)) +',' 
            + str(np.mean(result_monitor[:,2]))+","
            +src_ending+","
            +str(target_nnz['XeAe'])+"\n")
    f.close()
    
else:
    save_connections(str(iteration_offset+j))
    normalize_weights()
    save_connections()
    if save_simple_evaluation:
        order = structural_plasticity_evaluate.evaluate(target_path, 'XeAe', '',order={},tgtPath=target_path)
        structural_plasticity_evaluate.evaluate(target_path, 'AeAe', '',order=order,tgtPath=target_path)


#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors and not test_mode:
    b.figure()
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('rates of population ' + name)
    plt.savefig(target_path+'rates_'+str(iteration_offset)+'to'+str(iteration_offset+num_examples))
    plt.close()
    
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
#                 'XeAe', 
#                 'XeAi', 
#                 'AeAe', 
#                 'AeAi', 
#                 'AiAe', 
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










