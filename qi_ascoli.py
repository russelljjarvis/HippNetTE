
# Code for finding the the rheobase current injection values of excitatory and inhibitory neurons.

'''

from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization import get_neab
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON'))
attrs = {'a':0.02, 'b':0.2, 'c':-65+15*0.5, 'd':8-0.5**2 }
#from neuronunit.optimization.data_transport_container import DataTC
#dtc = DataTC()
from neuronunit.tests import fi
model.set_attrs(attrs)
from neuronunit.optimization import get_neab
rtp = get_neab.tests[0]
rheobase0 = rtp.generate_prediction(model)
rheobase = rheobase0
attrs2 = {'a':0.02+0.08*0.5, 'b':0.2-0.05*0.5, 'c':-65, 'd':2 }
#, i_offset=0)
#attrs = {'a':0.02, 'b':0.2, 'c':-65+15*0.5, 'd':8-0.5**2 }
#from neuronunit.optimization.data_transport_container import DataTC
#dtc = DataTC()
from neuronunit.tests import fi
model.set_attrs(attrs2)
from neuronunit.optimization import get_neab
rtp = get_neab.tests[0]
rheobase1 = rtp.generate_prediction(model)
print(rheobase0['value'],rheobase1['value'])
assert rheobase0['value'] != rheobase1['value']
'''

def sim_runner(wgf):
    wg = wgf

    import pyNN.neuron as sim
    nproc = sim.num_processes()
    node = sim.rank()
    print(nproc)
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({'font.size':16})

    #import mpi4py
    #threads  = sim.rank()
    threads = 1
    rngseed  = 98765
    parallel_safe = False
    #extra = {'threads' : threads}
    import os
    import pandas as pd
    import sys
    import numpy as np
    from pyNN.neuron import STDPMechanism
    import copy
    from pyNN.random import RandomDistribution, NumpyRNG
    import pyNN.neuron as neuron
    from pyNN.neuron import h
    from pyNN.neuron import StandardCellType, ParameterSpace
    from pyNN.random import RandomDistribution, NumpyRNG
    from pyNN.neuron import STDPMechanism, SpikePairRule, AdditiveWeightDependence, FromListConnector, TsodyksMarkramSynapse
    from pyNN.neuron import Projection, OneToOneConnector
    from numpy import arange
    import pyNN
    from pyNN.utility import get_simulator, init_logging, normalized_filename
    import random
    import socket
    #from neuronunit.optimization import get_neab
    import networkx as nx
    sim = pyNN.neuron

    # Get some hippocampus connectivity data, based on a conversation with
    # academic researchers on GH:
    # https://github.com/Hippocampome-Org/GraphTheory/issues?q=is%3Aissue+is%3Aclosed
    # scrape hippocamome connectivity data, that I intend to use to program neuromorphic hardware.
    # conditionally get files if they don't exist.


    path_xl = '_hybrid_connectivity_matrix_20171103_092033.xlsx'
    if not os.path.exists(path_xl):
        os.system('wget https://github.com/Hippocampome-Org/GraphTheory/files/1657258/_hybrid_connectivity_matrix_20171103_092033.xlsx')

    xl = pd.ExcelFile(path_xl)
    dfEE = xl.parse()
    dfEE.loc[0].keys()
    dfm = dfEE.as_matrix()

    rcls = dfm[:,:1] # real cell labels.
    rcls = rcls[1:]
    rcls = { k:v for k,v in enumerate(rcls) } # real cell labels, cast to dictionary
    import pickle
    with open('cell_names.p','wb') as f:
        pickle.dump(rcls,f)
    import pandas as pd
    pd.DataFrame(rcls).to_csv('cell_names.csv', index=False)

    filtered = dfm[:,3:]
    filtered = filtered[1:]
    rng = NumpyRNG(seed=64754)
    delay_distr = RandomDistribution('normal', [2, 1e-1], rng=rng)
    weight_distr = RandomDistribution('normal', [45, 1e-1], rng=rng)


    sanity_e = []
    sanity_i = []

    EElist = []
    IIlist = []
    EIlist = []
    IElist = []

    for i,j in enumerate(filtered):
      for k,xaxis in enumerate(j):
        if xaxis == 1 or xaxis == 2:
          source = i
          sanity_e.append(i)
          target = k

        if xaxis ==-1 or xaxis == -2:
          sanity_i.append(i)
          source = i
          target = k

    index_exc = list(set(sanity_e))
    index_inh = list(set(sanity_i))
    import pickle
    with open('cell_indexs.p','wb') as f:
        returned_list = [index_exc, index_inh]
        pickle.dump(returned_list,f)

    import numpy
    a = numpy.asarray(index_exc)
    numpy.savetxt('pickles/'+str(k)+'excitatory_nunber_labels.csv', a, delimiter=",")
    import numpy
    a = numpy.asarray(index_inh)
    numpy.savetxt('pickles/'+str(k)+'inhibitory_nunber_labels.csv', a, delimiter=",")

    for i,j in enumerate(filtered):
      for k,xaxis in enumerate(j):
        if xaxis==1 or xaxis == 2:
          source = i
          sanity_e.append(i)
          target = k
          delay = delay_distr.next()
          weight = 1.0
          if target in index_inh:
             EIlist.append((source,target,delay,weight))
          else:
             EElist.append((source,target,delay,weight))

        if xaxis==-1 or xaxis == -2:
          sanity_i.append(i)

          source = i
          target = k
          delay = delay_distr.next()
          weight = 1.0
          if target in index_exc:
              IElist.append((source,target,delay,weight))
          else:
              IIlist.append((source,target,delay,weight))


    internal_conn_ee = sim.FromListConnector(EElist)
    ee = internal_conn_ee.conn_list

    ee_srcs = ee[:,0]
    ee_tgs = ee[:,1]

    internal_conn_ie = sim.FromListConnector(IElist)
    ie = internal_conn_ie.conn_list
    ie_srcs = set([ int(e[0]) for e in ie ])
    ie_tgs = set([ int(e[1]) for e in ie ])

    internal_conn_ei = sim.FromListConnector(EIlist)
    ei = internal_conn_ei.conn_list
    ei_srcs = set([ int(e[0]) for e in ei ])
    ei_tgs = set([ int(e[1]) for e in ei ])

    internal_conn_ii = sim.FromListConnector(IIlist)
    ii = internal_conn_ii.conn_list
    ii_srcs = set([ int(e[0]) for e in ii ])
    ii_tgs = set([ int(e[1]) for e in ii ])

    for e in internal_conn_ee.conn_list:
        assert e[0] in ee_srcs
        assert e[1] in ee_tgs

    for i in internal_conn_ii.conn_list:
        assert i[0] in ii_srcs
        assert i[1] in ii_tgs


    ml = len(filtered[1])+1
    pre_exc = []
    post_exc = []
    pre_inh = []
    post_inh = []


    rng = NumpyRNG(seed=64754)
    delay_distr = RandomDistribution('normal', [2, 1e-1], rng=rng)

    plot_EE = np.zeros(shape=(ml,ml), dtype=bool)
    plot_II = np.zeros(shape=(ml,ml), dtype=bool)
    plot_EI = np.zeros(shape=(ml,ml), dtype=bool)
    plot_IE = np.zeros(shape=(ml,ml), dtype=bool)

    for i in EElist:
        plot_EE[i[0],i[1]] = int(0)
        #plot_ss[i[0],i[1]] = int(1)

        if i[0]!=i[1]: # exclude self connections
            plot_EE[i[0],i[1]] = int(1)

            pre_exc.append(i[0])
            post_exc.append(i[1])



    assert len(pre_exc) == len(post_exc)
    for i in IIlist:
        plot_II[i[0],i[1]] = int(0)
        if i[0]!=i[1]:
            plot_II[i[0],i[1]] = int(1)
            pre_inh.append(i[0])
            post_inh.append(i[1])

    for i in IElist:
        plot_IE[i[0],i[1]] = int(0)
        if i[0]!=i[1]: # exclude self connections
            plot_IE[i[0],i[1]] = int(1)
            pre_inh.append(i[0])
            post_inh.append(i[1])

    for i in EIlist:
        plot_EI[i[0],i[1]] = int(0)
        if i[0]!=i[1]:
            plot_EI[i[0],i[1]] = int(1)
            pre_exc.append(i[0])
            post_exc.append(i[1])

    plot_excit = plot_EI + plot_EE
    plot_inhib = plot_IE + plot_II

    assert len(pre_inh) == len(post_inh)

    num_exc = [ i for i,e in enumerate(plot_excit) if sum(e) > 0 ]
    num_inh = [ y for y,i in enumerate(plot_inhib) if sum(i) > 0 ]

    # the network is dominated by inhibitory neurons, which is unusual for modellers.
    assert num_inh > num_exc
    assert np.sum(plot_inhib) > np.sum(plot_excit)
    assert len(num_exc) < ml
    assert len(num_inh) < ml
    # # Plot all the Projection pairs as a connection matrix (Excitatory and Inhibitory Connections)

    import pickle
    with open('graph_inhib.p','wb') as f:
       pickle.dump(plot_inhib,f, protocol=2)


    import pickle
    with open('graph_excit.p','wb') as f:
       pickle.dump(plot_excit,f, protocol=2)


    #with open('cell_names.p','wb') as f:
    #    pickle.dump(rcls,f)
    import pandas as pd
    pd.DataFrame(plot_EE).to_csv('ee.csv', index=False)

    import pandas as pd
    pd.DataFrame(plot_IE).to_csv('ie.csv', index=False)

    import pandas as pd
    pd.DataFrame(plot_II).to_csv('ii.csv', index=False)

    import pandas as pd
    pd.DataFrame(plot_EI).to_csv('ei.csv', index=False)


    from scipy.sparse import coo_matrix
    m = np.matrix(filtered[1:])

    bool_matrix = np.add(plot_excit,plot_inhib)
    with open('bool_matrix.p','wb') as f:
       pickle.dump(bool_matrix,f, protocol=2)

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    Gexc_ud = nx.Graph(plot_excit)
    avg_clustering = nx.average_clustering(Gexc_ud)#, nodes=None, weight=None, count_zeros=True)[source]

    rc = nx.rich_club_coefficient(Gexc_ud,normalized=False)
    print('This graph structure as rich as: ',rc[0])
    gexc = nx.DiGraph(plot_excit)

    gexcc = nx.betweenness_centrality(gexc)
    top_exc = sorted(([ (v,k) for k, v in dict(gexcc).items() ]), reverse=True)

    in_degree = gexc.in_degree()
    top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
    in_hub = top_in[-1][1]
    out_degree = gexc.out_degree()
    top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
    out_hub = top_out[-1][1]
    mean_out = np.mean(list(out_degree.values()))
    mean_in = np.mean(list(in_degree.values()))

    mean_conns = int(mean_in + mean_out/2)

    k = 2 # number of neighbouig nodes to wire.
    p = 0.25 # probability of instead wiring to a random long range destination.
    ne = len(plot_excit)# size of small world network
    small_world_ring_excit = nx.watts_strogatz_graph(ne,mean_conns,0.25)



    k = 2 # number of neighbouring nodes to wire.
    p = 0.25 # probability of instead wiring to a random long range destination.
    ni = len(plot_inhib)# size of small world network
    small_world_ring_inhib   = nx.watts_strogatz_graph(ni,mean_conns,0.25)


    nproc = sim.num_processes()
    nproc = 8
    host_name = socket.gethostname()
    node_id = sim.setup(timestep=0.01, min_delay=1.0)#, **extra)
    print("Host #%d is on %s" % (node_id + 1, host_name))
    rng = NumpyRNG(seed=64754)

    #pop_size = len(num_exc)+len(num_inh)
    #num_exc = [ i for i,e in enumerate(plot_excit) if sum(e) > 0 ]
    #num_inh = [ y for y,i in enumerate(plot_inhib) if sum(i) > 0 ]
    #pop_exc =  sim.Population(len(num_exc), sim.Izhikevich(a=0.02, b=0.2, c=-65, d=8, i_offset=0))
    #pop_inh = sim.Population(len(num_inh), sim.Izhikevich(a=0.02, b=0.25, c=-65, d=2, i_offset=0))


    #index_exc = list(set(sanity_e))
    #index_inh = list(set(sanity_i))
    all_cells = sim.Population(len(index_exc)+len(index_inh), sim.Izhikevich(a=0.02, b=0.2, c=-65, d=8, i_offset=0))
    #all_cells = None
    #all_cells = pop_exc + pop_inh
    pop_exc = sim.PopulationView(all_cells,index_exc)
    pop_inh = sim.PopulationView(all_cells,index_inh)
    #print(pop_exc)
    #print(dir(pop_exc))
    for pe in pop_exc:
        print(pe)
        #import pdb
        pe = all_cells[pe]
        #pdb.set_trace()
        #pe = all_cells[i]
        r = random.uniform(0.0, 1.0)
        pe.set_parameters(a=0.02, b=0.2, c=-65+15*r, d=8-r**2, i_offset=0)
        #pop_exc.append(pe)

    #pop_exc = sim.Population(pop_exc)
    for pi in index_inh:
        pi = all_cells[pi]
        #print(pi)
        #pi = all_cells[i]
        r = random.uniform(0.0, 1.0)
        pi.set_parameters(a=0.02+0.08*r, b=0.25-0.05*r, c=-65, d= 2, i_offset=0)
        #pop_inh.append(pi)
    #pop_inh = sim.Population(pop_inh)

    '''
    for pe in pop_exc:
        r = random.uniform(0.0, 1.0)
        pe.set_parameters(a=0.02, b=0.2, c=-65+15*r, d=8-r**2, i_offset=0)

    for pi in pop_inh:
        r = random.uniform(0.0, 1.0)
        pi.set_parameters(a=0.02+0.08*r, b=0.25-0.05*r, c=-65, d= 2, i_offset=0)
    '''
    NEXC = len(num_exc)
    NINH = len(num_inh)

    exc_syn = sim.StaticSynapse(weight = wg, delay=delay_distr)
    assert np.any(internal_conn_ee.conn_list[:,0]) < ee_srcs.size
    prj_exc_exc = sim.Projection(all_cells, all_cells, internal_conn_ee, exc_syn, receptor_type='excitatory')
    prj_exc_inh = sim.Projection(all_cells, all_cells, internal_conn_ei, exc_syn, receptor_type='excitatory')
    inh_syn = sim.StaticSynapse(weight = wg, delay=delay_distr)
    delay_distr = RandomDistribution('normal', [1, 100e-3], rng=rng)
    prj_inh_inh = sim.Projection(all_cells, all_cells, internal_conn_ii, inh_syn, receptor_type='inhibitory')
    prj_inh_exc = sim.Projection(all_cells, all_cells, internal_conn_ie, inh_syn, receptor_type='inhibitory')
    inh_distr = RandomDistribution('normal', [1, 2.1e-3], rng=rng)


    def prj_change(prj,wg):
        prj.setWeights(wg)
    prj_change(prj_exc_exc,wg)
    prj_change(prj_exc_inh,wg)
    prj_change(prj_inh_exc,wg)
    prj_change(prj_inh_inh,wg)

    def prj_check(prj):
        for w in prj.weightHistogram():
            for i in w:
                print(i)
    prj_check(prj_exc_exc)
    prj_check(prj_exc_inh)
    prj_check(prj_inh_exc)
    prj_check(prj_inh_inh)

    #print(rheobase['value'])
    #print(float(rheobase['value']),1.25/1000.0)
    '''Old values that worked
    noise = sim.NoisyCurrentSource(mean=0.85/1000.0, stdev=5.00/1000.0, start=0.0, stop=2000.0, dt=1.0)
    pop_exc.inject(noise)
    #1000.0 pA


    noise = sim.NoisyCurrentSource(mean=1.740/1000.0, stdev=5.00/1000.0, start=0.0, stop=2000.0, dt=1.0)
    pop_inh.inject(noise)
    #1750.0 pA
    '''

    noise = sim.NoisyCurrentSource(mean=0.74/1000.0, stdev=4.00/1000.0, start=0.0, stop=2000.0, dt=1.0)
    pop_exc.inject(noise)
    #1000.0 pA


    noise = sim.NoisyCurrentSource(mean=1.440/1000.0, stdev=4.00/1000.0, start=0.0, stop=2000.0, dt=1.0)
    pop_inh.inject(noise)

    ##
    # Setup and run a simulation. Note there is no current injection into the neuron.
    # All cells in the network are in a quiescent state, so its not a surprise that xthere are no spikes
    ##

    sim = pyNN.neuron
    arange = np.arange
    import re
    all_cells.record(['v','spikes'])  # , 'u'])
    all_cells.initialize(v=-65.0, u=-14.0)
    # === Run the simulation =====================================================
    tstop = 2000.0
    sim.run(tstop)
    data = None
    data = all_cells.get_data().segments[0]

    #print(len(data.analogsignals[0].times))
    with open('pickles/qi'+str(wg)+'.p', 'wb') as f:
        pickle.dump(data,f)
    # make data none or else it will grow in a loop
    all_cells = None
    data = None
    noise = None



#iter_sim = [ (i,wg) for i,wg in enumerate(weight_gain_factors.keys()) ]
#import dask.bag as db
#iter_sim = db.from_sequence(iter_sim,4)
#from itertools import repeat
#_ = list(map(map_sim,iter_sim,repeat(sim)))
#_ = list(db.map(map_sim,iter_sim).compute());
