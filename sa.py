import matplotlib as mpl
mpl.use('Agg')

# coding: utf-8
import os
#os.system('bash install.sh')
import matplotlib.pyplot as plt
#mpl.use('Agg')

#plt.use('Agg')
import seaborn as sns
mpl.use('Agg')

os.system('pip install pyspike')
os.system('pip install natsort')
os.system('pip uninstall neo -y')
os.system('pip install neo')

import pickle
import numpy as np
import pyspike

from natsort import natsorted, ns

import elephant.conversion as conv
import pyspike as spk
import glob
from elephant.spectral import welch_cohere
import elephant
import elephant.conversion as conv
import neo as n
import quantities as pq
from neo.core import analogsignal
from elephant.statistics import cv
import matplotlib.pyplot as plt
import elephant
import numpy as np


import pyspike as spk


def te(mdf1):
    import numpy as np
    from idtxl.multivariate_te import MultivariateTE
    from idtxl.data import Data
    n_procs = 1

    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'n_perm_max_stat': 21,
        'max_lag_target': 5,
        'max_lag_sources': 5,
        'min_lag_sources': 4}
    settings['cmi_estimator'] = 'JidtDiscreteCMI'
    #JidtKraskovCMI
    binary_trains = []
    for spiketrain in mdf1.spiketrains:
        x = conv.BinnedSpikeTrain(spiketrain, binsize=5 * pq.ms, t_start=0 * pq.s)
        binary_trains.append(x.to_array())
    print(binary_trains)
    dat = Data(np.array(binary_trains), dim_order='spr')
    dat.n_procs = n_procs

    #import sklearn
    #NMF = sklearn.decomposition.NMF(sts)
    #print(NMF)

    settings = {'cmi_estimator': 'JidtKraskov',
            'max_lag_sources': 3,
            'max_lag_target': 3,
            'min_lag_sources': 1}
    print(dat)
    mte = MultivariateTE()

    #res_single = mte.analyse_single_target(settings=settings, data=data, target=3)
    res_full = mte.analyse_network(settings=settings, data=dat)

    # generate graph plots
    g_single = visualise_graph.plot_selected_vars(res_single, mte)
    g_full = visualise_graph.plot_network(res_full)
    #nw_0._initialise(settings, np.array(yy), 'all', target)
    #nw.analyse_single_target(settings=settings, data=dat, target=1)

iter_distances = natsorted(glob.glob('pickles/qi*.p'))
#print(iter_distances,'blah')

mdfl = pickle.load(open(iter_distances[0],'rb'))


mdfloop = {}

'''
for k,i in enumerate(iter_distances):
    with open(i, 'rb') as f:
        #from neo.core import analogsignal
        mdfloop[k] = pickle.load(f)

for k,mdf1 in mdfloop.items():
    print(mdf1,k)

titems = [ (k,mdf1) for k,mdf1 in enumerate(mdfloop.values()) ]
#titems = [ (k,mdf1) for k,mdf1 in enumerate(titems) ]
print(titems, 'titems')
import dask.bag as db
'''
#import pdb
#print(dir(mdfl))

#pdb.set_trace()

te(mdfl)



def iter_plot0(md):
    import seaborn as sns
    import pickle
    with open('cell_indexs.p','rb') as f:
        returned_list = pickle.load(f)
    index_exc = returned_list[0]
    index_inh = returned_list[1]
    index, mdf1 = md
    #wgf = {0.025:None,0.05:None,0.125:None,0.25:None,0.3:None,0.4:None,0.5:None,1.0:None,1.5:None,2.0:None,2.5:None,3.0:None}
    wgf = {0.0025:None,0.0125:None,0.025:None,0.05:None,0.125:None,0.25:None,0.3:None,0.4:None,0.5:None,1.0:None,1.5:None,2.0:None,2.5:None,3.0:None}

    weight_gain_factors = {k:v for k,v in enumerate(wgf.keys())}
    print(len(weight_gain_factors))
    print(weight_gain_factors.keys())
    #weight_gain_factors = {0:0.5,1:1.0,2:1.5,3:2.0,4:2.5,5:3}
    #weight_gain_factors = {:None,1.0:None,1.5:None,2.0:None,2.5:None}

    k = weight_gain_factors[index]
    #print(len(mdf1.segments),'length of block')

    ass = mdf1.analogsignals[0]

    time_points = ass.times
    avg = np.mean(ass, axis=0)  # Average over signals of Segment
    #maxx = np.max(ass, axis=0)  # Average over signals of Segment
    std = np.std(ass, axis=0)  # Average over signals of Segment
    #avg_minus =
    plt.figure()
    plt.plot([i for i in range(0,len(avg))], avg)
    plt.plot([i for i in range(0,len(std))], std)

    plt.title("Mean and Standard Dev of $V_{m}$ amplitude per neuron ")
    plt.xlabel('time $(ms)$')
    plt.xlabel('Voltage $(mV)$')

    plt.savefig(str(index)+'prs.png')
    vm_spiking = []
    vm_not_spiking = []
    spike_trains = []
    binary_trains = []
    max_spikes = 0

    vms = np.array(mdf1.analogsignals[0].as_array().T)
    #print(data)
    #for i,vm in enumerate(data):

    cnt = 0
    for spiketrain in mdf1.spiketrains:
        #spiketrain = mdf1.spiketrains[index]
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        #import sklearn
        #sklearn.decomposition.NMF(y)
        # argument edges is the time interval you want to be considered.
        pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,len(ass)))
        spike_trains.append(pspikes)
        if len(spiketrain) > max_spikes:
            max_spikes = len(spiketrain)

        if np.max(ass[spiketrain.annotations['source_id']]) > 0.0:
            vm_spiking.append(vms[spiketrain.annotations['source_id']])
        else:
            vm_not_spiking.append(vms[spiketrain.annotations['source_id']])
        cnt+= 1

    for spiketrain in mdf1.spiketrains:
        x = conv.BinnedSpikeTrain(spiketrain, binsize=1 * pq.ms, t_start=0 * pq.s)
        binary_trains.append(x)
    end_floor = np.floor(float(mdf1.t_stop))
    dt = float(mdf1.t_stop) % end_floor
    mdf1.t_start
    #v = mdf1.take_slice_of_analogsignalarray_by_unit()
    t_axis = np.arange(float(mdf1.t_start), float(mdf1.t_stop), dt)
    plt.figure()
    plt.clf()

    plt.figure()
    plt.clf()
    cleaned = []
    data = np.array(mdf1.analogsignals[0].as_array().T)
    #print(data)
    for i,vm in enumerate(data):
        if np.max(vm) > 900.0 or np.min(vm) < - 900.0:
            pass
        else:
            plt.plot(ass.times,vm)#,label='neuron identifier '+str(i)))
            cleaned.append(vm)
            #vm = s#.as_array()[:,i]

    assert len(cleaned) < len(ass)

    print(len(cleaned))
    plt.title('neuron $V_{m}$')
    #plt.legend(loc="upper left")
    plt.savefig(str('weight_')+str(k)+'analogsignals'+'.png');
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')

    plt.close()


        #pass

    plt.figure()
    plt.clf()
    plt.title('Single Neuron $V_{m}$ trace')
    plt.plot(ass.times[0:int(len(ass.times)/10)],vm_not_spiking[index_exc[0]][0:int(len(ass.times)/10)])
    plt.xlabel('$ms$')
    plt.ylabel('$mV$')
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.savefig(str('weight_')+str(k)+'eespecific_analogsignals'+'.png');
    plt.close()


    plt.figure()
    plt.clf()
    plt.title('Single Neuron $V_{m}$ trace')
    plt.plot(ass.times[0:int(len(ass.times)/10)],vm_not_spiking[index_inh[0]][0:int(len(ass.times)/10)])
    plt.xlabel('$ms$')
    plt.ylabel('$mV$')

    plt.savefig(str('weight_')+str(k)+'inhibitory_analogsignals'+'.png');
    plt.close()

    cvs = [0 for i in range(0,len(spike_trains))]
    cvsd = {}
    cvs = []
    cvsi = []
    rates = [] # firing rates per cell. in spikes a second.
    for i,j in enumerate(spike_trains):
        rates.append(float(len(j)/2.0))
        cva = cv(j)
        if np.isnan(cva) or cva == 0:
            pass
            #cvs[i] = 0
            #cvsd[i] = 0
        else:
            pass
            #cvs[i] = cva
            #cvsd[i] = cva
        cvs.append(cva)
    #import pickle
    #with open(str('weight_')+str(k)+'coefficients_of_variation.p','wb') as f:
    #   pickle.dump([cvs,cvsd],f)
    import numpy
    a = numpy.asarray(cvs)
    numpy.savetxt('pickles/'+str('weight_')+str(k)+'coefficients_of_variation.csv', a, delimiter=",")

    import numpy
    a = numpy.asarray(rates)
    numpy.savetxt('pickles/'+str('weight_')+str(k)+'firing_of_rate.csv', a, delimiter=",")

    cvs = [i for i in cvs if i!=0 ]
    cells = [i for i in range(0,len(cvs))]

    plt.clf()
    fig, axes = plt.subplots()
    axes.set_title('Coefficient of Variation Versus Neuron')
    axes.set_xlabel('Neuron number')
    axes.set_ylabel('CV estimate')
    mcv = np.mean(cvs)
    #plt.scatter(cells,cvs)
    cvs = np.array(cvs)
    plt.scatter(index_inh,cvs[index_inh],label="inhibitory cells")
    plt.scatter(index_exc,cvs[index_exc],label="excitatory cells")
    plt.legend(loc="upper left")


    fig.tight_layout()
    plt.savefig(str('weight_')+str(k)+'cvs_mean_'+str(mcv)+'.png');
    plt.close()


    plt.clf()
    #frequencies, power = elephant.spectral.welch_psd(ass)
    #mfreq = frequencies[np.where(power==np.max(power))[0][0]]
    #fig, axes = plt.subplots()
    axes.set_title('Firing Rate Versus Neuron Number at mean f='+str(np.mean(rates))+str('(Spike Per Second)'))
    axes.set_xlabel('Neuron number')
    axes.set_ylabel('Spikes per second')
    rates = np.array(rates)
    plt.scatter(index_inh,rates[index_inh],label="inhibitory cells")
    plt.scatter(index_exc,rates[index_exc],label="excitatory cells")
    plt.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(str('firing_rates_per_cell_')+str(k)+str(mcv)+'.png');
    plt.close()
    '''
    import pandas as pd
    d = {'coefficent_of_variation': cvs, 'cells': cells}
    df = pd.DataFrame(data=d)

    ax = sns.regplot(x='cells', y='coefficent_of_variation', data=df)#, fit_reg=False)
    plt.savefig(str('weight_')+str(k)+'cvs_regexp_'+str(mcv)+'.png');
    plt.close()
    '''

    spike_trains = []
    ass = mdf1.analogsignals[0]
    tstop = mdf1.t_stop
    np.max(ass.times) == mdf1.t_stop
    #assert tstop == 2000
    tstop = 2000
    vm_spiking = []

    for spiketrain in mdf1.spiketrains:
        vm_spiking.append(mdf1.analogsignals[0][spiketrain.annotations['source_id']])
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']

        # argument edges is the time interval you want to be considered.
        pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,tstop))
        spike_trains.append(pspikes)

    # plot the spike times

    plt.clf()
    for (i, spike_train) in enumerate(spike_trains):
        plt.scatter(spike_train, i*np.ones_like(spike_train), marker='.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Cell identifier')
    plt.title('Raster Plot for weight strength:'+str(k))

    plt.savefig(str('weight_')+str(k)+'raster_plot'+'.png');
    plt.close()


    f = spk.isi_profile(spike_trains, indices=[0, 1])
    x, y = f.get_plottable_data()

    #text_file.close()
    text_file = open(str('weight_')+str(index)+'net_out.txt', 'w')

    plt.figure()
    plt.plot(x, np.abs(y), '--k', label="ISI-profile")
    print("ISI-distance: %.8f" % f.avrg())
    f = spk.spike_profile(spike_trains, indices=[0, 1])
    x, y = f.get_plottable_data()
    plt.plot(x, y, '-b', label="SPIKE-profile")
    #print("SPIKE-distance: %.8f" % f.avrg())
    string_to_write = str("ISI-distance:")+str(f.avrg())+str("\n\n")
    plt.title(string_to_write)
    plt.xlabel('Time $(ms)$')
    plt.ylabel('ISI distance')
    plt.legend(loc="upper left")
    plt.savefig(str('weight_')+str(k)+'ISI_distance_bivariate'+'.png');plt.close()
    text_file.write(string_to_write)




    #text_file.write("SPIKE-distance: %.8f" % f.avrg())
    #text_file.write("\n\n")


    plt.figure()
    f = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
    x, y = f.get_plottable_data()
    plt.plot(x, y, '--ok', label="SPIKE-SYNC profile")
    print(f,f.avrg())
    print("Average:"+ str(f.avrg()))
    #print(len(f.avrg()),f.avrg())
    string_to_write = str("instantaneous synchrony:")+str(f.avrg())+'weight: '+str(index)

    plt.title(string_to_write)
    plt.xlabel('Time $(ms)$')
    plt.ylabel('instantaneous synchrony')

    text_file.write(string_to_write)

    #text_file.write(list())

    f = spk.spike_profile(spike_trains[0], spike_trains[1])
    x, y = f.get_plottable_data()

    plt.plot(x, y, '-b', label="SPIKE-profile")
    plt.axis([0, 4000, -0.1, 1.1])
    plt.legend(loc="center right")
    plt.clf()
    plt.figure()
    plt.subplot(211)

    f = spk.spike_sync_profile(spike_trains)
    x, y = f.get_plottable_data()
    plt.plot(x, y, '-b', alpha=0.7, label="SPIKE-Sync profile")
    x1, y1 = f.get_plottable_data(averaging_window_size=50)
    plt.plot(x1, y1, '-k', lw=2.5, label="averaged SPIKE-Sync profile")
    plt.subplot(212)

    f_psth = spk.psth(spike_trains, bin_size=50.0)
    x, y = f_psth.get_plottable_data()
    plt.plot(x, y, '-k', alpha=1.0, label="PSTH")

    plt.savefig(str('weight_')+str(k)+'multivariate_PSTH'+'.png');plt.close()
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Spikes per bin')

    plt.clf()
    plt.figure()

    f_psth = spk.psth(spike_trains, bin_size=50.0)
    x, y = f_psth.get_plottable_data()
    plt.plot(x, y, '-k', alpha=1.0, label="PSTH")


    plt.savefig(str('weight_')+str(k)+'exclusively_PSTH'+'.png');plt.close()


    plt.figure()
    isi_distance = spk.isi_distance_matrix(spike_trains)
    plt.imshow(isi_distance, interpolation='none')
    plt.title('Pairwise ISI distance, T=0-2000')
    plt.xlabel('post-synaptic neuron number')
    plt.ylabel('pre-synaptic neuron number')

    plt.title("ISI-distance")
    plt.savefig(str('weight_')+str(k)+'ISI_distance'+'.png');plt.close()

    #plt.show()

    plt.figure()
    plt.clf()
    import seaborn as sns

    sns.set()
    sns.clustermap(isi_distance)#,vmin=-,vmax=1);

    plt.savefig(str('weight_')+str(k)+'cluster_isi_distance'+'.png');plt.close()



    plt.figure()
    spike_distance = spk.spike_distance_matrix(spike_trains, interval=(0, float(tstop)))


    import pickle
    with open('spike_distance_matrix.p','wb') as f:
       pickle.dump(spike_distance,f)

    plt.imshow(spike_distance, interpolation='none')
    plt.title("Pairwise SPIKE-distance, T=0-2000")
    plt.xlabel('post-synaptic neuron number')
    plt.ylabel('pre-synaptic neuron number')


    plt.savefig(str('weight_')+str(k)+'spike_distance_matrix'+'.png');plt.close()
    plt.figure()
    plt.clf()
    sns.set()
    sns.clustermap(spike_distance);

    plt.savefig(str('weight_')+str(k)+'cluster_spike_distance'+'.png');plt.close()


    plt.figure()
    spike_sync = spk.spike_sync_matrix(spike_trains, interval=(0, float(tstop)))
    plt.imshow(spike_sync, interpolation='none')
    plt.title('Pairwise Spike Synchony, T=0-2000')
    plt.xlabel('post-synaptic neuron number')
    plt.ylabel('pre-synaptic neuron number')

    import numpy
    a = numpy.asarray(spike_sync)
    numpy.savetxt("spike_sync_matrix.csv", a, delimiter=",")

    plt.figure()
    plt.clf()
    sns.clustermap(spike_sync);
    plt.savefig(str('weight_')+str(k)+'cluster_spike_sync_distance'+'.png');plt.close()


def iter_plot1(md):
    index, mdf1 = md
    wgf = {0.025:None,0.05:None,0.125:None,0.25:None,0.3:None,0.4:None,0.5:None,1.0:None,1.5:None,2.0:None,2.5:None,3.0:None}
    weight_gain_factors = {k:v for k,v in enumerate(wgf.keys())}
    k = weight_gain_factors[index]
    ass = mdf1.analogsignals[0]
    vm_spiking = []
    vm_not_spiking = []
    spike_trains = []
    binary_trains = []
    max_spikes = 0
    cnt = 0
    for spiketrain in mdf1.spiketrains:
        #spiketrain = mdf1.spiketrains[index]
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        # argument edges is the time interval you want to be considered.
        pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,len(ass)))
        spike_trains.append(pspikes)
        if len(spiketrain) > max_spikes:
            max_spikes = len(spiketrain)

        if np.max(ass[spiketrain.annotations['source_id']]) > 0.0:
            vm_spiking.append(ass[spiketrain.annotations['source_id']])
        else:
            vm_not_spiking.append(ass[spiketrain.annotations['source_id']])
        cnt+= 1

    import elephant
    from scipy.signal import periodogram
    #dt = 0.0025
    #frequencies, power = periodogram(ass,fs=1/dt)
    frequencies, power = elephant.spectral.welch_psd(ass)

    mfreq = frequencies[np.where(power==np.max(power))[0][0]]
    import pickle
    with open(str(k)+'_'+str(mfreq)+'_'+'mfreq.p','wb') as f:
       pickle.dump(mfreq,f)

    def plot_periodogram(frequencies,power):
        plt.figure(figsize=(10,4))
        sns.heatmap(power)
        plt.xlabel('Frequency ($Hz$)')
        plt.ylabel('Power pre neuron ($V^2/Hz$)') # Note that power is now
                                       # a normalized density
        plt.savefig(str('weight_')+str(k)+'multi_variate_periodogram'+'.png');
        plt.close()

        plt.plot(frequencies,power[0])
        plt.savefig(str('weight_')+str(k)+'_single_neuron_periodogram'+'.png');

        return
    plot_periodogram(frequencies,power)

    lens = np.shape(ass.as_array())[1]
    coherance_matrix = np.zeros(shape=(lens,lens), dtype=float)
    for i in range(0,lens):
        for j in range(0,lens):
            if i!=j:
                x = ass.as_array()[i]
                y = ass.as_array()[j]
                coh = welch_cohere(x,y)
                if np.mean(coh) != 0:
                    coherance_matrix[i,j] = np.mean(coh)
    plt.figure()
    plt.clf()
    from matplotlib.colors import LogNorm
    #plt.imshow(coherance_matrix, interpolation='none',norm=cbar_kws)
    sns.heatmap(coherance_matrix)#,cbar_kws=cbar_kws)
    plt.title("Coherance Matrix")
    plt.xlabel('pre-synaptic cell')
    plt.ylabel('post-synaptic cell')

    plt.savefig(str('Coherance_matrix_weight_')+str(k)+str('freq_')+str(mfreq)+'.png');
    plt.close()

    import numpy
    a = numpy.asarray(coherance_matrix)
    numpy.savetxt("coherance_matrix.csv", a, delimiter=",")
    mdf1 = None
    coh = None

#grid = db.from_sequence(titems,npartitions = 3)
#_ = list(db.map(iter_plot0,grid).compute());

#grid = db.from_sequence(titems,npartitions = 3)
#_ = list(db.map(iter_plot1,grid).compute());

#from sciunit.utils import NotebookTools
#NotebookTools.do_notebook(name='Distribution')
import pcae
import pcai

import pickle
with open('bool_matrix.p','rb') as f:
   m = pickle.load(f)
print(type(m))
print(m)
import networkx as nx
G = nx.DiGraph(m)
in_degree = G.in_degree()
top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
in_hub = top_in[-1][1]
out_degree = G.out_degree()
top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
out_hub = top_out[-1][1]
sns.clustermap(m);
plt.figure()
plt.clf()
plt.imshow(m, interpolation='none')
plt.title("connection matrix")

mean_out = np.mean(list(out_degree.values()))
mean_in = np.mean(list(in_degree.values()))

mean_conns = int(np.floor(mean_in + mean_out)/2)

k = 2 # number of neighbouring nodes to wire.
p = 0.25 # probability of instead wiring to a random long range destination.
ne = np.shape(m)[0]# size of small world network
print(ne)
swe = nx.watts_strogatz_graph(ne,mean_conns,0.25)

#import networkx as nx
Gswe = nx.DiGraph(swe)
in_degree = Gswe.in_degree()
top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
in_hub = top_in[-1][1]
out_degree = Gswe.out_degree()
top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
out_hub = top_out[-1][1]


lswe = Gswe.adjacency_list()
excit_sm = np.zeros(shape=(ne,ne), dtype=bool)
#inhib_sm = np.zeros(shape=(ne,ne), dtype=bool)

for i,j in enumerate(lswe):
    for k in j:
        excit_sm[i,k] = int(1)

plt.figure()
plt.clf()
plt.imshow(excit_sm, interpolation='none')
plt.title("Small World connection matrix")
plt.show()

import numpy
a = numpy.asarray(excit_sm)
numpy.savetxt("small_world.csv", a, delimiter=",")
