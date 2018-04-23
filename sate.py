# coding: utf-8
import matplotlib as mpl
mpl.use('Agg')
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from natsort import natsorted, ns
import elephant.conversion as conv
import glob
import elephant
import elephant.conversion as conv
import neo as n
import quantities as pq
from neo.core import analogsignal
from elephant.statistics import cv
import elephant
import numpy as np


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
    binary_trains = []
    for spiketrain in mdf1.spiketrains:
        x = conv.BinnedSpikeTrain(spiketrain, binsize=5 * pq.ms, t_start=0 * pq.s)
        binary_trains.append(x.to_array())
    print(binary_trains)
    dat = Data(np.array(binary_trains), dim_order='spr')
    dat.n_procs = n_procs
    settings = {'cmi_estimator': 'JidtKraskov',
            'max_lag_sources': 3,
            'max_lag_target': 3,
            'min_lag_sources': 1}
    print(dat)
    mte = MultivariateTE()
    res_full = mte.analyse_network(settings=settings, data=dat)

    # generate graph plots
    g_single = visualise_graph.plot_selected_vars(res_single, mte)
    g_full = visualise_graph.plot_network(res_full)

iter_distances = natsorted(glob.glob('pickles/qi*.p'))
mdfl = pickle.load(open(iter_distances[0],'rb'))
mdfloop = {}
te(mdfl)
