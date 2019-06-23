#!/usr/bin/env python

"""
Visualize output of intro demo.
"""

import concurrent.futures

import matplotlib as mpl

mpl.use('agg')

import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                        'true':True, 'True':True}

def run(out_name):
    V = vis.visualizer()

    # Assumes that generic_lpu_0_input.h5 and generic_lpu_1_input.h5
    # contain the same data:
    V.add_LPU('./data/generic_lpu_0_input.h5', LPU='Sensory', is_input=True)
    V.add_plot({'type': 'waveform', 'uids': [['sensory_0']], 'variable':'I'},
                'input_Sensory')

    for i in [0, 1]:
        G = nx.read_gexf('./data/generic_lpu_%s.gexf.gz' % i)
        neu_proj = sorted([k for k, n in G.node.items() if \
                           n['name'][:4] == 'proj' and \
                           n['class'] == 'LeakyIAF'])
        N = len(neu_proj)
        V.add_LPU('generic_lpu_%s_%s_output.h5' % (i, out_name),
                  'Generic LPU %s' % i,
                  gexf_file='./data/generic_lpu_%s.gexf.gz' % i)
        V.add_plot({'type': 'raster', 'uids': [neu_proj],
                    'variable': 'spike_state',
                    'yticks': range(1, 1+N),
                    'yticklabels': neu_proj, 'title': 'Output'},
                    'Generic LPU %s' % i)

    V.rows = 3
    V.cols = 1
    V.fontsize = 8
    V.out_filename = '%s.mp4' % out_name
    V.codec = 'mpeg4'
    V.xlim = [0, 1.0]
    V.run()
    #V.run('%s.png' % out_name)

# Run the visualizations in parallel:
with concurrent.futures.ProcessPoolExecutor() as executor:
    fs_dict = {}
    for out_name in ['un', 'co']:
        res = executor.submit(run, out_name)
        fs_dict[out_name] = res
    concurrent.futures.wait(fs_dict.values())

    # Report any exceptions that may have occurred:
    for k in fs_dict:
        e = fs_dict[k].exception()
        if e:
            print '%s: %s' % (k, e)
