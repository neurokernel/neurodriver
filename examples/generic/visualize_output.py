#!/usr/bin/env python

"""
Visualize generic LPU demo output.

Notes
-----
Generate demo output by running

python generic_demo.py
"""

import numpy as np
import matplotlib as mpl
mpl.use('agg')

import h5py
import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

# Select IDs of spiking projection neurons:
G = nx.read_gexf('./data/generic_lpu.gexf.gz')
neu_proj = sorted([k for k, n in G.node.items() if \
                   n['name'][:4] == 'proj' and \
                   n['class'] == 'LeakyIAF'])

in_uid = 'sensory_0'
        
N = len(neu_proj)

V = vis.visualizer()
V.add_LPU('./data/generic_input.h5', LPU='Sensory', is_input=True)
V.add_plot({'type':'waveform', 'uids': [[in_uid]], 'variable':'I'},
           'input_Sensory')

V.add_LPU('new_output.h5',  'Generic LPU',
          gexf_file='./data/generic_lpu.gexf.gz')
V.add_plot({'type':'raster', 'uids': [neu_proj], 'variable': 'spike_state',
            'yticks': range(1, 1+N),
            'yticklabels': neu_proj, 'title': 'Output'},
            'Generic LPU')

V.rows = 2
V.cols = 1
V.fontsize = 8
V.xlim = [0, 1.0]

gen_video = True
if gen_video:
    V.out_filename = 'generic_output.mp4'
    V.codec = 'mpeg4'
    V.run()
else:
    V.update_interval = None
    V.run('generic_output.png')
