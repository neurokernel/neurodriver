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

import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

# Select IDs of spiking projection neurons:
G = nx.read_gexf('./data/generic_lpu.gexf.gz')
neu_proj = sorted([int(k) for k, n in G.node.items() if \
                   n['name'][:4] == 'proj' and \
                   n['spiking']])
N = len(neu_proj)

V = vis.visualizer()
V.add_LPU('./data/generic_input.h5', LPU='Sensory')
V.add_plot({'type':'waveform', 'ids': [[0]]}, 'input_Sensory')

V.add_LPU('generic_output_spike.h5',
          './data/generic_lpu.gexf.gz', 'Generic LPU')
V.add_plot({'type':'raster', 'ids': {0: range(N)},
            'yticks': range(1, 1+N),
            'yticklabels': neu_proj},
            'Generic LPU','Output')

V.rows = 2
V.cols = 1
V.fontsize = 18
V.dt = 0.0001
V.xlim = [0, 1.0]

gen_video = False
if gen_video:
    V.out_filename = 'generic_output.avi'
    V.codec = 'libtheora'
    V.run()
else:
    V.update_interval = None
    V.run('generic_output.png')
