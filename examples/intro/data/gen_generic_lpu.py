#!/usr/bin/env python

"""
Create generic LPU and simple pulse input signal.
"""

from itertools import product
import sys

import numpy as np
import h5py
import networkx as nx


def create_lpu_graph(lpu_name, N_sensory, N_local, N_proj):
    """
    Create a generic LPU graph.

    Creates a graph containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The graph
    also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model.

    Parameters
    ----------
    lpu_name : str
        Name of LPU. Used in port identifiers.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.

    Returns
    -------
    g : networkx.MultiDiGraph
        Generated graph.
    """

    # Set numbers of neurons:
    neu_type = ('sensory', 'local', 'proj')
    neu_num = (N_sensory, N_local, N_proj)

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.MultiDiGraph()

    in_port_idx = 0
    spk_out_id = 0
    gpot_out_id = 0

    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            id = t + "_" + str(i)
            name = t + "_" + str(i)

            # Half of the sensory neurons and projection neurons are
            # spiking neurons. The other half are graded potential neurons.
            # All local neurons are graded potential only.
            if t != 'local' and np.random.rand() < 0.5:
                G.add_node(id,
                           **{'class': 'LeakyIAF',
                              'name': name + '_s',
                              'initV': np.random.uniform(-60.0, -25.0),
                              'reset_potential': -67.5489770451,
                              'resting_potential': 0.0,
                              'threshold': -25.1355161007,
                              'resistance': 1002.445570216,
                              'capacitance': 0.0669810502993,
                              'circuit': 'proj' if t == 'proj' else 'local'
                              })

                # Projection neurons are all assumed to be attached to output
                # ports (which are represented as separate nodes):
                if t == 'proj':
                    G.add_node(id + '_port',
                               **{'class': 'Port',
                                  'name': name + 'port',
                                  'port_type': 'spike',
                                  'port_io': 'out',
                                  'selector': '/%s/out/spk/%s' % (lpu_name, str(spk_out_id))
                                  })
                    G.add_edge(id, id + '_port')
                    spk_out_id += 1
                else:
                    # An input port node is created for and attached to each non-projection
                    # neuron with a synapse; this assumes that data propagates from one LPU to
                    # another as follows:
                    # LPU0[projection neuron] -> LPU0[output port] -> LPU1[input port] ->
                    # LPU1[synapse] -> LPU1[non-projection neuron]
                    G.add_node('in_port' + str(in_port_idx),
                               **{'class': 'Port',
                                  'name': 'in_port' + str(in_port_idx),
                                  'port_type': 'spike',
                                  'port_io': 'in',
                                  'selector': '/%s/in/spk/%s' % (lpu_name, in_port_idx)
                                  })
                    G.add_node('synapse_' + 'in_port' + str(in_port_idx) + '_to_' + id,
                               **{'class': 'AlphaSynapse',
                                  'name': 'in_port' + str(in_port_idx) + '-' + name,
                                  'ad': 0.19 * 1000,
                                  'ar': 1.1 * 100,
                                  'gmax': 0.003 * 1e-3,
                                  'reverse': 65.0,
                                  'circuit': 'local'
                                  })
                    G.add_edge('in_port' + str(in_port_idx),
                               'synapse_' + 'in_port' + str(in_port_idx) + '_to_' + id)
                    G.add_edge('synapse_' + 'in_port' + str(in_port_idx) + '_to_' + id,
                               id)
                    in_port_idx += 1
            else:
                G.add_node(id,
                           **{'class': "MorrisLecar",
                              'name': name + '_g',
                              'V1': 30.,
                              'V2': 15.,
                              'V3': 0.,
                              'V4': 30.,
                              'phi': 0.025,
                              'offset': 0.,
                              'V_L': -50.,
                              'V_Ca': 100.0,
                              'V_K': -70.0,
                              'g_Ca': 1.1,
                              'g_K': 2.0,
                              'g_L': 0.5,
                              'initV': -52.14,
                              'initn': 0.02,
                              'circuit': 'proj' if t == 'proj' else 'local'
                              })

                # Projection neurons are all assumed to be attached to output
                # ports (which are not represented as separate nodes):
                if t == 'proj':
                    G.add_node(id + '_port',
                               **{'class': 'Port',
                                  'name': name + 'port',
                                  'port_type': 'gpot',
                                  'port_io': 'out',
                                  'selector': '/%s/out/gpot/%s' % (lpu_name, str(gpot_out_id))
                                  })
                    G.add_edge(id, id + '_port')
                    gpot_out_id += 1
                else:
                    G.add_node('in_port' + str(in_port_idx),
                               **{'class': 'Port',
                                  'name': 'in_port' + str(in_port_idx),
                                  'port_type': 'gpot',
                                  'port_io': 'in',
                                  'selector': '/%s/in/gpot/%s' % (lpu_name, in_port_idx)
                                  })
                    G.add_node('synapse_' + 'in_port' + str(in_port_idx) + '_to_' + id,
                               **{'class': 'PowerGPotGPot',
                                  'name': 'in_port' + str(in_port_idx) + '-' + name,
                                  'reverse': -80.0,
                                  'saturation': 0.03 * 1e-3,
                                  'slope': 0.8 * 1e-6,
                                  'power': 1.0,
                                  'threshold': -50.0,
                                  'circuit': 'local'
                                  })
                    G.add_edge('in_port' + str(in_port_idx),
                               'synapse_' + 'in_port' +
                               str(in_port_idx) + '_to_' + id,
                               delay=0.001)
                    G.add_edge('synapse_' + 'in_port' + str(in_port_idx) + '_to_' + id,
                               id)
                    in_port_idx += 1

    # Assume a probability of synapse existence for each group of synapses:
    # sensory -> local, sensory -> projection, local -> projection,
    # projection -> local:
    for r, (i, j) in zip((0.5, 0.1, 0.1, 0.3),
                         ((0, 1), (0, 2), (1, 2), (2, 1))):
        for src, tar in product(range(neu_num[i]), range(neu_num[j])):

            # Don't connect all neurons:
            if np.random.rand() > r:
                continue

            # Connections from the sensory neurons use the alpha function model;
            # all other connections use the power_gpot_gpot model:
            pre_id = neu_type[i] + "_" + str(src)
            post_id = neu_type[j] + "_" + str(tar)
            name = G.node[pre_id]['name'] + '-' + G.node[post_id]['name']
            synapse_id = 'synapse_' + name
            if G.node[pre_id]['class'] is 'LeakyIAF':
                G.add_node(synapse_id,
                           **{'class': 'AlphaSynapse',
                              'name': name,
                              'ar': 1.1 * 1e2,
                              'ad': 1.9 * 1e3,
                              'reverse': 65.0 if G.node[post_id]['class'] is 'LeakyIAF' else 10.0,
                              'gmax': 3 * 1e-6 if G.node[post_id]['class'] is 'LeakyIAF' else 3.1e-7,
                              'circuit': 'local'})
                G.add_edge(pre_id, synapse_id)
                G.add_edge(synapse_id, post_id)
            else:
                G.add_node(synapse_id,
                           **{'class': 'PowerGPotGPot',
                              'name': name,
                              'slope': 0.8 * 1e-6,
                              'threshold': -50.0,
                              'power': 1.0,
                              'saturation': 0.03 * 1e-3,
                              'reverse': -100.0,
                              'circuit': 'local'})
                G.add_edge(pre_id, synapse_id, delay=0.001)
                G.add_edge(synapse_id, post_id)

    return G


def create_lpu(file_name, lpu_name, N_sensory, N_local, N_proj):
    """
    Create a generic LPU graph.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The GEXF
    file also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model.

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    lpu_name : str
        Name of LPU. Used in port identifiers.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.

    Returns
    -------
    g : networkx.MultiDiGraph
        Generated graph.
    """

    g = create_lpu_graph(lpu_name, N_sensory, N_local, N_proj)
    nx.write_gexf(g, file_name)


def create_input(file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    g: networkx.MultiDiGraph
        NetworkX graph object representing the LPU
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """

    Nt = int(dur / dt)
    t = np.arange(0, dt * Nt, dt)

    uids = ["sensory_" + str(i) for i in range(N_sensory)]

    uids = np.array(uids)

    I = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[np.logical_and(t > start, t < stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('I/uids', data=uids)
        f.create_dataset('I/data', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='generic_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('in_file_name', nargs='?', default='generic_input.h5',
                        help='Input file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    parser.add_argument('-l', '--lpu', type=str, default='gen',
                        help='LPU name')

    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    dt = 1e-4
    dur = 1.0
    start = 0.3
    stop = 0.6
    I_max = 0.6
    neu_num = [np.random.randint(31, 40) for i in xrange(3)]

    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
    g = nx.read_gexf(args.lpu_file_name)
    create_input(args.in_file_name, neu_num[0], dt, dur, start, stop, I_max)
    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
