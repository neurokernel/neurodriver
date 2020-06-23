#!/usr/bin/env python

import inspect

import numpy as np
import networkx as nx

from ..LPU import LPU
from .NDComponent import NDComponent
from ..InputProcessors.BaseInputProcessor import BaseInputProcessor
from ..InputProcessors.ArrayInputProcessor import ArrayInputProcessor
from ..OutputProcessors.OutputRecorder import OutputRecorder
from ..OutputProcessors.FileOutputProcessor import FileOutputProcessor

def test_NDComponent(comp, dt, **kwargs):
    """
    Test an NDComponent.

    Parameters
    ----------
    comp: class or str
          Specifies the NDComponent class to test.
          Either the class itself, or its name.
          If name of the class is used, the path to the module containing
          the class must be specified by the argument `path`.
    dt: float
        time step used for simulation.
    path: str
          Path to the module containing the NDComponent class to test.
    input: InputProcessor or dict
           If input is an instance of the subclass of BaseInputProcessor,
           return input, check there is one 1 uid (can be multiple instances)
           in the InputProcessor, and the length is None.

           If input is a dict,
           should be provided in the following example format:
           {var1: InputProcessor,
            var2: np.ndarray, ...},
           where var1, var2 are variable name to be injected to,
           the values can be InputProcessors or np.ndarray.
           If np.ndarray is provided as input, an ArrayInputProcessor
           will be created for it and uid taking from InputProcessor
           specified from other variables, or 'component' if no InputProcessor
           is used.

           If not specified, no model will run without inputs.
    dur: float
         duration of simulation.
         If not specified, infer from other arguments.
    steps: int
           number of time steps to simulate.
           If not specified, infer from other arguments.
    device: int
            The device ID of the GPU to run stimulation (default: 0).
    print_timing: bool
                  Whether or not to print out the timing of the test (default: False).
    Return
    ------
    output: OurputProcessor.OutputRecorder or str
            If filename not specified, returns an OurputProcessor.OutputRecorder
            object, in which the uid and data of outputs can be accessed by
            `output[var]`.
            If filename is provided, returns the filename of the stored outputs.

    Examples
    --------
    import numpy as np
    from neurokernel.LPU.NDComponents.testing import test_NDComponent
    from neurokernel.LPU.NDComponents.AxonHillockModels.HodgkinHuxley2 import HodgkinHuxley2

    dt = 1e-6
    output = test_NDComponent(HodgkinHuxley2, dt,
                              g_K = 36.0, g_Na = 120.0, g_L = 0.3,
                              E_Na = 50., E_K = -77., E_L = -54.387,
                              input = {'I': np.zeros((int(1e5),1))+10})
    print('spike times:', np.where(output.output['spike_state']['data'])[0]*dt)
    """
    if inspect.isclass(comp):
        if issubclass(comp, NDComponent):
            comp_name = comp.__name__
            extra_comp = comp
        else:
            raise ValueError('comp is a class but not a subclass of NDComponent.')
    elif isinstance(comp, str):
        path = kwargs.get('module_path', [])
        cls = [(name, obj) for name, obj in LPU.import_NDcomponent_from_path(path) is name == comp]
        if len(cls) == 0:
            raise ValueError('Cannot find any {} model from {}'.format(comp, path))
        else:
            comp_name, extra_comp = cls[0]

    accesses = extra_comp.accesses
    updates = extra_comp.updates
    params = extra_comp.params
    extra_params = extra_comp.extra_params
    internals = extra_comp.internals

    def process_input(input, extra_comp):
        """
        extract and create InputProcessors from input

        parameters
        ----------
        input: InputProcessor or dict
               If input is an instance of the subclass of BaseInputProcessor,
               return input, check there is one 1 uid (can be multiple instances)
               in the InputProcessor, and the length is None.

               If input is a dict,
               should be provided in the following example format:
               {var1: InputProcessor,
                var2: np.ndarray, ...},
               where var1, var2 are variable name to be injected to,
               the values can be InputProcessors or np.ndarray.
               If np.ndarray is provided as input, an ArrayInputProcessor
               will be created for it and uid taking from InputProcessor
               specified from other variables, or 'component' if no InputProcessor
               is used.

        extra_comp: class
                    an NDComponent class

        Return
        ------
        input_processors: list
                          a list of input_processors
        uid: str
             uid that can be used to specify the node id
        length: int or None
                If np.ndarray is provided in any of the variables,
                the number of steps will be returned as length.
                Otherwise None is returned.
        """
        if input is None:
            return [], 'component', None

        model_name = extra_comp.__name__
        accesses = set(extra_comp.accesses)
        if isinstance(input, BaseInputProcessor):
            uids = []
            for var in input.variables:
                if var not in accesses:
                    raise ValueError('InputProcessor contains variable {}, not in the accesses field of {} model with accesses = [{}]'.format(var, model_name, ','.join(list(accesses))))
                uids.append(input.variables[var]['uids'])
            uid = set(uids)
            if len(uid > 1):
                raise ValueError('More than 1 uids specified in the InputProcessor')
            uid = list(uid)[0] #just need to return the uid of the component
            return [input], uid, None
        elif isinstance(input, dict):
            length = []
            # check var
            var = set(input.keys())
            assert var.issubset(accesses), 'input contains variable(s) not in the accesses field of {} model with accesses = [{}]'.format(var, model_name, ','.join(list(accesses)))

            # check uid
            input_processors = [v for k, v in input.items() if isinstance(v, BaseInputProcessor)]
            if not len(input_processors):
                uid = 'component'
            else:
                uids = []
                for v in input_processors:
                    for var in v.variables:
                        uids.append(v.variables[var]['uids'])
                uid = set(uids)
                if len(uid > 1):
                    raise ValueError('More than 1 uids specified in the InputProcessor')
                uid = list(uid)[0]
            input_processors = []
            for k, v in input.items():
                if isinstance(v, BaseInputProcessor):
                    input_processors.append(v)
                elif isinstance(v, np.ndarray):
                    assert len(v.shape) == 2, 'Input must be a 2D'
                    input_processors.append(
                        ArrayInputProcessor({k: {'uids': [uid]*v.shape[1], 'data': v}}))
                    length.append(v.shape[0])
                elif isinstance(v, dict) and k == 'spike_state':
                    input_processors.append(
                        ArrayInputProcessor({k: {'uids': [uid]*(v['index'].max()+1), 'data': v}}))
                    length.append(v['time'][-1]+0.5)
            if len(length) == 0:
                length = None
            else:
                length = int(np.max(np.array(length)))
            return input_processors, uid, length

    input = kwargs.get('input', None)
    input_processors, uid, length = process_input(input, extra_comp)

    set_params = {k: v for k, v in kwargs.items() if k in params+extra_params}
    set_internals = {'init{}'.format(k): v for k, v in kwargs.items() if k in internals}

    args = set_params
    args.update(set_internals)
    G = nx.MultiDiGraph()
    G.add_node(uid, name = 'component',
               model = comp_name,
               **args
               )

    dur = kwargs.get('dur', None)
    steps = kwargs.get('steps', None)

    if steps is None:
        if dur is not None:
            steps = int(np.floor(dur/dt))
        else:
            if input is not None and length is not None:
                steps = length
                dur = steps*dt
    if dur is None:
        if steps is not None:
            dur = steps*dt
        else:
            if input is not None and length is not None:
                steps = length
                dur = steps*dt

    output_processor = OutputRecorder([(var, [uid]) for var in updates])

    print('Testing component {} for duration {} at dt = {} and a total number of {} steps...'.format(
                            comp_name, dur, dt, steps))

    lpu = LPU(dt, 'obj', {'graph': G, 'kwargs': {'class_key': 'model'}},
              device = kwargs.get('device', 0),
              id = 'comp_test', input_processors = input_processors,
              output_processors = [output_processor],
              debug = True,
              manager = False,
              print_timing = kwargs.get('print_timing', False),
              extra_comps = [extra_comp])

    lpu.run(steps = steps)

    return output_processor.get_output(uids = uid)[uid]
