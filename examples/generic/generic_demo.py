#!/usr/bin/env python

"""
Generic LPU demo

Notes
-----
Generate input file and LPU configuration by running

cd data
python gen_generic_lpu.py
"""

import argparse

import neurokernel.core_gpu as core
from neurokernel.tools.logging import setup_logger

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.LPU import LPU
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

dt = 1e-4
dur = 1.0
steps = int(dur/dt)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=steps, type=int,
                    help='Number of steps [default: %s]' % steps)
parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                    help='GPU device number [default: 0]')
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)

man = core.Manager()

(comp_dict, conns) = LPU.lpu_parser('./data/generic_lpu.gexf.gz')

#st_input_processor = StepInputProcessor('I', ['73','74','75','76','77'] , 10, 0.1,0.4)
fl_input_processor = FileInputProcessor('./data/generic_input.h5')
fl_output_processor = FileOutputProcessor([('V',None),('spike_state',None)], 'new_output.h5', sample_interval=1)
'''
a = LPU(dt, comp_dict, conns, device=args.gpu_dev, input_processors = [st_input_processor], id='ge')
'''
man.add(LPU, 'ge', dt, comp_dict, conns,
        device=args.gpu_dev, input_processors = [fl_input_processor],
        output_processors = [fl_output_processor], debug=args.debug)

man.spawn()
man.start(steps=args.steps)
man.wait()
