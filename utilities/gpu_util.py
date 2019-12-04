#! /usr/bin/env python3

import subprocess

def get_gpu_memory_used():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    except:
        return -1
    else:
        return int(result)

def get_gpu_memory_total():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
    except:
        return -1
    else:
        return int(result)

def get_gpu_memory_free():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
    except:
        return -1
    else:
        return int(result)
    return int(result)    