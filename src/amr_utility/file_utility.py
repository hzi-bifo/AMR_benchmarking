#!/usr/bin/python
import sys
import os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import math

def get_full_d(wd_results):

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    wd_results = os.path.join(fileDir, wd_results)
    return wd_results
def get_directory(path):
    os.path.dirname(path)
    return path


def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

def get_absolute_pathname(p_names):
    # get full path from a relative path.
    fileDir=os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, p_names)
    filename = os.path.abspath(os.path.realpath(filename))
    return filename


def roundup(x):
    return int(math.ceil(x / 50.0)) * 50





