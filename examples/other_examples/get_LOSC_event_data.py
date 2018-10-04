#!/usr/bin/env python
""" Helper script to faciliate downloading data from LOSC

Usage: To download the GW150914 data from https://losc.ligo.org/events/

$ python get_LOSC_event_data -e GW150914

"""

from __future__ import division
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Script to download LOSC data.')
parser.add_argument('-e', '--event', metavar='event', type=str)
parser.add_argument('-o', '--outdir', metavar='outdir',
                    default='tutorial_data')

args = parser.parse_args()

url_dictionary = dict(
    GW150914="https://losc.ligo.org/s/events/GW150914/{}-{}1_LOSC_4_V2-1126259446-32.txt.gz",
    LVT151012="https://losc.ligo.org/s/events/LVT151012/{}-{}1_LOSC_4_V2-1128678884-32.txt.gz",
    GW151226="https://losc.ligo.org/s/events/GW151226/{}-{}1_LOSC_4_V2-1135136334-32.txt.gz",
    GW170104="https://losc.ligo.org/s/events/GW170104/{}-{}1_LOSC_4_V1-1167559920-32.txt.gz",
    GW170608="https://losc.ligo.org/s/events/GW170608/{}-{}1_LOSC_CLN_4_V1-1180922478-32.txt.gz",
    GW170814="https://dcc.ligo.org/public/0146/P1700341/001/{}-{}1_LOSC_CLN_4_V1-1186741845-32.txt.gz",
    GW170817="https://dcc.ligo.org/public/0146/P1700349/001/{}-{}1_LOSC_CLN_4_V1-1187007040-2048.txt.gz")

outdir = 'tutorial_data'

data = []
for det, in ['H', 'L']:
    url = url_dictionary[args.event].format(det, det)
    filename = os.path.basename(url)
    if os.path.isfile(filename.rstrip('.gz')) is False:
        print("Downloading data from {}".format(url))
        os.system("wget {} ".format(url))
        os.system("gunzip {}".format(filename))
        filename = filename.rstrip('.gz')
    data.append(np.loadtxt(filename))
    with open(filename, 'r') as f:
        header = f.readlines()[:3]
        event = header[0].split(' ')[5]
        detector = header[0].split(' ')[7]
        sampling_frequency = header[1].split(' ')[4]
        starttime = header[2].split(' ')[3]
        duration = header[2].split(' ')[5]
        print('Loaded data for event={}, detector={}, sampling_frequency={}'
              ', starttime={}, duration={}'.format(
                  event, detector, sampling_frequency, starttime, duration))
    os.remove(filename)

time = np.arange(0, int(duration), 1 / int(sampling_frequency)) + int(starttime)
arr = [time] + data

outfile = '{}/{}_strain_data.npy'.format(args.outdir, args.event)
np.save(outfile, arr)
print("Saved data to {}".format(outfile))
