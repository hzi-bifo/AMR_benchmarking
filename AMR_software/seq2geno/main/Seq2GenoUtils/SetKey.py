#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2021 Gary Robertson and Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later
import os.path
import ssl
import json
from urllib.request import urlopen

def SetKey():
    # file to save our user API key in - one key per client install, normally
    assert 'SEQ2GENO_HOME' in os.environ, 'SEQ2GENO_HOME not available'
    fname = os.path.join(os.environ['SEQ2GENO_HOME'], 'keyfile')

    key = ""
    if os.path.isfile(fname):
        with open(fname, 'r') as file:
            key = file.read().replace('\n', '')
    else:
        exthost="https://galaxy.bifo.helmholtz-hzi.de/keygen"
        # This is an auth key which can be embedded in the client - just to
        # stop random HTTP requests generating new users
        data={"authKey": "ighuoHez2shaimee"}
        context = ssl._create_unverified_context()
        # server-side, I have an admin API key which is able to generate new users
        response = urlopen(exthost, json.dumps(data).encode(), context=context)
        # new API key for new user is returned
        key = response.read().decode()
        with open(fname, 'w') as file:
           file.write(key)

    # use key
    return(key)
