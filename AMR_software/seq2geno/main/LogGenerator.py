# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os
from LoadFile import LoadFile


def make_logger(log_f=''):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_sh = logging.StreamHandler()
    # set the level of messages to report
    log_sh.setLevel(logging.DEBUG)
    # format the messages
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')
    log_sh.setFormatter(formatter)
    logger.addHandler(log_sh)

    # for the file
    if log_f != '':
        if not os.path.isfile(log_f):
            log_fh = open(log_f, 'w')
            log_fh.close()
        log_fh = logging.FileHandler(filename=log_f)
        log_fh.setLevel(logging.INFO)
        log_fh.setFormatter(formatter)
        logger.addHandler(log_fh)
    return(logger)
