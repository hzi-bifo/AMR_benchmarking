# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import sys


class RedirectText(object):
    # Class to extend the scrolledText
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.insert(tk.END, string)


def create_verbose_frame(win_root, outlevel, width=1000, height=100):
    # make a panel to display the verbose output (stdout)
    panel_verbose = ttk.Frame(win_root, width=width, height=height)
    v_box = scrolledtext.ScrolledText(win_root, width=width)
    v_box.pack()
    redir = RedirectText(v_box)
    # recognizable choices
    all_outlevels = ['stdout', 'stderr']
    outlevel = [line for line in all_outlevels if line in all_outlevels]
    if 'stdout' in outlevel:
        sys.stdout = redir
    if 'stderr' in outlevel:
        sys.stderr = redir
    return(v_box)
