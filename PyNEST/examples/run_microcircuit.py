# -*- coding: utf-8 -*-
#
# run_microcircuit.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Microcircuit example (Potjans & Diesmann 2014)
----------------------------------------------

Example illustrating usage of the `microcircuit` python package.
"""

#####################
import time
import nest
import numpy as np

## from microcircuit import network
from microcircuit.model import Model

## import parameter definitions
from microcircuit.parameter_definitions import Parameters

#####################

P = Parameters()

## set network scale
scaling_factor = 0.2
P.N_scaling = scaling_factor
P.K_scaling = scaling_factor

## set path for storing spike data and figures
P.data_path = "data_scale_%.2f/" % scaling_factor


def main():
    ## start timer
    time_start = time.time()

    ## create instance of the model
    model = Model(P)

    time_network = time.time()

    ## create all nodes (neurons, devices)
    model.create()
    time_create = time.time()

    ## connect nework
    model.connect()
    time_connect = time.time()

    ## pre-simulation (warm-up phase)
    model.simulate(P.t_presim)
    time_presimulate = time.time()

    ## simulation
    model.simulate(P.t_sim)
    time_simulate = time.time()

    ## store metadata
    model.store_metadata()

    ## current memory consumption of the python process (in MB)
    import psutil

    mem = psutil.Process().memory_info().rss / (1024 * 1024)

    #####################
    ## plot spikes and firing rate distribution
    print()
    print("##########################################")
    print()
    observation_interval = np.array([P.t_presim, P.t_presim + P.t_sim])
    model.evaluate(observation_interval, observation_interval)
    print()
    print("Raster plot                  : see %s " % (P.data_path / "raster_plot.png"))
    print("Distributions of firing rates: see %s " % (P.data_path / "box_plot.png"))
    time_evaluate = time.time()

    #####################
    ## print timers and memory consumption

    print()
    print("##########################################")
    print()
    print("Times of Rank %d:" % nest.Rank())
    print("    Total time:")
    print("    Time to initialize  : %.3fs" % (time_network - time_start))
    print("    Time to create      : %.3fs" % (time_create - time_network))
    print("    Time to connect     : %.3fs" % (time_connect - time_create))
    print("    Time to presimulate : %.3fs" % (time_presimulate - time_connect))
    print("    Time to simulate    : %.3fs" % (time_simulate - time_presimulate))
    print("    Time to evaluate    : %.3fs" % (time_evaluate - time_simulate))
    print()
    print("Memory consumption: %dMB" % mem)
    print()
    print("##########################################")
    print()


#####################

if __name__ == "__main__":
    main()
