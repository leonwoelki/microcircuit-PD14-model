# -*- coding: utf-8 -*-
#
# test_parameter_legacy_parity.py
#
# This file is part of https://github.com/INM-6/microcircuit-PD14-model
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Numerical parity checks between the `Parameters` computed fields and the
legacy dict-based derivation logic (formerly `Network.__derive_parameters()`
in `network.py`, before that module was renamed to `model.py`).

`_legacy_derive_parameters()` below is a frozen re-implementation of that
method against the still-present legacy dicts (`network_params.py`,
`stimulus_params.py`); it calls the same `helpers` functions the current
`Parameters` computed fields call, so any numeric divergence points at an
actual bug in the new derivation rather than an unrelated helper change.
"""

import copy

import numpy as np
import pytest

from microcircuit import helpers
from microcircuit.network_params import default_net_dict
from microcircuit.parameter_definitions import Parameters
from microcircuit.stimulus_params import default_stim_dict


def _legacy_derive_parameters(net_dict, stim_dict):
    net_dict = copy.deepcopy(net_dict)
    stim_dict = copy.deepcopy(stim_dict)

    num_pops = len(net_dict["populations"])

    full_num_synapses = helpers.num_synapses_from_conn_probs(
        net_dict["conn_probs"], net_dict["full_num_neurons"], net_dict["full_num_neurons"]
    )

    num_neurons = np.round(net_dict["full_num_neurons"] * net_dict["N_scaling"]).astype(int)
    num_synapses = np.round(
        full_num_synapses * net_dict["N_scaling"] * net_dict["K_scaling"]
    ).astype(int)
    ext_indegrees = np.round(net_dict["K_ext"] * net_dict["K_scaling"]).astype(int)

    PSC_over_PSP = helpers.postsynaptic_potential_to_current(
        net_dict["neuron_params"]["C_m"],
        net_dict["neuron_params"]["tau_m"],
        net_dict["neuron_params"]["tau_syn"],
    )
    PSC_matrix_mean = net_dict["PSP_matrix_mean"] * PSC_over_PSP
    PSC_ext = net_dict["PSP_exc_mean"] * PSC_over_PSP

    if net_dict["bg_input_type"] == "poisson":
        DC_amp = np.zeros(num_pops)
    else:
        DC_amp = helpers.dc_input_compensating_poisson(
            net_dict["bg_rate"],
            net_dict["K_ext"],
            net_dict["neuron_params"]["tau_syn"],
            PSC_ext,
        )

    if net_dict["K_scaling"] != 1:
        PSC_matrix_mean, PSC_ext, DC_amp = helpers.adjust_weights_and_input_to_synapse_scaling(
            net_dict["full_num_neurons"],
            full_num_synapses,
            net_dict["K_scaling"],
            PSC_matrix_mean,
            PSC_ext,
            net_dict["neuron_params"]["tau_syn"],
            net_dict["full_mean_rates"],
            DC_amp,
            net_dict["bg_input_type"],
            net_dict["bg_rate"],
            net_dict["K_ext"],
        )

    num_th_synapses = helpers.num_synapses_from_conn_probs(
        stim_dict["conn_probs_th"], stim_dict["num_th_neurons"], net_dict["full_num_neurons"]
    )[0]
    weight_th = stim_dict["PSP_th"] * PSC_over_PSP
    if net_dict["K_scaling"] != 1:
        num_th_synapses = num_th_synapses * net_dict["K_scaling"]
        weight_th = weight_th / np.sqrt(net_dict["K_scaling"])
    num_th_synapses = np.round(num_th_synapses).astype(int)

    # final per-population amplitude of transient DC stimulation, as computed
    # inline in the legacy `__create_dc_stim_input` (not stored in stim_dict)
    dc_amp_stim = stim_dict["dc_transient_amp"] * net_dict["K_ext"]

    return {
        "num_pops": num_pops,
        "full_num_synapses": full_num_synapses,
        "num_neurons": num_neurons,
        "num_synapses": num_synapses,
        "ext_indegrees": ext_indegrees,
        "PSC_matrix_mean": PSC_matrix_mean,
        "PSC_ext": PSC_ext,
        "DC_amp": DC_amp,
        "num_th_synapses": num_th_synapses,
        "weight_th": weight_th,
        "dc_amp_stim": dc_amp_stim,
        "PSP_matrix_mean": net_dict["PSP_matrix_mean"],
        "delay_matrix_mean": net_dict["delay_matrix_mean"],
    }


def _assert_matches_legacy(params, legacy):
    assert params.num_pops == legacy["num_pops"]
    assert params.full_num_synapses == legacy["full_num_synapses"].tolist()
    assert params.num_neurons == legacy["num_neurons"].tolist()
    assert params.num_synapses == legacy["num_synapses"].tolist()
    assert params.ext_indegrees == legacy["ext_indegrees"].tolist()
    assert np.array(params.PSC_matrix_mean) == pytest.approx(legacy["PSC_matrix_mean"])
    assert params.PSC_ext == pytest.approx(legacy["PSC_ext"])
    assert np.array(params.DC_amp) == pytest.approx(legacy["DC_amp"])
    assert params.num_th_synapses == legacy["num_th_synapses"].tolist()
    assert params.weight_th == pytest.approx(legacy["weight_th"])
    assert np.array(params.dc_transient_amp_populations) == pytest.approx(legacy["dc_amp_stim"])
    assert np.array(params.PSP_matrix_mean) == pytest.approx(legacy["PSP_matrix_mean"])
    assert np.array(params.delay_matrix_mean) == pytest.approx(legacy["delay_matrix_mean"])


def test_legacy_parity_at_default_scaling():
    legacy = _legacy_derive_parameters(default_net_dict, default_stim_dict)
    params = Parameters()

    _assert_matches_legacy(params, legacy)


def test_legacy_parity_at_scaled_N_and_K():
    net_dict = copy.deepcopy(default_net_dict)
    net_dict["N_scaling"] = 0.2
    net_dict["K_scaling"] = 0.2
    legacy = _legacy_derive_parameters(net_dict, default_stim_dict)

    params = Parameters()
    params.N_scaling = 0.2
    params.K_scaling = 0.2

    _assert_matches_legacy(params, legacy)


def test_legacy_parity_with_poisson_background_input():
    net_dict = copy.deepcopy(default_net_dict)
    net_dict["bg_input_type"] = "poisson"
    net_dict["K_scaling"] = 0.2
    legacy = _legacy_derive_parameters(net_dict, default_stim_dict)

    params = Parameters()
    params.CC_type = "poisson"
    params.K_scaling = 0.2

    _assert_matches_legacy(params, legacy)
