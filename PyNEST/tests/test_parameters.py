"""Unit tests for the Pydantic parameter workflow."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from microcircuit.parameter_definitions import Parameters


def test_stimulus_defaults():
    params = Parameters()

    assert params.thalamic_input is False
    assert params.th_start == 700.0
    assert params.th_duration == 10.0
    assert params.th_rate == 120.0
    assert params.num_th_neurons == 902
    assert params.conn_probs_th == [
        0.0,
        0.0,
        0.0983,
        0.0619,
        0.0,
        0.0,
        0.0512,
        0.0196,
    ]
    assert params.dc_transient is False
    assert params.dc_transient_start == 650.0
    assert params.dc_transient_dur == 100.0
    assert params.dc_transient_amp == 0.3


def test_valid_stimulus_assignment():
    params = Parameters()

    params.thalamic_input = True
    params.th_start = 0.0
    params.th_duration = 20.0
    params.th_rate = 0.0
    params.num_th_neurons = 100
    params.dc_transient = True
    params.dc_transient_start = 0.0
    params.dc_transient_dur = 50.0

    assert params.thalamic_input is True
    assert params.th_start == 0.0
    assert params.th_duration == 20.0
    assert params.th_rate == 0.0
    assert params.num_th_neurons == 100
    assert params.dc_transient is True
    assert params.dc_transient_start == 0.0
    assert params.dc_transient_dur == 50.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("th_start", -1.0),
        ("th_duration", 0.0),
        ("th_duration", -1.0),
        ("th_rate", -1.0),
        ("num_th_neurons", 0),
        ("num_th_neurons", -1),
        ("dc_transient_start", -1.0),
        ("dc_transient_dur", 0.0),
        ("dc_transient_dur", -1.0),
    ],
)
def test_invalid_stimulus_assignment_is_rejected(field_name, invalid_value):
    params = Parameters()

    with pytest.raises(ValidationError):
        setattr(params, field_name, invalid_value)


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_invalid_thalamic_probability_is_rejected(invalid_probability):
    params = Parameters()
    probabilities = params.conn_probs_th.copy()
    probabilities[0] = invalid_probability

    with pytest.raises(ValidationError):
        params.conn_probs_th = probabilities


def test_probability_boundaries_are_valid():
    params = Parameters()
    probabilities = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    params.conn_probs_th = probabilities

    assert params.conn_probs_th == probabilities


@pytest.mark.parametrize("length_offset", [-1, 1])
def test_invalid_thalamic_probability_length_is_rejected(length_offset):
    params = Parameters()
    probabilities = [0.0] * (len(params.populations) + length_offset)

    with pytest.raises(ValidationError):
        params.conn_probs_th = probabilities


def test_stimulus_parameters_are_serializable():
    params = Parameters(thalamic_input=True, dc_transient=True)

    data = params.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )

    assert data["thalamic_input"] is True
    assert data["conn_probs_th"] == params.conn_probs_th
    assert data["dc_transient"] is True
    assert "populations" not in data
    assert "PSP_th" not in data
    assert "delay_th_mean" not in data
    assert "delay_th_rel_std" not in data


def test_populations_are_not_editable_parameters():
    assert "populations" not in Parameters.model_fields
    assert "populations" not in Parameters.model_json_schema()["properties"]


def test_simulation_defaults():
    params = Parameters()

    assert params.t_presim == 500.0
    assert params.t_sim == 1000.0
    assert params.sim_resolution == 0.1
    assert params.rec_dev == ["spike_recorder"]
    assert params.data_path == Path("data")
    assert params.rng_seed == 55
    assert params.local_num_threads == 4
    assert params.rec_V_int == 1.0
    assert params.overwrite_files is True
    assert params.print_time is True
    assert params.store_metadata is True


def test_valid_simulation_assignment():
    params = Parameters()

    params.t_presim = 0.0
    params.t_sim = 0.0
    params.sim_resolution = 0.5
    params.rec_dev = ["spike_recorder", "voltmeter"]
    params.data_path = "results"
    params.rng_seed = 1
    params.local_num_threads = 1
    params.rec_V_int = 0.5
    params.overwrite_files = False
    params.print_time = False
    params.store_metadata = False

    assert params.t_presim == 0.0
    assert params.t_sim == 0.0
    assert params.sim_resolution == 0.5
    assert params.rec_dev == ["spike_recorder", "voltmeter"]
    assert params.data_path == Path("results")
    assert params.rng_seed == 1
    assert params.local_num_threads == 1
    assert params.rec_V_int == 0.5
    assert params.overwrite_files is False
    assert params.print_time is False
    assert params.store_metadata is False


def test_empty_rec_dev_is_valid():
    params = Parameters()

    params.rec_dev = []

    assert params.rec_dev == []


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("t_presim", -1.0),
        ("t_sim", -1.0),
        ("sim_resolution", 0.0),
        ("sim_resolution", -1.0),
        ("rng_seed", 0),
        ("rng_seed", -1),
        ("local_num_threads", 0),
        ("local_num_threads", -1),
        ("rec_V_int", 0.0),
        ("rec_V_int", -1.0),
    ],
)
def test_invalid_simulation_assignment_is_rejected(field_name, invalid_value):
    params = Parameters()

    with pytest.raises(ValidationError):
        setattr(params, field_name, invalid_value)


def test_invalid_rec_dev_entry_is_rejected():
    params = Parameters()

    with pytest.raises(ValidationError):
        params.rec_dev = ["not_a_device"]


def test_unknown_simulation_field_is_rejected():
    with pytest.raises(ValidationError):
        Parameters(not_a_real_simulation_field=1)


def test_simulation_parameters_are_serializable():
    params = Parameters(data_path="results", rec_dev=["spike_recorder", "voltmeter"])

    data = params.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )

    assert data["t_presim"] == params.t_presim
    assert data["t_sim"] == params.t_sim
    assert data["sim_resolution"] == params.sim_resolution
    assert data["rec_dev"] == ["spike_recorder", "voltmeter"]
    assert data["data_path"] == "results"
    assert data["rng_seed"] == params.rng_seed
    assert data["local_num_threads"] == params.local_num_threads
    assert data["rec_V_int"] == params.rec_V_int
    assert data["overwrite_files"] is True
    assert data["print_time"] is True
    assert data["store_metadata"] is True
