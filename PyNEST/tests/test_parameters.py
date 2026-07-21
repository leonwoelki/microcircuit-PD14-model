"""Unit tests for the Pydantic parameter workflow."""

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
