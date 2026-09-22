# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``gym.rollout.completed_total``: the driver-side attempt counter, against a real in-memory reader."""

import pytest

from nemo_gym import rollout_collection
from nemo_gym.telemetry import gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

COMPLETED = gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT
OUTCOME = gym_metrics.ROLLOUT_OUTCOME_ATTRIBUTE
CLASS = gym_metrics.FAILURE_CLASS_ATTRIBUTE
KIND = gym_metrics.FAILURE_KIND_ATTRIBUTE
TYPE = gym_metrics.FAILURE_TYPE_ATTRIBUTE


def _by_attrs(collected):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(COMPLETED, [])}


def _key(**attrs):
    return tuple(sorted(attrs.items()))


def test_outcomes_are_counted_with_bounded_failure_attributes(collected_metrics):  # noqa: F811
    gym_metrics.record_rollout_completed("scored")
    gym_metrics.record_rollout_completed("scored")
    gym_metrics.record_rollout_completed(
        "dropped", failure_class="agent_run_error", failure_kind="agent_run_error", failure_type="ClientResponseError"
    )
    gym_metrics.record_rollout_completed(
        "dropped", failure_class="agent_run_error", failure_type="ClientResponseError"
    )
    gym_metrics.record_rollout_completed("omitted", failure_class="kill_shaped")
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored"}): 2,
        _key(
            **{CLASS: "agent_run_error", KIND: "agent_run_error", OUTCOME: "dropped", TYPE: "ClientResponseError"}
        ): 1,
        _key(**{CLASS: "agent_run_error", OUTCOME: "dropped", TYPE: "ClientResponseError"}): 1,
        _key(**{CLASS: "kill_shaped", OUTCOME: "omitted"}): 1,
    }


def test_recording_without_telemetry_is_a_no_op(monkeypatch):
    from nemo_gym.telemetry import setup as telemetry_setup

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    gym_metrics.record_rollout_completed("scored")


def test_driver_helper_reads_the_record_and_respects_the_span_group(collected_metrics, monkeypatch):  # noqa: F811
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: True)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored")
    rollout_collection._record_rollout_outcome(
        {
            rollout_collection.NG_FAILURE_CLASS_KEY: "agent_request_failed",
            "failure_kind": "transport_timeout",
            "_ng_failure_type": "ServerTimeoutError",
            "_ng_failure_message": "free text that must not become an attribute",
        },
        "dropped",
    )
    rollout_collection._record_rollout_outcome({rollout_collection.NG_NO_PERSIST_KEY: True}, "omitted")
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: False)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored")
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored"}): 1,
        _key(
            **{
                CLASS: "agent_request_failed",
                KIND: "transport_timeout",
                OUTCOME: "dropped",
                TYPE: "ServerTimeoutError",
            }
        ): 1,
        _key(**{OUTCOME: "omitted"}): 1,
    }
