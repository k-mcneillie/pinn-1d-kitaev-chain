"""Tests for kitaev.training.utils.utils."""

from __future__ import annotations

import torch

from kitaev.training.utils.utils import EarlyStopping, EpochAccumulator, TrainingHistory


class TestEpochAccumulator:
    def test_averages_returns_empty_dict_when_never_updated(self) -> None:
        accumulator = EpochAccumulator()
        assert accumulator.averages() == {}

    def test_averages_single_update(self) -> None:
        accumulator = EpochAccumulator()
        accumulator.update({"loss": 2.0, "fsm": 4.0})
        assert accumulator.averages() == {"loss": 2.0, "fsm": 4.0}

    def test_averages_multiple_updates_computes_mean(self) -> None:
        accumulator = EpochAccumulator()
        accumulator.update({"loss": 1.0})
        accumulator.update({"loss": 3.0})
        assert accumulator.averages() == {"loss": 2.0}


class TestTrainingHistory:
    def test_record_creates_series_and_appends(self) -> None:
        history = TrainingHistory()
        history.record("train_loss", 1.0)
        history.record("train_loss", 2.0)
        assert history["train_loss"] == [1.0, 2.0]

    def test_contains(self) -> None:
        history = TrainingHistory()
        assert "train_loss" not in history
        history.record("train_loss", 1.0)
        assert "train_loss" in history

    def test_as_dict_returns_independent_dict_object(self) -> None:
        history = TrainingHistory()
        history.record("train_loss", 1.0)

        as_dict = history.as_dict()

        assert as_dict == {"train_loss": [1.0]}
        assert as_dict is not history._series

    def test_get_returns_series_or_default(self) -> None:
        history = TrainingHistory()
        history.record("train_loss", 1.0)

        assert history.get("train_loss") == [1.0]
        assert history.get("val_loss") == []
        assert history.get("val_loss", [9.0]) == [9.0]


class TestEarlyStopping:
    def test_first_call_is_always_recorded_as_best(self) -> None:
        early_stopping = EarlyStopping(patience=None)
        model = torch.nn.Linear(1, 1)

        should_stop = early_stopping.step(5.0, epoch=1, unwrapped_model=model)

        assert should_stop is False
        assert early_stopping.best_loss == 5.0
        assert early_stopping.best_epoch == 1
        assert early_stopping.best_state is not None

    def test_patience_none_never_signals_stop(self) -> None:
        early_stopping = EarlyStopping(patience=None)
        model = torch.nn.Linear(1, 1)

        early_stopping.step(1.0, epoch=1, unwrapped_model=model)
        should_stop = early_stopping.step(2.0, epoch=2, unwrapped_model=model)

        assert should_stop is False

    def test_improvement_resets_the_counter(self) -> None:
        early_stopping = EarlyStopping(patience=2)
        model = torch.nn.Linear(1, 1)

        assert early_stopping.step(5.0, epoch=1, unwrapped_model=model) is False
        assert early_stopping.step(10.0, epoch=2, unwrapped_model=model) is False
        assert early_stopping.step(3.0, epoch=3, unwrapped_model=model) is False

        assert early_stopping.best_loss == 3.0
        assert early_stopping.best_epoch == 3

    def test_non_improvement_increments_until_patience_exhausted(self) -> None:
        early_stopping = EarlyStopping(patience=2)
        model = torch.nn.Linear(1, 1)

        assert early_stopping.step(5.0, epoch=1, unwrapped_model=model) is False
        assert early_stopping.step(6.0, epoch=2, unwrapped_model=model) is False
        assert early_stopping.step(7.0, epoch=3, unwrapped_model=model) is True

    def test_best_state_is_a_real_deep_copy(self) -> None:
        early_stopping = EarlyStopping(patience=None)
        model = torch.nn.Linear(1, 1)

        early_stopping.step(1.0, epoch=1, unwrapped_model=model)
        original_weight = early_stopping.best_state["weight"].clone()

        with torch.no_grad():
            model.weight.fill_(999.0)

        assert torch.equal(early_stopping.best_state["weight"], original_weight)
        assert not torch.equal(early_stopping.best_state["weight"], model.weight)

    def test_track_state_false_records_best_without_snapshotting(self) -> None:
        early_stopping = EarlyStopping(patience=2, track_state=False)
        model = torch.nn.Linear(1, 1)

        early_stopping.step(5.0, epoch=1, unwrapped_model=model)
        early_stopping.step(2.0, epoch=2, unwrapped_model=model)

        assert early_stopping.best_loss == 2.0
        assert early_stopping.best_epoch == 2
        assert early_stopping.best_state is None

    def test_track_state_false_still_signals_stop_on_patience(self) -> None:
        early_stopping = EarlyStopping(patience=1, track_state=False)
        model = torch.nn.Linear(1, 1)

        assert early_stopping.step(1.0, epoch=1, unwrapped_model=model) is False
        assert early_stopping.step(2.0, epoch=2, unwrapped_model=model) is True
