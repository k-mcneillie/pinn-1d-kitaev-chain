"""Tests for the Adam-then-L-BFGS two-phase runner."""

from __future__ import annotations

import pytest
import torch

from kitaev.models import SirenPINNChiral
from kitaev.training.callbacks import TrainingCallback
from kitaev.training.config import TrainerConfig, TwoPhaseConfig
from kitaev.training.loss import ChiralFSMLoss
from kitaev.training.sampling import SamplingConfig, build_sampling
from kitaev.training.trainer import (
    _build_kitaev_operators,
    _concat_histories,
    run_two_phase,
)
from kitaev.training.utils import TrainingHistory

N_SITES = 6


@pytest.fixture
def operators():
    return _build_kitaev_operators(N_SITES, 1.0, 0.5)


@pytest.fixture
def streaming_loader():
    from kitaev.data.sampling_region import SamplingRegion

    regions = (SamplingRegion(low=0.05, high=4.0, weight=1.0),)
    loader, _ = build_sampling(
        SamplingConfig(mode="infinite", batch_size=64, steps_per_epoch=2), regions
    )
    return loader


def _model():
    return SirenPINNChiral(n_sites=N_SITES, hidden_features=8, hidden_layers=1)


def test_concat_histories_appends_series_end_to_end() -> None:
    first = TrainingHistory()
    second = TrainingHistory()
    for value in (1.0, 2.0):
        first.record("train_loss", value)
    for value in (3.0, 4.0, 5.0):
        second.record("train_loss", value)

    merged = _concat_histories(first, second)
    assert merged["train_loss"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_run_two_phase_records_both_phases(
    accelerator, make_session, operators, streaming_loader
) -> None:
    H_base, H_mu_diag, Xi = operators
    model = _model()

    trained, history = run_two_phase(
        session=make_session(),
        accelerator=accelerator,
        model=model,
        loss_fn=ChiralFSMLoss(n_sites=N_SITES),
        train_loader=streaming_loader,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=TwoPhaseConfig(
            adam_epochs=4,
            lbfgs_epochs=3,
            lbfgs_max_iter=4,
            lbfgs_history_size=8,
        ),
        base_config=TrainerConfig(epochs=1, print_freq=100, patience=None),
    )

    assert isinstance(trained, torch.nn.Module)
    assert len(history["train_loss"]) == 7  # 4 AdamW + 3 L-BFGS


def test_run_two_phase_can_skip_lbfgs(
    accelerator, make_session, operators, streaming_loader
) -> None:
    H_base, H_mu_diag, Xi = operators

    _, history = run_two_phase(
        session=make_session(),
        accelerator=accelerator,
        model=_model(),
        loss_fn=ChiralFSMLoss(n_sites=N_SITES),
        train_loader=streaming_loader,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=TwoPhaseConfig(adam_epochs=3, lbfgs_epochs=0),
    )

    assert len(history["train_loss"]) == 3


def test_run_two_phase_uses_dedicated_lbfgs_loader_without_sampling_callbacks(
    accelerator, make_session, operators
) -> None:
    """A frozen L-BFGS loader is honoured and its streaming callback dropped."""
    from kitaev.data.sampling_region import SamplingRegion

    regions = (SamplingRegion(low=0.05, high=4.0, weight=1.0),)
    adam_loader, adam_cbs = build_sampling(
        SamplingConfig(mode="infinite", batch_size=64, steps_per_epoch=2), regions
    )
    lbfgs_loader, _ = build_sampling(
        SamplingConfig(mode="frozen", batch_size=128, total_samples=128), regions
    )
    assert len(adam_cbs) == 1  # the streaming resampling callback

    spy = _SpyEpochCounter()
    H_base, H_mu_diag, Xi = operators

    _, history = run_two_phase(
        session=make_session(),
        accelerator=accelerator,
        model=_model(),
        loss_fn=ChiralFSMLoss(n_sites=N_SITES),
        train_loader=adam_loader,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=TwoPhaseConfig(
            adam_epochs=3, lbfgs_epochs=2, lbfgs_max_iter=4, lbfgs_history_size=8
        ),
        base_config=TrainerConfig(epochs=1, print_freq=100, patience=None),
        callbacks=[spy, *adam_cbs],
        lbfgs_train_loader=lbfgs_loader,
    )

    assert len(history["train_loss"]) == 5  # 3 AdamW + 2 L-BFGS
    # The spy callback runs only during the AdamW phase.
    assert spy.epochs == [1, 2, 3]


def test_run_two_phase_keeps_explicit_lbfgs_callbacks(
    accelerator, make_session, operators, streaming_loader
) -> None:
    """lbfgs_callbacks overrides the drop, so a probe spans both phases."""
    from kitaev.data.sampling_region import SamplingRegion

    lbfgs_loader, _ = build_sampling(
        SamplingConfig(mode="frozen", batch_size=128, total_samples=128),
        (SamplingRegion(low=0.05, high=4.0, weight=1.0),),
    )
    adam_spy = _SpyEpochCounter()
    both_spy = _SpyEpochCounter()
    H_base, H_mu_diag, Xi = operators

    run_two_phase(
        session=make_session(),
        accelerator=accelerator,
        model=_model(),
        loss_fn=ChiralFSMLoss(n_sites=N_SITES),
        train_loader=streaming_loader,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=TwoPhaseConfig(
            adam_epochs=3, lbfgs_epochs=2, lbfgs_max_iter=4, lbfgs_history_size=8
        ),
        base_config=TrainerConfig(epochs=1, print_freq=100, patience=None),
        callbacks=[adam_spy, both_spy],
        lbfgs_train_loader=lbfgs_loader,
        lbfgs_callbacks=[both_spy],
    )

    assert adam_spy.epochs == [1, 2, 3]  # dropped for the L-BFGS phase
    assert both_spy.epochs == [1, 2, 3, 1, 2]  # kept for both


class _SpyEpochCounter(TrainingCallback):
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def on_epoch_start(self, epoch, model, history) -> None:
        del model, history
        self.epochs.append(epoch)

    def on_epoch_end(self, epoch, model, history) -> None:
        del epoch, model, history


def test_run_two_phase_actually_updates_parameters(
    accelerator, make_session, operators, streaming_loader
) -> None:
    H_base, H_mu_diag, Xi = operators
    model = _model()
    before = [p.detach().clone() for p in model.parameters()]

    trained, _ = run_two_phase(
        session=make_session(),
        accelerator=accelerator,
        model=model,
        loss_fn=ChiralFSMLoss(n_sites=N_SITES),
        train_loader=streaming_loader,
        H_base=H_base,
        H_mu_diag=H_mu_diag,
        Xi=Xi,
        two_phase=TwoPhaseConfig(
            adam_epochs=3, lbfgs_epochs=2, lbfgs_max_iter=4, lbfgs_history_size=8
        ),
    )

    after = [p.detach().cpu() for p in trained.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


if __name__ == "__main__":
    pytest.main()
