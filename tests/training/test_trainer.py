"""Tests for kitaev.training.trainer."""

from __future__ import annotations

import math

import pytest
import torch

from kitaev.training.config import TrainerConfig
from kitaev.training.loss.losses import PinnedFSMLoss
from kitaev.training.trainer import (
    UnifiedTrainer,
    _build_kitaev_operators,
    _ExampleSirenStandIn,
)

_UNSET = object()


@pytest.fixture
def trainer_factory(accelerator, tiny_model_factory, make_session):
    """Factory building a fully wired, real `UnifiedTrainer`.

    Returns a `(trainer, session)` pair so tests can inspect both the
    trainer's state and the real log file written by its `Session`.
    """

    def _make(
        optimiser_fn=None,
        scheduler_fn=None,
        config=_UNSET,
        loss_fn=None,
        n_sites: int = 2,
        hidden_features: int = 4,
    ) -> tuple[UnifiedTrainer, object]:
        model = tiny_model_factory(n_sites=n_sites, hidden_features=hidden_features)

        if optimiser_fn is None:

            def optimiser_fn(params):
                return torch.optim.AdamW(params, lr=1e-2)

        optimiser = optimiser_fn(model.parameters())

        scheduler = scheduler_fn(optimiser) if scheduler_fn is not None else None

        if loss_fn is None:
            loss_fn = PinnedFSMLoss(total_epochs=3, anneal_duration=2)

        if config is _UNSET:
            config = TrainerConfig(
                epochs=3, print_freq=1000, patience=None, grad_clip_norm=1.0
            )

        session = make_session()

        trainer = UnifiedTrainer(
            session=session,
            accelerator=accelerator,
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            scheduler=scheduler,
            config=config,
        )
        return trainer, session

    return _make


@pytest.fixture
def mu_batch(accelerator) -> torch.Tensor:
    """A small, real mu batch already placed on the accelerator's device."""
    return (torch.rand(4, 1) * 6.0 - 3.0).to(accelerator.device)


def _read_log(session) -> str:
    return (session.output_dir / "run.log").read_text()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_without_scheduler_sets_scheduler_none(trainer_factory) -> None:
    trainer, _ = trainer_factory(scheduler_fn=None)
    assert trainer.scheduler is None
    assert trainer.model is not None
    assert trainer.optimiser is not None


def test_init_with_scheduler_prepares_three_things(trainer_factory) -> None:
    trainer, _ = trainer_factory(
        scheduler_fn=lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        )
    )
    assert trainer.scheduler is not None


def test_init_default_config_when_none(trainer_factory) -> None:
    trainer, _ = trainer_factory(config=None)
    assert isinstance(trainer.config, TrainerConfig)
    assert trainer.config == TrainerConfig()


def test_init_explicit_config_is_used(trainer_factory) -> None:
    trainer, _ = trainer_factory(config=TrainerConfig(epochs=7))
    assert trainer.config.epochs == 7


def test_init_is_lbfgs_true_for_lbfgs_optimiser(trainer_factory) -> None:
    trainer, _ = trainer_factory(optimiser_fn=lambda params: torch.optim.LBFGS(params))
    assert trainer._is_lbfgs is True


def test_init_is_lbfgs_false_for_non_lbfgs_optimiser(trainer_factory) -> None:
    trainer, _ = trainer_factory()
    assert trainer._is_lbfgs is False


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_without_val_loader_runs_full_epochs(
    trainer_factory, tiny_loader_factory, kitaev_operators
) -> None:
    trainer, session = trainer_factory()
    H_base, H_mu_diag, Xi = kitaev_operators
    train_loader = tiny_loader_factory(n_samples=4, batch_size=4)

    result = trainer.fit(train_loader, H_base, H_mu_diag, Xi, val_loader=None)

    assert isinstance(result, torch.nn.Module)
    assert len(trainer.history["train_loss"]) == 3
    assert "val_loss" not in trainer.history
    assert trainer._early_stopping.best_state is None

    log_text = _read_log(session)
    assert "Loaded best model state" not in log_text
    assert "Early stopping triggered" not in log_text


def test_fit_with_val_loader_checkpoints_best_state(
    trainer_factory, tiny_loader_factory, kitaev_operators
) -> None:
    trainer, session = trainer_factory()
    H_base, H_mu_diag, Xi = kitaev_operators
    train_loader = tiny_loader_factory(n_samples=4, batch_size=4)
    val_loader = tiny_loader_factory(n_samples=4, batch_size=4)

    trainer.fit(train_loader, H_base, H_mu_diag, Xi, val_loader=val_loader)

    assert len(trainer.history["val_loss"]) == 3
    assert trainer._early_stopping.best_state is not None

    log_text = _read_log(session)
    assert "Loaded best model state from validation" in log_text


def test_fit_with_scheduler_updates_learning_rate(
    trainer_factory, tiny_loader_factory, kitaev_operators
) -> None:
    trainer, _ = trainer_factory(
        scheduler_fn=lambda opt: torch.optim.lr_scheduler.StepLR(
            opt, step_size=1, gamma=0.5
        )
    )
    H_base, H_mu_diag, Xi = kitaev_operators
    train_loader = tiny_loader_factory(n_samples=4, batch_size=4)
    initial_lr = trainer.optimiser.param_groups[0]["lr"]

    trainer.fit(train_loader, H_base, H_mu_diag, Xi, val_loader=None)

    final_lr = trainer.optimiser.param_groups[0]["lr"]
    assert final_lr == pytest.approx(initial_lr * 0.5**3)


def test_fit_logs_on_period_and_on_final_epoch(
    trainer_factory, tiny_loader_factory, kitaev_operators
) -> None:
    trainer, session = trainer_factory(
        config=TrainerConfig(epochs=5, print_freq=2, patience=None, grad_clip_norm=1.0)
    )
    H_base, H_mu_diag, Xi = kitaev_operators
    train_loader = tiny_loader_factory(n_samples=4, batch_size=4)

    trainer.fit(train_loader, H_base, H_mu_diag, Xi, val_loader=None)

    log_text = _read_log(session)
    # print_freq=2 over 5 epochs hits the periodic branch at 2 and 4, and the
    # final-epoch fallback at 5 (which is not itself a multiple of 2) — three
    # logged epochs in total, covering both sides of the `or`.
    assert log_text.count("Epoch 000") == 3


def test_fit_early_stopping_triggers_and_breaks(
    trainer_factory, tiny_loader_factory, kitaev_operators
) -> None:
    # A zero-learning-rate SGD optimiser leaves the model's weights bit-for-bit
    # unchanged epoch to epoch, and anneal_duration=1 pins PinnedFSMLoss's
    # pin_weight at its floor from epoch 1 onward. With a single, fixed-content
    # validation batch, the validation loss is therefore identical every
    # epoch, so it improves only on epoch 1 (against the initial `inf`) and
    # never again — a fully deterministic, real (non-mocked) non-improvement
    # streak that trips patience=2 at exactly epoch 3.
    trainer, session = trainer_factory(
        optimiser_fn=lambda params: torch.optim.SGD(params, lr=0.0),
        loss_fn=PinnedFSMLoss(total_epochs=3, anneal_duration=1),
        config=TrainerConfig(epochs=5, print_freq=1000, patience=2, grad_clip_norm=1.0),
    )
    H_base, H_mu_diag, Xi = kitaev_operators
    train_loader = tiny_loader_factory(n_samples=4, batch_size=4)
    val_loader = tiny_loader_factory(n_samples=4, batch_size=4)

    trainer.fit(train_loader, H_base, H_mu_diag, Xi, val_loader=val_loader)

    assert len(trainer.history["train_loss"]) == 3

    log_text = _read_log(session)
    assert (
        "Early stopping triggered at epoch 3 (no improvement for 2 epochs)." in log_text
    )


# ---------------------------------------------------------------------------
# _run_epoch
# ---------------------------------------------------------------------------


def test_run_epoch_train_true_puts_model_in_train_mode(
    trainer_factory, tiny_loader_factory, kitaev_operators, accelerator
) -> None:
    trainer, _ = trainer_factory()
    H_base, H_mu_diag, Xi = kitaev_operators
    loader = accelerator.prepare(tiny_loader_factory(n_samples=4, batch_size=4))

    metrics = trainer._run_epoch(loader, H_base, H_mu_diag, Xi, train=True, epoch=1)

    assert math.isfinite(metrics["loss"])
    assert trainer.model.training is True


def test_run_epoch_train_false_uses_no_grad(
    trainer_factory, tiny_loader_factory, kitaev_operators, accelerator
) -> None:
    trainer, _ = trainer_factory()
    H_base, H_mu_diag, Xi = kitaev_operators
    loader = accelerator.prepare(tiny_loader_factory(n_samples=4, batch_size=4))

    metrics = trainer._run_epoch(loader, H_base, H_mu_diag, Xi, train=False, epoch=1)

    assert trainer.model.training is False
    assert math.isfinite(metrics["loss"])
    # No backward pass occurs on the eval path, so no parameter accumulates a
    # gradient from this call.
    assert all(p.grad is None for p in trainer.model.parameters())


# ---------------------------------------------------------------------------
# _train_step dispatch
# ---------------------------------------------------------------------------


def test_train_step_dispatches_to_standard_step_for_non_lbfgs(
    trainer_factory, kitaev_operators, mu_batch
) -> None:
    trainer, _ = trainer_factory()
    H_base, H_mu_diag, Xi = kitaev_operators

    loss_value, metrics = trainer._train_step(mu_batch, H_base, H_mu_diag, Xi, epoch=1)

    assert isinstance(loss_value, float)
    assert math.isfinite(loss_value)
    assert isinstance(metrics, dict)


def test_train_step_dispatches_to_lbfgs_step_for_lbfgs(
    trainer_factory, kitaev_operators, mu_batch
) -> None:
    trainer, _ = trainer_factory(optimiser_fn=lambda params: torch.optim.LBFGS(params))
    H_base, H_mu_diag, Xi = kitaev_operators

    loss_value, metrics = trainer._train_step(mu_batch, H_base, H_mu_diag, Xi, epoch=1)

    assert isinstance(loss_value, float)
    assert math.isfinite(loss_value)
    assert "pin_wt" in metrics


# ---------------------------------------------------------------------------
# _standard_step
# ---------------------------------------------------------------------------


def test_standard_step_with_grad_clip_norm_set(
    trainer_factory, kitaev_operators, mu_batch
) -> None:
    trainer, _ = trainer_factory(config=TrainerConfig(epochs=1, grad_clip_norm=1.0))
    H_base, H_mu_diag, Xi = kitaev_operators

    loss_value, _ = trainer._standard_step(mu_batch, H_base, H_mu_diag, Xi, epoch=1)

    assert math.isfinite(loss_value)
    grad_norms = [
        p.grad.detach().norm() for p in trainer.model.parameters() if p.grad is not None
    ]
    assert grad_norms
    total_norm = torch.norm(torch.stack(grad_norms))
    assert total_norm <= 1.0 + 1e-4


def test_standard_step_with_grad_clip_norm_none(
    trainer_factory, kitaev_operators, mu_batch
) -> None:
    trainer, _ = trainer_factory(config=TrainerConfig(epochs=1, grad_clip_norm=None))
    H_base, H_mu_diag, Xi = kitaev_operators

    loss_value, _ = trainer._standard_step(mu_batch, H_base, H_mu_diag, Xi, epoch=1)

    assert math.isfinite(loss_value)


# ---------------------------------------------------------------------------
# _lbfgs_step
# ---------------------------------------------------------------------------


def test_lbfgs_step_returns_captured_loss_not_none(
    trainer_factory, kitaev_operators, mu_batch
) -> None:
    # Regression guard: accelerate's AcceleratedOptimizer.step(closure)
    # discards the closure's return value, so the loss must come from the
    # closure's own captured state, not from optimiser.step()'s return value.
    trainer, _ = trainer_factory(optimiser_fn=lambda params: torch.optim.LBFGS(params))
    H_base, H_mu_diag, Xi = kitaev_operators

    loss_value, metrics = trainer._lbfgs_step(mu_batch, H_base, H_mu_diag, Xi, epoch=1)

    assert isinstance(loss_value, float)
    assert math.isfinite(loss_value)
    assert not math.isnan(loss_value)
    assert "pin_wt" in metrics


# ---------------------------------------------------------------------------
# _record
# ---------------------------------------------------------------------------


def test_record_prefixes_keys_into_history(trainer_factory) -> None:
    trainer, _ = trainer_factory()

    trainer._record("train", {"loss": 1.0, "fsm": 0.5})

    assert trainer.history["train_loss"] == [1.0]
    assert trainer.history["train_fsm"] == [0.5]


# ---------------------------------------------------------------------------
# _log_epoch
# ---------------------------------------------------------------------------


def test_log_epoch_without_val_metrics_and_without_pin_wt(trainer_factory) -> None:
    trainer, session = trainer_factory()

    trainer._log_epoch(5, {"loss": 1.0, "fsm": 0.2}, None)

    log_text = _read_log(session)
    assert "Epoch 0005" in log_text
    assert "loss: 1.000000" in log_text
    assert "pin_wt" not in log_text
    assert "||" not in log_text


def test_log_epoch_with_val_metrics_and_with_pin_wt(trainer_factory) -> None:
    trainer, session = trainer_factory()

    trainer._log_epoch(5, {"loss": 1.0, "pin_wt": 0.42}, {"loss": 0.9})

    log_text = _read_log(session)
    assert "||" in log_text
    assert "val_loss: 0.900000" in log_text
    assert "pin_wt: 0.420" in log_text


# ---------------------------------------------------------------------------
# _finalise_model
# ---------------------------------------------------------------------------


def test_finalise_model_restores_best_state_when_present(trainer_factory) -> None:
    trainer, session = trainer_factory()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    trainer._early_stopping.step(0.1, epoch=2, unwrapped_model=unwrapped)

    result = trainer._finalise_model()

    for key, value in result.state_dict().items():
        assert torch.equal(value, trainer._early_stopping.best_state[key])

    log_text = _read_log(session)
    assert "epoch 2," in log_text


def test_finalise_model_returns_final_state_when_never_validated(
    trainer_factory,
) -> None:
    trainer, session = trainer_factory()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    pre_call_state = {
        key: value.clone() for key, value in unwrapped.state_dict().items()
    }

    result = trainer._finalise_model()

    for key, value in result.state_dict().items():
        assert torch.equal(value, pre_call_state[key])

    log_text = _read_log(session)
    assert "Loaded best model state" not in log_text


# ---------------------------------------------------------------------------
# _build_kitaev_operators
# ---------------------------------------------------------------------------


def test_build_kitaev_operators_shapes() -> None:
    H_base, H_mu_diag, Xi = _build_kitaev_operators(4, hopping=1.0, pairing=0.5)
    assert H_base.shape == (8, 8)
    assert H_mu_diag.shape == (8, 8)
    assert Xi.shape == (8, 8)


def test_build_kitaev_operators_h_base_is_symmetric() -> None:
    H_base, _, _ = _build_kitaev_operators(4, hopping=1.0, pairing=0.5)
    assert torch.allclose(H_base, H_base.T)


def test_build_kitaev_operators_h_mu_diag_sign_pattern() -> None:
    n_sites = 4
    _, H_mu_diag, _ = _build_kitaev_operators(n_sites, hopping=1.0, pairing=0.5)

    diagonal = H_mu_diag.diagonal()
    assert torch.all(diagonal[:n_sites] == -1.0)
    assert torch.all(diagonal[n_sites:] == 1.0)
    off_diagonal = H_mu_diag - torch.diag(diagonal)
    assert torch.all(off_diagonal == 0.0)


def test_build_kitaev_operators_xi_block_structure() -> None:
    n_sites = 4
    _, _, Xi = _build_kitaev_operators(n_sites, hopping=1.0, pairing=0.5)

    assert torch.equal(Xi[:n_sites, n_sites:], torch.eye(n_sites))
    assert torch.equal(Xi[n_sites:, :n_sites], torch.eye(n_sites))
    assert torch.all(Xi[:n_sites, :n_sites] == 0.0)
    assert torch.all(Xi[n_sites:, n_sites:] == 0.0)
    assert torch.equal(Xi, Xi.T)


def test_build_kitaev_operators_explicit_entries() -> None:
    n_sites = 3
    H_base, _, _ = _build_kitaev_operators(n_sites, hopping=1.0, pairing=0.5)

    assert H_base[0, 1] == -1.0
    assert H_base[1, 0] == -1.0
    assert H_base[0, n_sites + 1] == 0.5
    assert H_base[n_sites + 1, 0] == 0.5
    assert H_base[1, n_sites] == -0.5
    assert H_base[n_sites, 1] == -0.5


def test_build_kitaev_operators_single_site_edge_case() -> None:
    H_base, H_mu_diag, Xi = _build_kitaev_operators(1, hopping=1.0, pairing=0.5)
    assert H_base.shape == (2, 2)
    assert torch.all(H_base == 0.0)
    assert H_mu_diag.shape == (2, 2)
    assert Xi.shape == (2, 2)


# ---------------------------------------------------------------------------
# _ExampleSirenStandIn
# ---------------------------------------------------------------------------


def test_example_siren_stand_in_forward_output_shape() -> None:
    model = _ExampleSirenStandIn(n_sites=3, hidden_features=8)
    output = model(torch.rand(5, 1))
    assert output.shape == (5, 6)


def test_example_siren_stand_in_output_is_unit_l2_normalised() -> None:
    model = _ExampleSirenStandIn(n_sites=3, hidden_features=8)
    output = model(torch.rand(5, 1))
    norms = torch.norm(output, dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_example_siren_stand_in_default_hidden_features_is_64() -> None:
    model = _ExampleSirenStandIn(n_sites=2)
    assert model.net[0].out_features == 64
    output = model(torch.rand(3, 1))
    assert output.shape == (3, 4)
