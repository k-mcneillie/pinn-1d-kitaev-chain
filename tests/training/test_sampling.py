"""Tests for kitaev.training.sampling and the trainer callback hook."""

from __future__ import annotations

import pytest
import torch

from kitaev.data.mu_sampler import MuSampler
from kitaev.data.sampling_region import SamplingRegion
from kitaev.models import SirenPINNChiral
from kitaev.training.callbacks import TrainingCallback
from kitaev.training.config import TrainerConfig
from kitaev.training.loss import ChiralFSMLoss
from kitaev.training.sampling import (
    AdaptiveSampling,
    CurriculumSampling,
    FixedRegionSampling,
    SamplingConfig,
    StreamingMuDataset,
    build_sampling,
    streaming_dataloader,
)
from kitaev.training.trainer import UnifiedTrainer, _build_kitaev_operators

REGIONS = (
    SamplingRegion(low=0.05, high=4.0, weight=1.0),
    SamplingRegion(low=1.8, high=2.2, weight=1.0),
)


# ---------------------------------------------------------------------------
# StreamingMuDataset / streaming_dataloader
# ---------------------------------------------------------------------------


def test_streaming_dataset_yields_fixed_count_of_fresh_batches() -> None:
    strategy = FixedRegionSampling.from_regions(REGIONS)
    dataset = StreamingMuDataset(strategy, batch_size=16, steps_per_epoch=5)

    epoch_one = list(dataset)
    epoch_two = list(dataset)

    assert len(epoch_one) == 5
    assert all(item[0].shape == (16, 1) for item in epoch_one)
    # Fresh draws: no batch repeats across epochs.
    assert not torch.equal(epoch_one[0][0], epoch_two[0][0])


def test_streaming_dataloader_passes_batches_through_untouched() -> None:
    loader = streaming_dataloader(
        FixedRegionSampling.from_regions(REGIONS), batch_size=8, steps_per_epoch=3
    )
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (8, 1)


# ---------------------------------------------------------------------------
# FixedRegionSampling
# ---------------------------------------------------------------------------


def test_fixed_region_sampling_draws_within_regions() -> None:
    strategy = FixedRegionSampling(MuSampler(REGIONS))
    batch = strategy.draw(2000)
    assert batch.shape == (2000, 1)
    assert batch.min() >= 0.05 - 1e-6
    assert batch.max() <= 4.0 + 1e-6


# ---------------------------------------------------------------------------
# CurriculumSampling
# ---------------------------------------------------------------------------


def test_curriculum_switches_active_stage_on_epoch_start() -> None:
    model = SirenPINNChiral(n_sites=4)
    strategy = CurriculumSampling(
        [
            (1, (SamplingRegion(low=0.0, high=1.0, weight=1.0),)),
            (5, REGIONS),
        ]
    )

    strategy.on_epoch_start(1, model, None)
    early = strategy.draw(3000)
    assert early.max() <= 1.0 + 1e-6

    strategy.on_epoch_start(5, model, None)
    late = strategy.draw(3000)
    assert late.max() > 2.0


def test_curriculum_orders_unsorted_stages() -> None:
    strategy = CurriculumSampling(
        [
            (10, (SamplingRegion(low=0.0, high=4.0, weight=1.0),)),
            (1, (SamplingRegion(low=0.0, high=0.5, weight=1.0),)),
        ]
    )
    assert [start for start, _ in strategy.stages] == [1, 10]


def test_curriculum_rejects_empty_schedule() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        CurriculumSampling([])


# ---------------------------------------------------------------------------
# AdaptiveSampling
# ---------------------------------------------------------------------------


def test_adaptive_is_base_only_before_first_refinement() -> None:
    strategy = AdaptiveSampling(MuSampler(REGIONS), n_sites=6)
    assert strategy._hot is None
    batch = strategy.draw(64)
    assert batch.shape == (64, 1)


def test_adaptive_refines_on_schedule_only() -> None:
    model = SirenPINNChiral(n_sites=6)
    strategy = AdaptiveSampling(
        MuSampler(REGIONS),
        n_sites=6,
        refine_every=3,
        candidate_pool=200,
        keep_fraction=0.25,
    )

    strategy.on_epoch_end(2, model, None)
    assert strategy._hot is None  # not a multiple of refine_every

    strategy.on_epoch_end(3, model, None)
    assert strategy._hot is not None
    assert strategy._hot.shape == (50,)  # 200 * 0.25

    # Hot points are the worst-residual candidates.
    from kitaev.training.loss import chiral_pointwise_residual

    hot = strategy._hot.unsqueeze(-1)
    residual = chiral_pointwise_residual(model, hot, 6)
    assert residual.min() > 0.0


def test_adaptive_draw_mixes_and_clamps_after_refinement() -> None:
    model = SirenPINNChiral(n_sites=6)
    strategy = AdaptiveSampling(
        MuSampler(REGIONS),
        n_sites=6,
        refine_every=1,
        candidate_pool=128,
        adaptive_fraction=0.5,
        jitter=0.1,
        domain=(0.0, 4.0),
    )
    strategy.on_epoch_end(1, model, None)
    batch = strategy.draw(80)
    assert batch.shape == (80, 1)
    assert batch.min() >= 0.0
    assert batch.max() <= 4.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"keep_fraction": 0.0}, "keep_fraction"),
        ({"keep_fraction": 1.5}, "keep_fraction"),
        ({"adaptive_fraction": -0.1}, "adaptive_fraction"),
        ({"adaptive_fraction": 1.1}, "adaptive_fraction"),
    ],
)
def test_adaptive_validates_fractions(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        AdaptiveSampling(MuSampler(REGIONS), n_sites=6, **kwargs)


# ---------------------------------------------------------------------------
# SamplingConfig / build_sampling
# ---------------------------------------------------------------------------


def test_sampling_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown sampling mode"):
        SamplingConfig(mode="turbo")


def test_build_sampling_frozen_returns_pool_loader_and_no_callbacks() -> None:
    config = SamplingConfig(mode="frozen", batch_size=64, total_samples=256)
    loader, callbacks = build_sampling(config, REGIONS)
    assert callbacks == []
    batches = list(loader)
    assert len(batches) == 256 // 64
    assert batches[0][0].shape == (64, 1)


@pytest.mark.parametrize("mode", ["infinite", "curriculum", "adaptive"])
def test_build_sampling_streaming_modes_return_strategy_callback(mode) -> None:
    config = SamplingConfig(mode=mode, batch_size=32, steps_per_epoch=4, n_sites=6)
    loader, callbacks = build_sampling(config, REGIONS)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], TrainingCallback)
    batches = list(loader)
    assert len(batches) == 4
    assert batches[0][0].shape == (32, 1)


def test_build_sampling_curriculum_falls_back_to_base_regions() -> None:
    config = SamplingConfig(mode="curriculum", batch_size=8, steps_per_epoch=2)
    _, callbacks = build_sampling(config, REGIONS)
    assert isinstance(callbacks[0], CurriculumSampling)
    assert len(callbacks[0].stages) == 1


# ---------------------------------------------------------------------------
# Trainer callback hook
# ---------------------------------------------------------------------------


class _SpyCallback(TrainingCallback):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def on_epoch_start(self, epoch, model, history) -> None:
        del model, history
        self.calls.append(("start", epoch))

    def on_epoch_end(self, epoch, model, history) -> None:
        del model, history
        self.calls.append(("end", epoch))


def test_trainer_invokes_callbacks_each_epoch_in_order(
    accelerator, make_session
) -> None:
    n_sites = 4
    model = SirenPINNChiral(n_sites=n_sites, hidden_features=8, hidden_layers=1)
    H_base, H_mu_diag, Xi = _build_kitaev_operators(n_sites, 1.0, 0.5)
    loader, strategy_callbacks = build_sampling(
        SamplingConfig(mode="infinite", batch_size=32, steps_per_epoch=2), REGIONS
    )
    spy = _SpyCallback()

    trainer = UnifiedTrainer(
        session=make_session(),
        accelerator=accelerator,
        model=model,
        loss_fn=ChiralFSMLoss(n_sites=n_sites),
        optimiser=torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=TrainerConfig(epochs=3, print_freq=100, patience=None),
        callbacks=[spy, *strategy_callbacks],
    )
    trainer.fit(loader, H_base, H_mu_diag, Xi)

    assert spy.calls == [
        ("start", 1),
        ("end", 1),
        ("start", 2),
        ("end", 2),
        ("start", 3),
        ("end", 3),
    ]


if __name__ == "__main__":
    pytest.main()
