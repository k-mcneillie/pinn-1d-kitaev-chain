"""Symbolic read-out of a trained operator. Deferred to the operator phase.

This is a placeholder for the interpretability step that only becomes
meaningful once the ``(mu, N, Delta)`` neural operator exists. Two things
are planned.

- Fit a closed form to the operator's effective localisation length as a
  function of ``mu`` and ``Delta``, and check whether it recovers the decay
  rate implied by the characteristic equation
  ``(t + Delta) z^2 + mu z + (t - Delta) = 0``.
- Compare a DeepONet trunk basis against the exact BdG edge and bulk modes.

Neither is implemented here. See ``docs/markdown/xai-methods.md`` for the
intended design.
"""

from __future__ import annotations


def fit_localisation_length(*args: object, **kwargs: object) -> None:
    """Not implemented. Planned for the neural-operator phase.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "Symbolic localisation-length read-out is planned for the "
        "neural-operator phase; see docs/markdown/xai-methods.md."
    )
