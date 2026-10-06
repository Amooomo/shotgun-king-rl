"""Mechanism diagnostics for immediate-move-danger behaviour.

This module does **not** implement any attack rules. It consumes the danger
vector produced by :meth:`shotgun_king.env.ShotgunKingEnv._get_move_danger_features`
(the same source of truth used by the ``relative_v4`` observation) plus the
environment action mask, so the diagnostic and the observation feature are
guaranteed to share identical semantics.

Definitions (all counted on the decision state, i.e. before ``env.step``):

* ``decision_states_total``            -> one per policy decision.
* ``legal_moves_total``                -> action-mask legal MOVE (0-7) count.
* ``dangerous_legal_moves_total``      -> legal MOVE with immediate danger == 1.
* ``states_with_legal_move``           -> state has >= 1 legal MOVE.
* ``states_with_dangerous_legal_move`` -> state has >= 1 dangerous legal MOVE.
* ``selected_move_actions``            -> policy actually picked a MOVE action.
* ``selected_dangerous_moves``         -> picked MOVE that is legal + dangerous.
* ``selected_moves_in_danger_states``  -> picked MOVE in a state that exposes a
  dangerous legal MOVE.
"""

from dataclasses import dataclass

import numpy as np

from .core import Direction, MOVE_BASE


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Divide safely, returning ``0.0`` when the denominator is zero."""

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class MoveDangerState:
    """Per-decision snapshot of legal / dangerous legal MOVE actions."""

    legal_move_actions: tuple[int, ...]
    dangerous_move_actions: tuple[int, ...]

    @property
    def has_legal_move(self) -> bool:
        return bool(self.legal_move_actions)

    @property
    def has_dangerous_legal_move(self) -> bool:
        return bool(self.dangerous_move_actions)


class MoveDangerDiagnostics:
    """Accumulate immediate-move-danger counters across decisions."""

    def __init__(self) -> None:

        self.decision_states_total = 0
        self.states_with_legal_move = 0
        self.states_with_dangerous_legal_move = 0

        self.legal_moves_total = 0
        self.dangerous_legal_moves_total = 0

        self.selected_move_actions = 0
        self.selected_dangerous_moves = 0
        self.selected_moves_in_danger_states = 0

    def observe(
        self,
        action_mask,
        danger,
    ) -> MoveDangerState:
        """Record one decision state.

        ``action_mask`` is the environment action mask (length ``N_ACTIONS``).
        ``danger`` is the 8-dim immediate move danger vector, ordered by
        :class:`shotgun_king.core.Direction`.

        Returns a :class:`MoveDangerState` to be passed to :meth:`select`.
        """

        danger = np.asarray(
            danger,
            dtype=np.float32,
        ).reshape(-1)

        legal_move_actions = []
        dangerous_move_actions = []

        for direction in Direction:

            action = (
                MOVE_BASE
                + int(direction)
            )

            is_legal = bool(
                action_mask[action]
            )

            if not is_legal:
                continue

            legal_move_actions.append(action)

            # ``_get_move_danger_features`` already returns 0.0 for illegal
            # moves, but we still intersect with legality explicitly so the
            # definition cannot drift.
            if float(danger[int(direction)]) == 1.0:
                dangerous_move_actions.append(action)

        self.decision_states_total += 1

        self.legal_moves_total += len(
            legal_move_actions
        )

        self.dangerous_legal_moves_total += len(
            dangerous_move_actions
        )

        if legal_move_actions:
            self.states_with_legal_move += 1

        if dangerous_move_actions:
            self.states_with_dangerous_legal_move += 1

        return MoveDangerState(
            legal_move_actions=tuple(
                legal_move_actions
            ),
            dangerous_move_actions=tuple(
                dangerous_move_actions
            ),
        )

    def select(
        self,
        state: MoveDangerState,
        action: int,
    ) -> None:
        """Record the action finally chosen for ``state``."""

        action = int(action)

        is_move = (
            MOVE_BASE
            <= action
            < MOVE_BASE + len(Direction)
        )

        if not is_move:
            return

        self.selected_move_actions += 1

        if action in state.dangerous_move_actions:
            self.selected_dangerous_moves += 1

        if state.has_dangerous_legal_move:
            self.selected_moves_in_danger_states += 1

    def counts(self) -> dict:
        """Raw counters."""

        return {
            "decision_states_total":
                self.decision_states_total,

            "states_with_legal_move":
                self.states_with_legal_move,

            "states_with_dangerous_legal_move":
                self.states_with_dangerous_legal_move,

            "legal_moves_total":
                self.legal_moves_total,

            "dangerous_legal_moves_total":
                self.dangerous_legal_moves_total,

            "selected_move_actions":
                self.selected_move_actions,

            "selected_dangerous_moves":
                self.selected_dangerous_moves,

            "selected_moves_in_danger_states":
                self.selected_moves_in_danger_states,
        }

    def rates(self) -> dict:
        """Derived mechanism rates (zero-safe)."""

        return {
            "dangerous_state_exposure_rate":
                safe_ratio(
                    self.states_with_dangerous_legal_move,
                    self.decision_states_total,
                ),

            "dangerous_move_share":
                safe_ratio(
                    self.dangerous_legal_moves_total,
                    self.legal_moves_total,
                ),

            "dangerous_selection_rate":
                safe_ratio(
                    self.selected_dangerous_moves,
                    self.states_with_dangerous_legal_move,
                ),

            "dangerous_rate_given_move_in_danger_state":
                safe_ratio(
                    self.selected_dangerous_moves,
                    self.selected_moves_in_danger_states,
                ),

            "overall_dangerous_action_rate":
                safe_ratio(
                    self.selected_dangerous_moves,
                    self.decision_states_total,
                ),
        }

    def summary(self) -> dict:
        """Counts + rates in one dict."""

        return {
            **self.counts(),
            **self.rates(),
        }
