"""Search-tuning knobs (watercooler fork): each is an env override whose default keeps
the previous behaviour. Offline: no driver, no services."""

import os
import subprocess
import sys
from pathlib import Path

from graphiti_core.search import search_utils
from graphiti_core.search.search_utils import rrf

REPO = Path(__file__).resolve().parents[3]


def _module_value(module: str, name: str, env: dict) -> str:
    """`module.name` as imported in a fresh interpreter with `env` added (import-time knobs)."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith('GRAPHITI_')}
    clean.update(env, PYTHONPATH=str(REPO))
    out = subprocess.run(
        [sys.executable, '-c', f'import {module} as m; print(repr(m.{name}))'],
        env=clean,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    return out.stdout.strip()


# --- RRF rank constant -----------------------------------------------------------

# X is first in leg A and absent from leg B; Y is second in A and third in B.
LEGS = [['X', 'Y', 'Z'], ['W', 'V', 'Y']]


def test_rrf_rank_const_1_lets_a_single_leg_top_hit_win():
    uuids, scores = rrf(LEGS, rank_const=1)
    assert uuids[0] == 'X'
    assert scores[0] == 1.0


def test_rrf_rank_const_60_prefers_agreement_across_legs():
    uuids, scores = rrf(LEGS, rank_const=60)
    assert uuids[:2] == ['Y', 'X']
    assert scores[0] == 1 / 61 + 1 / 62


def test_rrf_without_rank_const_uses_module_default(monkeypatch):
    monkeypatch.setattr(search_utils, 'RRF_RANK_CONST', 60)
    assert rrf(LEGS)[0][0] == 'Y'
    monkeypatch.setattr(search_utils, 'RRF_RANK_CONST', 1)
    assert rrf(LEGS)[0][0] == 'X'


def test_rrf_rank_const_default_and_env_override():
    mod = 'graphiti_core.search.search_utils'
    assert _module_value(mod, 'RRF_RANK_CONST', {}) == '1'
    assert _module_value(mod, 'RRF_RANK_CONST', {'GRAPHITI_RRF_RANK_CONST': '60'}) == '60'
