import json
from math import pi

import numpy as np
import pytest

from drone_battle import match
from drone_battle.config import Config
from drone_battle.replay import write_replay
from drone_battle.state import make_state


def test_replay_records_commands_before_boundary_clipping(monkeypatch, tmp_path):
    cfg = Config(n_drones=1, bases_per_team=1, base_values=(1.0,), max_steps=1)
    before = make_state(cfg, [(39.6, 13), (23, 20)], [pi / 2, 0], [(8, 14), (32, 14)])
    monkeypatch.setattr(match, "reset", lambda _cfg, _rng: before)

    def east_and_turn(_obs, _cfg, _rng):
        return np.array([[1, 2]])

    result = match.play_episode(cfg, east_and_turn, east_and_turn, 42, record=True)
    label = "example </script>"
    path = write_replay(result, cfg, tmp_path / "replay.html", blue=label, seed=42)
    html = path.read_text(encoding="utf-8")
    data = json.loads(html.split('id="replay-data">', 1)[1].split("</script>", 1)[0])

    assert data["policies"][0] == label
    assert label not in html
    assert data["moveNames"][1] == "East"
    first, turn = data["frames"]
    assert first["pos"][0] == [39.6, 13]
    assert turn["actions"] == [[1, 2], [5, 0]]
    assert turn["pos"][0] == [40, 13]
    assert turn["disp"][0] == pytest.approx([0.4, 0])
    assert turn["heading"][0] == pytest.approx(3 * pi / 4)
    assert turn["reward"] == 0
    assert turn["cumulative"] == data["finalReturn"] == 0
    assert data["cause"] == "truncation"
