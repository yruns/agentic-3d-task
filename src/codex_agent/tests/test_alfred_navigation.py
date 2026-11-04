"""Observed-space navigation never treats unexplored gaps as traversable."""

from codex_agent.alfred.navigation import shortest_observed_path


def test_observed_path_avoids_unknown_shortcut() -> None:
    free = {(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)}
    assert shortest_observed_path((0, 0), {(2, 0)}, free) == (
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 0),
    )
    assert shortest_observed_path((0, 0), {(2, 0)}, {(0, 0), (2, 0)}) == ()
