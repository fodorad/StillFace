import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stillface.features.audio.lanes import (
    Aggregation,
    Band,
    BandSpec,
    Event,
    Lane,
    LaneSpec,
    LaneTable,
    band_to_events,
    grid_times,
    legend_from_dict,
    legend_to_dict,
    read_events,
    read_lanes,
    runs,
    threshold_band,
    to_grid,
    write_events,
    write_lanes,
)

PROB = LaneSpec("ast", "Baby cry", "baby", "prob", (0.0, 1.0))
F0 = LaneSpec("prosody", "f0", "mother", "Hz", (0.0, 2000.0), Aggregation.MEAN)
TYPE = LaneSpec(
    "fusion", "type", "baby", "category", (0.0, 2.0), Aggregation.MAJORITY, ("cry", "fuss", "laugh")
)


class TestGrid(unittest.TestCase):
    def test_length_and_times(self):
        t = grid_times(2.05)
        self.assertEqual(len(t), 21)
        self.assertAlmostEqual(t[-1], 2.0)
        self.assertEqual(len(grid_times(2.0)), 21)
        self.assertEqual(len(grid_times(0.0)), 1)

    def test_negative_duration(self):
        with self.assertRaises(ValueError):
            grid_times(-1)


class TestSpecs(unittest.TestCase):
    def test_name(self):
        self.assertEqual(PROB.name, "ast.Baby cry.baby")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            LaneSpec("a", "b", "nobody", "prob", (0, 1))
        with self.assertRaises(ValueError):
            LaneSpec("a", "b", "baby", "prob", (1, 0))
        with self.assertRaises(ValueError):
            LaneSpec("a", "b", "baby", "cat", (0, 1), Aggregation.MAJORITY)
        with self.assertRaises(ValueError):
            LaneSpec("a", "b", "baby", "prob", (0, 1), categories=("x",))
        with self.assertRaises(ValueError):
            BandSpec("x.on", "x", 0.5, "guess", "R1")

    def test_lane_range_and_shape(self):
        with self.assertRaises(ValueError):
            Lane(PROB, np.array([0.2, 1.5]))
        with self.assertRaises(ValueError):
            Lane(PROB, np.zeros((2, 2)))
        Lane(PROB, np.array([np.nan, 0.0, 1.0]))
        with self.assertRaises(ValueError):
            Band(BandSpec("x.on", "x", None, "config", "R"), np.zeros((2, 2)))


class TestToGrid(unittest.TestCase):
    def test_max_keeps_peaks_and_empty_is_nan(self):
        times = np.array([0.0, 0.02, 0.04, 0.31])
        lane = to_grid(times, np.array([0.1, 0.9, 0.3, 0.5]), 0.5, PROB)
        np.testing.assert_allclose(lane.values, [0.9, np.nan, np.nan, 0.5, np.nan, np.nan])

    def test_mean_ignores_nan(self):
        times = np.array([0.0, 0.01, 0.02, 0.1])
        lane = to_grid(times, np.array([100.0, np.nan, 200.0, 300.0]), 0.1, F0)
        np.testing.assert_allclose(lane.values, [150.0, 300.0])

    def test_majority_and_tie(self):
        times = np.array([0.0, 0.01, 0.02, 0.1, 0.11])
        lane = to_grid(times, np.array([2, 2, 1, 1, 0]), 0.1, TYPE)
        np.testing.assert_array_equal(lane.values, [2.0, 0.0])

    def test_outside_samples_dropped_and_shape(self):
        rng = np.random.default_rng(0)
        times = np.sort(rng.uniform(-1, 4, 500))
        lane = to_grid(times, rng.uniform(0, 1, 500), 3.0, PROB)
        self.assertEqual(lane.values.shape, (31,))
        self.assertTrue(np.isfinite(lane.values).all())

    def test_hold_fills_gaps_of_slow_sources(self):
        times = np.array([0.125, 0.375, 0.625])
        lane = to_grid(times, np.array([0.1, 0.5, 0.9]), 1.0, PROB, hold_s=0.175)
        np.testing.assert_allclose(lane.values[:9], [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
        self.assertTrue(np.isnan(lane.values[9:]).all())
        single = to_grid(np.array([0.5]), np.array([0.3]), 1.0, PROB, hold_s=0.2)
        self.assertEqual(int(np.isfinite(single.values).sum()), 5)

    def test_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            to_grid(np.zeros(3), np.zeros(2), 1.0, PROB)


class TestBandsAndEvents(unittest.TestCase):
    def test_threshold_band(self):
        lane = Lane(PROB, np.array([0.1, 0.5, np.nan, 0.7]))
        band = threshold_band(lane, 0.5, "probe", "B1")
        np.testing.assert_array_equal(band.values, [False, True, False, True])
        self.assertEqual(band.spec, BandSpec("ast.Baby cry.baby.on", PROB.name, 0.5, "probe", "B1"))

    def test_runs(self):
        self.assertEqual(runs(np.array([1, 1, 0, 1, 0, 0, 1], bool)), [(0, 2), (3, 4), (6, 7)])
        self.assertEqual(runs(np.zeros(3, bool)), [])

    def test_events_merge_and_min_duration(self):
        mask = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0], bool)
        band = Band(BandSpec("x.on", "x", 0.2, "config", "R7"), mask)
        conf = Lane(PROB, np.array([0.4, 0.6, 0.0, 0.8, 0, 0, 0, np.nan, 0, 0]))
        events = band_to_events(
            band, "baby", "cry", "fusion", conf, merge_gap_s=0.1, min_duration_s=0.2
        )
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertAlmostEqual(e.start_s, 0.0)
        self.assertAlmostEqual(e.end_s, 0.4)
        self.assertAlmostEqual(e.confidence, 0.45)
        self.assertEqual((e.threshold, e.rule_id), (0.2, "R7"))

    def test_events_default_confidence_and_bad_role(self):
        band = Band(BandSpec("x.on", "x", None, "config", "R"), np.array([0, 1, 0], bool))
        conf = Lane(PROB, np.array([np.nan] * 3))
        self.assertEqual(band_to_events(band, "baby", "x", "s", conf)[0].confidence, 1.0)
        with self.assertRaises(ValueError):
            band_to_events(band, "cat", "x", "s")


class TestIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def table(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 21)
        p[3] = np.nan
        lanes = [Lane(PROB, p), Lane(F0, rng.uniform(80, 400, 21)), Lane(TYPE, np.zeros(21))]
        bands = [threshold_band(lanes[0], 0.5, "vendor", "V1")]
        return LaneTable(2.0, lanes, bands, notes={"session": "x"})

    def test_lane_round_trip(self):
        table = self.table()
        write_lanes(table, self.dir / "l.csv", self.dir / "l.json")
        back = read_lanes(self.dir / "l.csv", self.dir / "l.json")
        self.assertEqual(back.legend, table.legend)
        for a, b in zip(table.lanes, back.lanes, strict=True):
            np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(back.bands[0].values, table.bands[0].values)
        self.assertTrue(math.isnan(back.lane(PROB.name).values[3]))
        self.assertIs(back.band("ast.Baby cry.baby.on"), back.bands[0])

    def test_lookup_errors(self):
        table = self.table()
        with self.assertRaises(KeyError):
            table.lane("nope")
        with self.assertRaises(KeyError):
            table.band("nope")

    def test_table_validation(self):
        with self.assertRaises(ValueError):
            LaneTable(1.0, [Lane(PROB, np.zeros(3))])
        with self.assertRaises(ValueError):
            LaneTable(0.1, [Lane(PROB, np.zeros(2)), Lane(PROB, np.zeros(2))])

    def test_header_mismatch(self):
        table = self.table()
        write_lanes(table, self.dir / "l.csv", self.dir / "l.json")
        (self.dir / "l.csv").write_text("time_s,other\n0.0,1\n")
        with self.assertRaises(ValueError):
            read_lanes(self.dir / "l.csv", self.dir / "l.json")

    def test_legend_version(self):
        data = legend_to_dict(self.table().legend)
        self.assertEqual(legend_from_dict(data), self.table().legend)
        data["version"] = 99
        with self.assertRaises(ValueError):
            legend_from_dict(data)

    def test_event_round_trip(self):
        events = [
            Event(0.1, 0.5, "baby", "cry", 0.75, "fusion", 0.2, "R1"),
            Event(1.0, 1.2, "mother", "speech", 1.0, "vtc", None, "R2"),
        ]
        write_events(events, self.dir / "e.csv")
        self.assertEqual(read_events(self.dir / "e.csv"), events)
        write_events([], self.dir / "empty.csv")
        self.assertEqual(read_events(self.dir / "empty.csv"), [])
        (self.dir / "bad.csv").write_text("a,b\n")
        with self.assertRaises(ValueError):
            read_events(self.dir / "bad.csv")


if __name__ == "__main__":
    unittest.main()
