import json
import tempfile
import unittest
from pathlib import Path

from stillface.features.gaze.config import GazeConfig
from stillface.features.gaze.heads import HeadBox, HeadTrack
from stillface.features.gaze.roles import (
    RoleReferences,
    assign_roles,
    camera_persons,
    head_per_frame,
    load_role_overrides,
    missing_reason,
)
from tests.features.gaze.fakes import box, track

CFG = GazeConfig()
N = 10


def assign(tracks, camera="window", phase="play", **kwargs):
    return assign_roles(tracks, camera, phase, N, CFG, **kwargs)


class TestSidePrior(unittest.TestCase):
    def test_window_mother_is_on_the_right_even_if_smaller(self) -> None:
        result = assign([track(1, 0.3, width=0.12), track(2, 0.8, width=0.09)])
        self.assertEqual((result.mother, result.baby), (2, 1))
        self.assertEqual(result.flags, ())

    def test_door_mother_is_on_the_left(self) -> None:
        result = assign([track(1, 0.8), track(2, 0.15)], camera="door", phase="stillface")
        self.assertEqual((result.mother, result.baby), (2, 1))

    def test_track_order_does_not_matter(self) -> None:
        a = assign([track(1, 0.3), track(2, 0.8)])
        b = assign([track(2, 0.8), track(1, 0.3)])
        self.assertEqual((a.mother, a.baby), (b.mother, b.baby))

    def test_candidates_at_the_same_x_are_ambiguous(self) -> None:
        result = assign([track(1, 0.5), track(2, 0.52)])
        self.assertEqual((result.mother, result.baby), (None, None))
        self.assertIn("roles_ambiguous", result.flags)

    def test_single_track_is_not_assigned_by_image_half(self) -> None:
        for x in (0.3, 0.8):
            result = assign([track(1, x)])
            self.assertEqual((result.mother, result.baby), (None, None))
            self.assertEqual(result.flags, ("single_track", "roles_ambiguous"))

    def test_single_track_can_be_pinned_by_roles_json(self) -> None:
        overrides = {("window", "play"): {"mother": 1}}
        result = assign([track(1, 0.3)], overrides=overrides)
        self.assertEqual((result.mother, result.baby), (1, None))

    def test_no_tracks(self) -> None:
        result = assign([])
        self.assertEqual((result.mother, result.baby, result.flags), (None, None, ()))


class TestBackgroundPeople(unittest.TestCase):
    def test_short_lived_track_is_ignored(self) -> None:
        background = track(3, 0.5, width=0.2, frames=[0, 1])
        result = assign([track(1, 0.3), track(2, 0.8), background])
        self.assertEqual((result.mother, result.baby), (2, 1))

    def test_smaller_third_person_is_ignored(self) -> None:
        result = assign([track(1, 0.3), track(2, 0.8), track(3, 0.55, width=0.03)])
        self.assertEqual((result.mother, result.baby), (2, 1))


class TestBaseline(unittest.TestCase):
    def test_baseline_ignores_the_side_prior_and_uses_size(self) -> None:
        result = assign([track(1, 0.2, width=0.14), track(2, 0.7, width=0.09)], phase="baseline")
        self.assertEqual((result.mother, result.baby), (1, 2))

    def test_similar_sizes_in_baseline_are_ambiguous(self) -> None:
        result = assign([track(1, 0.2, width=0.1), track(2, 0.7, width=0.095)], phase="baseline")
        self.assertEqual((result.mother, result.baby), (None, None))
        self.assertIn("roles_ambiguous", result.flags)

    def test_identity_beats_size_in_baseline(self) -> None:
        refs = RoleReferences(mother=(1.0, 0.0), baby=(0.0, 1.0))
        tracks = [
            track(1, 0.2, width=0.14, embedding=(0.0, 1.0)),
            track(2, 0.7, width=0.09, embedding=(1.0, 0.0)),
        ]
        result = assign(tracks, phase="baseline", references=refs)
        self.assertEqual((result.mother, result.baby), (2, 1))

    def test_weak_identity_match_falls_back_to_size(self) -> None:
        refs = RoleReferences(mother=(1.0, 0.0), baby=(0.0, 1.0))
        tracks = [
            track(1, 0.2, width=0.14, embedding=(0.7, -0.7)),
            track(2, 0.7, width=0.09, embedding=(-0.7, 0.7)),
        ]
        result = assign(tracks, phase="baseline", references=refs)
        self.assertEqual((result.mother, result.baby), (1, 2))


class TestTaggedTracks(unittest.TestCase):
    def tagged(self, track_id: int, cx: float, role: str, width: float = 0.07) -> HeadTrack:
        return HeadTrack(track_id, {f: box(cx, width=width) for f in range(10)}, role=role)

    def test_prompted_baby_and_best_face_for_the_mother(self) -> None:
        baby = self.tagged(1000, 0.3, "baby")
        result = assign([baby, track(2, 0.4, width=0.09), track(3, 0.85, width=0.12)])
        self.assertEqual((result.mother, result.baby), (3, 1000))

    def test_tiny_background_face_is_not_the_mother(self) -> None:
        baby = self.tagged(1000, 0.3, "baby")
        result = assign([baby, track(2, 0.6, width=0.015)])
        self.assertEqual((result.mother, result.baby), (None, 1000))

    def test_both_roles_prompted(self) -> None:
        result = assign([self.tagged(1000, 0.3, "baby"), self.tagged(1001, 0.8, "mother")])
        self.assertEqual((result.mother, result.baby), (1001, 1000))

    def test_prompted_baby_works_in_baseline_without_position_priors(self) -> None:
        baby = self.tagged(1000, 0.8, "baby")
        result = assign([baby, track(2, 0.2, width=0.09)], phase="baseline")
        self.assertEqual((result.mother, result.baby), (2, 1000))

    def test_short_lived_faces_are_not_candidates(self) -> None:
        baby = self.tagged(1000, 0.3, "baby")
        result = assign([baby, track(2, 0.8, width=0.1, frames=[0])])
        self.assertEqual((result.mother, result.baby), (None, 1000))


class TestCloseCameras(unittest.TestCase):
    def test_owner_is_the_dominant_track(self) -> None:
        tracks = [track(1, 0.5, width=0.3), track(2, 0.1, width=0.05)]
        mother = assign(tracks, camera="mother")
        self.assertEqual((mother.mother, mother.baby), (1, None))
        baby = assign(tracks, camera="baby", phase="baseline")
        self.assertEqual((baby.mother, baby.baby), (None, 1))

    def test_confident_face_beats_a_larger_weak_partner_box(self) -> None:
        owner = track(1, 0.5, width=0.2, score=0.88)
        partner_edge = track(2, 0.05, width=0.5, score=0.5)  # partner's face entering the view
        result = assign([partner_edge, owner], camera="baby")
        self.assertEqual((result.mother, result.baby), (None, 1))

    def test_close_camera_without_head(self) -> None:
        result = assign([], camera="baby")
        self.assertEqual((result.mother, result.baby), (None, None))


class TestOverrides(unittest.TestCase):
    def test_override_wins_over_automatic_assignment(self) -> None:
        tracks = [track(1, 0.3), track(2, 0.8), track(3, 0.6)]
        overrides = {("window", "play"): {"mother": 3, "baby": 1}}
        result = assign(tracks, overrides=overrides)
        self.assertEqual((result.mother, result.baby), (3, 1))
        self.assertEqual(result.flags, ())

    def test_override_for_another_phase_is_not_applied(self) -> None:
        overrides = {("window", "baseline"): {"mother": 1, "baby": 2}}
        result = assign([track(1, 0.3), track(2, 0.8)], overrides=overrides)
        self.assertEqual((result.mother, result.baby), (2, 1))

    def test_override_resolves_an_ambiguous_session(self) -> None:
        overrides = {("window", "play"): {"mother": 2, "baby": 1}}
        result = assign([track(1, 0.5), track(2, 0.52)], overrides=overrides)
        self.assertEqual((result.mother, result.baby), (2, 1))

    def test_load_role_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roles.json"
            path.write_text(json.dumps({"window": {"play": {"mother": 4, "baby": 2}}}))
            self.assertEqual(
                load_role_overrides(path), {("window", "play"): {"mother": 4, "baby": 2}}
            )
            self.assertEqual(load_role_overrides(Path(tmp) / "missing.json"), {})

    def test_load_role_overrides_validates(self) -> None:
        bad_documents = [
            {"ceiling": {"play": {"mother": 1}}},
            {"window": {"dinner": {"mother": 1}}},
            {"window": {"play": {"father": 1}}},
            {"window": {"play": {"mother": "x"}}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for document in bad_documents:
                path = Path(tmp) / "roles.json"
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    load_role_overrides(path)


class TestHeadPerFrame(unittest.TestCase):
    def test_boxes_follow_the_assigned_tracks(self) -> None:
        mother = HeadTrack(2, {0: box(0.8), 2: box(0.81)})
        baby = HeadTrack(1, {0: box(0.3), 1: box(0.3), 2: box(0.3)})
        result = assign([baby, mother])
        heads = head_per_frame(result, [baby, mother], [0, 1, 2])
        self.assertEqual(heads["baby"], [box(0.3)] * 3)
        self.assertEqual(heads["mother"], [box(0.8), None, box(0.81)])

    def test_unassigned_person_has_no_heads(self) -> None:
        heads = head_per_frame(assign([]), [], [0, 1])
        self.assertEqual(heads, {"mother": [None, None], "baby": [None, None]})

    def test_camera_persons(self) -> None:
        self.assertEqual(camera_persons("window"), ("mother", "baby"))
        self.assertEqual(camera_persons("door"), ("mother", "baby"))
        self.assertEqual(camera_persons("baby"), ("baby",))
        self.assertEqual(camera_persons("mother"), ("mother",))

    def test_missing_reason(self) -> None:
        self.assertEqual(missing_reason("door", "mother"), "occluded")
        self.assertEqual(missing_reason("door", "baby"), "no_head")
        self.assertEqual(missing_reason("window", "mother"), "no_head")


class TestHeadTrackEvidence(unittest.TestCase):
    def test_evidence_and_weight_use_the_scores(self) -> None:
        scored = HeadTrack(
            1, {0: box(0.5, width=0.2), 1: box(0.5, width=0.2)}, scores={0: 0.5, 1: 1.0}
        )
        self.assertAlmostEqual(scored.evidence(), 1.5)
        area = box(0.5, width=0.2).area
        self.assertAlmostEqual(scored.weight(), area * 1.5)
        unscored = HeadTrack(2, {0: box(0.5, width=0.2)})
        self.assertAlmostEqual(unscored.evidence(), 1.0)


class TestHeadBox(unittest.TestCase):
    def test_geometry(self) -> None:
        head = HeadBox(0.2, 0.2, 0.4, 0.6)
        self.assertAlmostEqual(head.area, 0.08)
        self.assertAlmostEqual(head.center[0], 0.3)
        self.assertAlmostEqual(head.center[1], 0.4)

    def test_expand_is_relative_and_clipped(self) -> None:
        grown = HeadBox(0.0, 0.2, 0.2, 0.4).expand(0.5)
        for got, want in zip(vars(grown).values(), (0.0, 0.1, 0.3, 0.5), strict=True):
            self.assertAlmostEqual(got, want)


if __name__ == "__main__":
    unittest.main()
