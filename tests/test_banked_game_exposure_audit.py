"""Synthetic metadata controls for the banked structural exposure audit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location("exposure_audit", Path(__file__).resolve().parents[1] / "tools" / "banked_game_exposure_audit" / "audit_exposure.py")
assert spec is not None
assert spec.loader is not None
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture() -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, int]]:
    def ref(n):
        return {"path": "/synthetic/" + n + ".json", "sha256": "0" * 64}
    row: dict[str, Any] = {k: ref(k) for k in a.ROW_REFS}
    row.update(study="Q50", cycle=1, study_uuid="study", expected_games=192)
    bank = {"root_acceptance": row["bank_root"], "independent_review": row["bank_independent"]}
    games: list[dict[str, Any]] = [{"physical_game_id": "collection:" + str(i), "game_id": i, "bank": "train", "rows": 1 + i % 2,
              "label_class": "NATURAL", "shard": {"path": "/synthetic/unopened/" + str(i),
              "files": [{"path": str(i) + ".npy", "bytes": 1, "sha256": "0" * 64}]},
              "producer_proof": ref("unopened-proof-" + str(i))} for i in range(192)]
    candidate = {"schema": "CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1", "candidate_only": True,
                 "learning_admitted": False, "root_postclosure_promotion_required": True,
                 "collection_uuid": "collection", "run_uuid": "run", "source_revision": "source",
                 "generation": 1, "logical_generation": 1, "games": games}
    membership = [[1, a.digest(g["shard"]["files"]), g["rows"]] for g in games]
    pop = {"sources": 192, "rows": 288, "plan": row["compiler_plan"], "membership_sha256": a.digest(membership)}
    sampling = {"schema": "STORED_ROW_EXACT_ONCE_PASS_V2", "order": "python_random_mt19937_shuffle_v1",
                "order_seed": 7, "permutation_sha256": a.digest(a.permutation_for(288, 7)),
                "expected_rows": 288, "presentations_per_row": 1, "mirror_shares_quota": True,
                "partial_batch": "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING", "lineage_uuid": "run"}
    root: dict[str, Any] = {"status": "ROOT_ACCEPTS_CLOSED_CORE_GENERATION_DATA_BANK_V1", "audit_protocol": ref("unopened-protocol"),
            "data_spec": ref("unopened-spec"), "readout": ref("unopened-report"), "candidate": row["candidate"],
            "audit_source": {}, "audit_owner": {}, "learning_admitted": False}
    ind: dict[str, Any] = {k: root[k] for k in ("audit_protocol", "data_spec", "readout", "candidate")}
    ind.update(status="ACCEPT_CLOSED_CORE_GENERATION_DATA_BANK_NO_FINDINGS_V1", root_acceptance=row["bank_root"], findings=[])
    compiler = {"status": "ROOT_ACCEPTS_CLOSED_CAPPED_CYCLE_POPULATION_COMPILE_V1", "method": ref("whole-method"),
                "cycle": 1, "bank": bank, "population": pop, "sampling": sampling,
                "readout": ref("unopened-compiler-readout"), "owner_terminal": ref("unopened-terminal"), "training_authorized": False}
    contract = {"schema": "core_capped_on_policy_cycle_v1", "study_uuid": "study", "cycle": 1, "run_uuid": "run",
                "source_revision": "source", "bank": bank, "population": pop, "sampling": sampling,
                "batch_size": 256, "maximum_presentations": 1}
    plan = {"schema": "FIXED_SELFPLAY_UNION_PLAN_V1", "branches": [{"kind": "bank", **bank}], "copies": [],
            "expected_rows": 288, "expected_sources": 192}
    tech = {"status": "ROOT_ACCEPTS_CLOSED_CAPPED_CYCLE_TRAINING_TECHNICAL_V1", "contract": row["contract"],
            "method": compiler["method"], "supported_exit_code": 0, "actual_views": 288,
            "actual_updates": 2, "maximum_observed_presentations": 1, "last_batch_rows": 32}
    values: dict[str, Any] = dict(zip((row[k]["path"] for k in ("bank_root", "bank_independent", "candidate", "compiler_root", "compiler_plan", "contract", "learner_technical")),
                      (root, ind, candidate, compiler, plan, contract, tech)))
    class MemoryReader:
        def read(self, r, candidate=False):
            _ = candidate
            return values[r["path"]]
    return row, values, MemoryReader(), {"games_per_bank": 192, "rows_per_bank": 86400}


class MetadataControls(unittest.TestCase):
    def test_normalized_mass_is_exposure_not_independence(self):
        x = a.row_masses([{"rows": 1}, {"rows": 3}], 4)
        assert [g["mass"] for g in x["game_row_exposure_mass"]] == [.25, .75]
        assert x["sum_squared_game_exposure_masses"] == .625
        assert x["inverse_sum_squared_exposure_concentration"] == 1.6
        assert "not independent samples" in x["concentration_interpretation"]

    def test_explicit_batch_game_membership_includes_partial(self):
        x = a.batch_statistics([3, 5], [0, 3, 1, 4, 2], 2)
        assert [b["distinct_game_groups"] for b in x["batches"]] == [2, 2, 1]
        assert x["last_partial_batch"] == {"batch": 3, "rows": 1, "distinct_game_groups": 1, "largest_group_rows": 1}

    def test_real_receipt_join_reaches_structural_summary(self):
        row, _values, reader, caps = fixture()
        x = a.one_bank(row, reader, caps)
        assert (x["games"], x["rows"]) == (192, 288)
        assert x["actual_technical_dose"]["actual_views"] == 288
        assert x["batch_diversity"]["last_partial_batch"]["rows"] == 32
        assert x["mirror_shares_physical_row_quota"]

    def test_specific_candidate_and_membership_refusals(self):
        for mutation in ("candidate_ref", "physical_id", "row_total", "membership"):
            with self.subTest(mutation=mutation):
                row, values, reader, caps = fixture()
                a.one_bank(row, reader, caps)
                if mutation == "candidate_ref":
                    values[row["bank_root"]["path"]]["candidate"] = {"path": "/synthetic/other.json", "sha256": "1" * 64}
                elif mutation == "physical_id":
                    values[row["candidate"]["path"]]["games"][1]["physical_game_id"] = "collection:0"
                elif mutation == "row_total":
                    values[row["candidate"]["path"]]["games"][0]["rows"] = 3
                else:
                    values[row["compiler_root"]["path"]]["population"]["membership_sha256"] = "1" * 64
                with pytest.raises(ValueError, match={
                    "candidate_ref": "paired bank independent receipt",
                    "physical_id": "physical collection ordinal/order",
                    "row_total": "actual game/row totals",
                    "membership": "candidate slot order/rows do not match compiler membership",
                }[mutation]):
                    a.one_bank(row, reader, caps)

    def test_permutation_mismatch_is_unavailable_not_guessed(self):
        row, values, reader, caps = fixture()
        values[row["compiler_root"]["path"]]["sampling"]["permutation_sha256"] = "1" * 64
        x = a.one_bank(row, reader, caps)
        assert not x["batch_diversity"]["available"]
        assert x["batch_diversity"]["reason"] == "RECONSTRUCTED_PERMUTATION_SHA_MISMATCH"
        assert x["rows"] == 288

    def test_unknown_technical_schema_is_not_actual_dose(self):
        row, values, reader, caps = fixture()
        values[row["learner_technical"]["path"]]["status"] = "OTHER_HISTORICAL_SCHEMA"
        x = a.one_bank(row, reader, caps)
        assert not x["actual_technical_dose"]["available"]
        assert x["batch_diversity"]["basis"] == "SOURCE_AUTHENTICATED_COMPILER_ORDER_RECONSTRUCTION_NOT_JOURNAL_REPLAY"

    def test_compact_reader_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory).resolve() / "metadata.json"
            r = a.write_exclusive(p, {"x": 1})
            reader = a.Reader([r], {"compact_bytes": 1024, "bank_two_pass_bytes": 2048}, 2048)
            assert reader.read(r) == {"x": 1}
            reader.finish()
            with pytest.raises(FileExistsError):
                a.write_exclusive(p, {"x": 2})
            with pytest.raises(ValueError, match="unselected reference"):
                reader.read({"path": str(p.with_name("unselected.json")), "sha256": "0" * 64})

    def test_nonfinite_and_duplicate_json_refuse(self):
        for raw in (b'{"x":NaN}', b'{"x":1e400}', b'{"x":1,"x":2}'):
            with self.subTest(raw=raw), pytest.raises(ValueError, match=r"nonfinite JSON|duplicate JSON key"):
                a.parse(raw)


if __name__ == "__main__":
    unittest.main()
