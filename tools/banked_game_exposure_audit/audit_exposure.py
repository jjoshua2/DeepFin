"""Owned-phase-only structural exposure audit. Stdlib; no payload or target reads.
Root supplies every path/hash. This is not an admission controller or bank validator.
"""
from __future__ import annotations
import argparse
import bisect
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
from typing import Any

MAX_BANKS = 24
MAX_GAMES = 192
MAX_ROWS = 86400
COMPACT = 4 << 20
CANDIDATE = 32 << 20
TOTAL = 4 << 30
ROW_REFS = {"bank_root", "bank_independent", "candidate", "compiler_root", "compiler_plan", "contract", "learner_technical"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value: Any, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, "invalid stored-row integer")
    return value


def permutation_for(population_rows: int, seed: int) -> list[int]:
    """Uniform unweighted shuffle with an independent explicit-seed RNG.

    The contract pins its digest and state pins the full permutation. A runtime
    whose shuffle differs refuses the contract/restore instead of remapping rows.
    Never seed or consume the caller's augmentation RNG here.
    """
    order = list(range(integer(population_rows, 1)))
    random.Random(integer(seed)).shuffle(order)
    return order


def reference(value):
    require(type(value) is dict and set(value) == {"path", "sha256"}, "strict reference")
    p, h = value["path"], value["sha256"]
    require(type(p) is str and Path(p).is_absolute() and str(Path(p)) == p and ".." not in Path(p).parts
            and Path(p).suffix == ".json", "canonical JSON reference")
    require(type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h), "SHA256")
    return value


def stamp(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode


def pairs(items):
    out = {}
    for k, v in items:
        require(k not in out, "duplicate JSON key")
        out[k] = v
    return out


def finite_float(text):
    v = float(text)
    require(math.isfinite(v), "nonfinite JSON float")
    return v


def bad_constant(text):
    raise ValueError("nonfinite JSON constant: " + text)


def parse(raw):
    v = json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float, parse_constant=bad_constant)
    require(type(v) is dict, "JSON object required")
    return v


class Reader:
    """Only explicit files; reserve first read plus final hash recheck in advance."""
    def __init__(self, selected, caps, total_remaining):
        self.allowed = {}
        for r in selected:
            reference(r)
            require(r["path"] not in self.allowed or self.allowed[r["path"]] == r["sha256"], "conflicting pins")
            self.allowed[r["path"]] = r["sha256"]
        self.caps, self.remaining = caps, total_remaining
        self.files = {}
        self.reserved = 0

    def raw(self, path, expected):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            require(stamp(os.fstat(fd)) == expected, "opened metadata changed")
            chunks, left = [], expected[2]
            while left:
                b = os.read(fd, min(left, 1 << 20))
                require(bool(b), "short metadata read")
                chunks.append(b)
                left -= len(b)
            require(stamp(os.fstat(fd)) == stamp(path.lstat()) == expected, "unstable metadata")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def read(self, r, candidate=False):
        reference(r)
        name = r["path"]
        require(self.allowed.get(name) == r["sha256"], "unselected reference; no recursive reads")
        if name in self.files:
            return self.files[name][2]
        p = Path(name)
        require(p.resolve() == p, "canonical metadata path")
        s = p.lstat()
        cap = self.caps["candidate_bytes" if candidate else "compact_bytes"]
        require(stat.S_ISREG(s.st_mode) and not s.st_mode & 0o222 and 0 < s.st_size <= cap,
                "bounded sealed regular metadata")
        require(self.reserved + 2 * s.st_size <= min(self.remaining, self.caps["bank_two_pass_bytes"]),
                "reserve both reads before opening")
        self.reserved += 2 * s.st_size
        raw = self.raw(p, stamp(s))
        require(hashlib.sha256(raw).hexdigest() == r["sha256"], "metadata SHA differs")
        v = parse(raw)
        self.files[name] = (r["sha256"], stamp(s), v)
        return v

    def finish(self):
        for name, (sha, before, _) in self.files.items():
            p = Path(name)
            require(p.resolve() == p, "metadata path changed")
            require(hashlib.sha256(self.raw(p, before)).hexdigest() == sha, "final metadata hash differs")
        return self.reserved


def metadata_games(candidate, maximum_games, maximum_rows):
    require(candidate["schema"] == "CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1"
            and candidate["candidate_only"] is True and candidate["learning_admitted"] is False
            and candidate["root_postclosure_promotion_required"] is True, "TRAIN candidate only")
    games = candidate["games"]
    require(type(games) is list and 0 < len(games) <= maximum_games, "game count cap")
    ids, sources, result, total = set(), set(), [], 0
    for ordinal, g in enumerate(games):
        require(type(g) is dict and set(g) == {"physical_game_id", "game_id", "bank", "rows", "label_class", "shard", "producer_proof"}, "game descriptor schema")
        require(g["physical_game_id"] == candidate["collection_uuid"] + ":" + str(ordinal), "physical collection ordinal/order")
        integer(g["game_id"])
        require(g["physical_game_id"] not in ids and g["bank"] == "train"
                and g["label_class"] in {"NATURAL", "TB_ADJUDICATED"}, "duplicate/protected/unresolved game")
        ids.add(g["physical_game_id"])
        rows = integer(g["rows"], 1)
        total += rows
        require(total <= maximum_rows, "row cap before permutation")
        sh = g["shard"]
        require(type(sh) is dict and set(sh) == {"path", "files"} and type(sh["files"]) is list and sh["files"], "recorded shard descriptor")
        # Only descriptor bytes are hashed. Never stat/open any referenced shard/file/proof.
        source = digest(sh["files"])
        require(source not in sources, "duplicate source asset/alias")
        sources.add(source)
        reference(g["producer_proof"])
        result.append({"physical_game_id": g["physical_game_id"], "source_sha256": source, "rows": rows})
    return result, total


def row_masses(games, n):
    counts = sorted(g["rows"] for g in games)
    require(sum(counts) == n and n > 0, "row mass total")
    square = sum(x * x for x in counts)
    return {"rows_per_game_histogram": [[k, v] for k, v in sorted(Counter(counts).items())],
            "rows_per_game_min": counts[0], "rows_per_game_max": counts[-1],
            "game_row_exposure_mass": [{**g, "mass": g["rows"] / n} for g in games],
            "sum_squared_game_exposure_masses": square / (n * n),
            "inverse_sum_squared_exposure_concentration": n * n / square,
            "concentration_interpretation": "Kish-style exposure concentration only; not independent samples or gradient ESS"}


def batch_statistics(ends, order, batch):
    result = []
    for lo in range(0, len(order), batch):
        members = Counter(bisect.bisect_right(ends, i) for i in order[lo:lo + batch])
        result.append({"batch": len(result) + 1, "rows": min(batch, len(order) - lo),
                       "distinct_game_groups": len(members), "largest_group_rows": max(members.values())})
    hist = Counter(x["distinct_game_groups"] for x in result)
    return {"available": True, "basis": "SOURCE_AUTHENTICATED_COMPILER_ORDER_RECONSTRUCTION_NOT_JOURNAL_REPLAY",
            "batches": result, "distinct_game_count_histogram": [[k, v] for k, v in sorted(hist.items())],
            "last_partial_batch": result[-1] if len(order) % batch else None,
            "minimum_distinct_games": min(hist), "maximum_distinct_games": max(hist)}


def one_bank(row, reader, caps):
    require(set(row) == ROW_REFS | {"study", "cycle", "study_uuid", "expected_games"}
            and row["study"] in {"F12", "Q50"} and 1 <= integer(row["cycle"], 1) <= 12
            and row["expected_games"] == 192, "explicit selected F12/Q50 bank row")
    root, ind = reader.read(row["bank_root"]), reader.read(row["bank_independent"])
    require(set(root) == {"status", "audit_protocol", "data_spec", "readout", "candidate", "audit_source", "audit_owner", "learning_admitted"}
            and root["status"] == "ROOT_ACCEPTS_CLOSED_CORE_GENERATION_DATA_BANK_V1" and root["learning_admitted"] is False,
            "genuine caller-selected closed TRAIN Root8")
    require(set(ind) == {"status", "root_acceptance", "audit_protocol", "data_spec", "readout", "candidate", "findings"}
            and ind["status"] == "ACCEPT_CLOSED_CORE_GENERATION_DATA_BANK_NO_FINDINGS_V1"
            and ind["root_acceptance"] == row["bank_root"] and ind["findings"] == []
            and all(ind[k] == root[k] for k in ("audit_protocol", "data_spec", "readout", "candidate")), "paired bank independent receipt")
    require(root["candidate"] == row["candidate"], "selected candidate hash/path differs")
    candidate = reader.read(row["candidate"], candidate=True)
    c, compiler, plan = reader.read(row["contract"]), reader.read(row["compiler_root"]), reader.read(row["compiler_plan"])
    bank = {"root_acceptance": row["bank_root"], "independent_review": row["bank_independent"]}
    require(c["schema"] == "core_capped_on_policy_cycle_v1" and c["study_uuid"] == row["study_uuid"]
            and c["cycle"] == row["cycle"] == candidate["generation"] == candidate["logical_generation"]
            and c["run_uuid"] == candidate["run_uuid"] and c["source_revision"] == candidate["source_revision"]
            and c["bank"] == bank, "contract/candidate/source-qualified cycle identity")
    require(set(compiler) == {"status", "method", "cycle", "bank", "population", "sampling", "readout", "owner_terminal", "training_authorized"}
            and compiler["status"] == "ROOT_ACCEPTS_CLOSED_CAPPED_CYCLE_POPULATION_COMPILE_V1"
            and compiler["bank"] == bank and compiler["cycle"] == row["cycle"]
            and canonical(compiler["population"]) == canonical(c["population"])
            and canonical(compiler["sampling"]) == canonical(c["sampling"])
            and compiler["training_authorized"] is False, "trusted actual single-bank compiler Root9")
    pop, sampling = compiler["population"], compiler["sampling"]
    require(plan["schema"] == "FIXED_SELFPLAY_UNION_PLAN_V1" and plan["branches"] == [{"kind": "bank", **bank}]
            and plan["copies"] == [] and pop["plan"] == row["compiler_plan"]
            and plan["expected_rows"] == pop["rows"] and plan["expected_sources"] == pop["sources"], "single bank/no aliases/order source")
    games, n = metadata_games(candidate, caps["games_per_bank"], caps["rows_per_bank"])
    require(len(games) == row["expected_games"] == pop["sources"] and n == pop["rows"], "actual game/row totals")
    membership = [[candidate["generation"], g["source_sha256"], g["rows"]] for g in games]
    require(digest(membership) == pop["membership_sha256"], "candidate slot order/rows do not match compiler membership")
    batch = integer(c["batch_size"], 1)
    require(batch == 256, "fixed F12/Q50 nominal batch")
    require(integer(c["maximum_presentations"], 1) == 1 and sampling["schema"] == "STORED_ROW_EXACT_ONCE_PASS_V2"
            and integer(sampling["expected_rows"], 1) == n and integer(sampling["presentations_per_row"], 1) == 1
            and sampling["mirror_shares_quota"] is True and sampling["partial_batch"] == "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING"
            and sampling["lineage_uuid"] == c["run_uuid"], "one physical-row pass; mirror shares quota")
    technical = reader.read(row["learner_technical"])
    actual = {"available": False, "reason": "UNSUPPORTED_LEARNER_TECHNICAL_SCALAR_SCHEMA",
              "observed_status": technical.get("status")}
    if technical.get("status") == "ROOT_ACCEPTS_CLOSED_CAPPED_CYCLE_TRAINING_TECHNICAL_V1":
        updates = (n + batch - 1) // batch
        require(technical["contract"] == row["contract"] and technical["method"] == compiler["method"]
                and integer(technical["supported_exit_code"]) == 0 and integer(technical["actual_views"], 1) == n
                and integer(technical["actual_updates"], 1) == updates and integer(technical["maximum_observed_presentations"], 1) == 1
                and integer(technical["last_batch_rows"], 1) == (n - 1) % batch + 1, "actual complete technical dose differs")
        actual = {"available": True, "actual_views": n, "actual_updates": updates,
                  "maximum_observed_presentations": 1, "last_batch_rows": (n - 1) % batch + 1}
    diversity = {"available": False, "reason": "UNSUPPORTED_DECLARED_ORDER"}
    if sampling["order"] == "python_random_mt19937_shuffle_v1":
        order = permutation_for(n, integer(sampling["order_seed"]))
        diversity = {"available": False, "reason": "RECONSTRUCTED_PERMUTATION_SHA_MISMATCH"}
        reconstructed = digest(order)
        diversity.update(declared_permutation_sha256=sampling["permutation_sha256"], reconstructed_permutation_sha256=reconstructed)
        if reconstructed == sampling["permutation_sha256"]:
            ends, total = [], 0
            for game in games:
                total += game["rows"]
                ends.append(total)
            diversity = {**batch_statistics(ends, order, batch),
                         "declared_permutation_sha256": sampling["permutation_sha256"],
                         "reconstructed_permutation_sha256": reconstructed}
    return {"study": row["study"], "cycle": row["cycle"], "bank_root": row["bank_root"],
            "candidate": row["candidate"], "compiler_root": row["compiler_root"], "contract": row["contract"],
            "learner_technical": row["learner_technical"], "games": len(games), "rows": n,
            "authenticated_membership_sha256": pop["membership_sha256"],
            "identity_namespace": {k: candidate[k] for k in ("run_uuid", "collection_uuid", "generation", "source_revision")},
            "role": "train", "physical_game_id_scope": "identity_namespace plus physical_game_id; never collapse across origins",
            "row_mass_basis": "one complete physical-stored-row pass, pre-mirror; not measured gradient mass",
            "mirror_shares_physical_row_quota": True, "actual_technical_dose": actual,
            "batch_diversity": diversity, **row_masses(games, n)}


def write_exclusive(path, result):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.suffix == ".json", "canonical output")
    raw = (canonical(result) + "\n").encode()
    require(len(raw) <= COMPACT, "summary cap")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--selection", required=True)
    args.add_argument("--selection-sha256", required=True)
    args.add_argument("--out", required=True)
    a = args.parse_args()
    selected_ref = reference({"path": a.selection, "sha256": a.selection_sha256})
    bootstrap = Reader([selected_ref], {"compact_bytes": COMPACT, "bank_two_pass_bytes": 2 * COMPACT}, 2 * COMPACT)
    selection = bootstrap.read(selected_ref)
    require(set(selection) == {"schema", "purpose", "limits", "banks"}
            and selection["schema"] == "ROOT_SELECTED_BANKED_GAME_EXPOSURE_AUDIT_INPUTS_V1"
            and selection["purpose"] == "STRUCTURAL_EXPOSURE_ONLY_NO_SCIENTIFIC_SELECTION", "explicit Root-selected input manifest")
    caps = selection["limits"]
    require(set(caps) == {"candidate_bytes", "compact_bytes", "bank_two_pass_bytes", "total_two_pass_bytes", "banks", "games_per_bank", "rows_per_bank"}, "exact finite IO caps")
    maxima = {"candidate_bytes": CANDIDATE, "compact_bytes": COMPACT, "bank_two_pass_bytes": 128 << 20,
              "total_two_pass_bytes": TOTAL, "banks": MAX_BANKS, "games_per_bank": MAX_GAMES, "rows_per_bank": MAX_ROWS}
    require(all(0 < integer(caps[k], 1) <= v for k, v in maxima.items()), "selected caps cannot expand source maxima")
    rows = selection["banks"]
    require(type(rows) is list and 0 < len(rows) <= caps["banks"], "explicit bank list cap")
    require(len({(r["study"], r["cycle"]) for r in rows}) == len(rows), "duplicate selected study/cycle")
    used = bootstrap.reserved
    summaries = []
    for row in rows:
        require(used < caps["total_two_pass_bytes"], "total IO cap")
        reader = Reader([row[k] for k in sorted(ROW_REFS)], caps, caps["total_two_pass_bytes"] - used)
        summaries.append(one_bank(row, reader, caps))
        used += reader.finish()  # Drop the candidate and all per-bank decoded metadata before next bank.
    bootstrap.finish()  # Final input-selection bracket, counted in its pre-reserved two passes.
    result = {"schema": "BANKED_GAME_STRUCTURAL_EXPOSURE_AUDIT_V1", "input_selection": selected_ref,
              "banks": summaries, "actual_reserved_and_read_bytes": used,
              "limits": "No independence, gradient ESS/covariance, causal, optimum, strength, outcome-score or label-independence claim. No board deduplication; cross-origin same-game equivalence UNKNOWN.",
              "input_scope": "Only explicit accepted bank/compiler/contract/technical JSON descriptors. All referenced shards/proofs/models/books/rosters/ledgers/journals/manifests/order arrays remain unopened."}
    print(canonical({"summary": write_exclusive(a.out, result)}))


if __name__ == "__main__":
    main()
