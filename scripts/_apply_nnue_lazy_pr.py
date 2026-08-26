#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{path}: expected one match, got {n}: {old[:80]!r}")
    write(path, text.replace(old, new, 1))


def replace_function(path: str, marker: str, new_func: str) -> None:
    text = read(path)
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"{path}: function marker not found: {marker!r}")
    brace = text.find("{", start)
    if brace < 0:
        raise RuntimeError(f"{path}: opening brace not found")
    depth = 0
    end = None
    for i in range(brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise RuntimeError(f"{path}: closing brace not found")
    write(path, text[:start] + new_func.rstrip() + text[end:])


# ---------------------------------------------------------------------------
# 1. Cheap position-intrinsic PSQT component from an already-built NNUE state.
# ---------------------------------------------------------------------------
state_h = "chess_anti_engine/nnue/_nnue_state.h"
replace_once(
    state_h,
    "static int cae_nnue_state_evaluate(\n",
    r'''/* The cheap tier used by S1 lazy evaluation. This is EXACTLY the PSQT
 * term the full evaluator adds, in the same internal-unit scale, but does not
 * build the transformed feature vector and does not run any FC layer. It is a
 * position-intrinsic fact and therefore safe to store in the canonical DAG. */
static int cae_nnue_state_psqt_value(
    const CaeNnueWeights *w, const CaeNnueState *state, int32_t *out_value)
{
    if (!w) return CAE_VALUE_ERR_NOT_LOADED;
    if (state->pos.in_check) return CAE_VALUE_ERR_IN_CHECK;
    const int bucket = cae_nnue_bucket(&state->pos);
    const int rc = cae_nnue_check_bucket(w, bucket);
    if (rc != CAE_VALUE_OK) return rc;
    const int p0 = state->pos.side_to_move;
    const int p1 = p0 ^ 1;
    int32_t psqt = state->acc.psqt[p0][bucket] - state->acc.psqt[p1][bucket];
    psqt /= 2;  /* same truncation point as cae_nnue_transform() */
    *out_value = psqt / CAE_NNUE_OUTPUT_SCALE;
    return CAE_VALUE_OK;
}

static int cae_nnue_state_evaluate(
''',
)


# ---------------------------------------------------------------------------
# 2. DAG payload: NONE / PSQT_ONLY / FULL, monotone upgrade, window accessor.
# ---------------------------------------------------------------------------
store_h = "chess_anti_engine/nnue/_nnue_dag_store.h"
replace_once(
    store_h,
    "typedef struct CaeNnueDagHandle {\n",
    r'''/* S1 evaluation tiers. They describe how much of a position-intrinsic
 * static evaluation has been completed; they are NOT alpha-beta outcomes.
 * Upgrade is monotone NONE -> (for non-check nodes) PSQT_ONLY -> FULL. */
#define CAE_DAG_VALUE_NONE       0u
#define CAE_DAG_VALUE_PSQT_ONLY  1u
#define CAE_DAG_VALUE_FULL       2u

/* Result kinds from the one window-aware accessor. LOWER/UPPER are ephemeral
 * search bounds derived from the configured PSQT error envelope and are never
 * stored in the node. */
#define CAE_DAG_WINDOW_EXACT 0
#define CAE_DAG_WINDOW_LOWER 1
#define CAE_DAG_WINDOW_UPPER 2

#define CAE_DAG_LAZY_DEFAULT_ENABLED 0
#define CAE_DAG_LAZY_DEFAULT_MARGIN  0

typedef struct CaeNnueDagHandle {
''',
)
replace_once(
    store_h,
    """    CaeNnueState *states;\n    int32_t *values;\n    uint8_t *value_valid;\n""",
    """    CaeNnueState *states;\n    /* FULL raw NNUE value. `value_valid` remains the compatibility spelling\n     * for \"FULL exists\"; PSQT_ONLY deliberately leaves it false. */\n    int32_t *values;\n    int32_t *psqt_values;\n    uint8_t *value_valid;\n    uint8_t *value_tier;\n""",
)
replace_once(
    store_h,
    """    int32_t payload_cap;\n\n    uint64_t state_inits;\n    uint64_t state_makes;\n    uint64_t nnue_evals;\n    uint64_t node_reuses;\n""",
    """    int32_t payload_cap;\n\n    /* Snapshotted at DAG-context construction. A live store is never retuned:\n     * changing the globals configures the NEXT arm handle only. */\n    int lazy_enabled;\n    int32_t lazy_margin;\n\n    uint64_t state_inits;\n    uint64_t state_makes;\n    uint64_t nnue_evals;       /* FULL propagations actually performed */\n    uint64_t node_reuses;\n    uint64_t full_nodes;       /* current FULL tier; == nnue_evals after reset */\n    uint64_t psqt_only_nodes;  /* current PSQT_ONLY tier */\n    uint64_t no_value_nodes;   /* current in-check/NONE tier */\n    uint64_t psqt_probes;      /* window probes while still PSQT_ONLY */\n    uint64_t lazy_lower_bounds;\n    uint64_t lazy_upper_bounds;\n    uint64_t lazy_upgrades;\n""",
)
replace_once(
    store_h,
    """    free(h->states);\n    free(h->values);\n    free(h->value_valid);\n    free(h->quiet_bits);\n""",
    """    free(h->states);\n    free(h->values);\n    free(h->psqt_values);\n    free(h->value_valid);\n    free(h->value_tier);\n    free(h->quiet_bits);\n""",
)
replace_once(
    store_h,
    """    h->states = NULL;\n    h->values = NULL;\n    h->value_valid = NULL;\n    h->quiet_bits = NULL;\n""",
    """    h->states = NULL;\n    h->values = NULL;\n    h->psqt_values = NULL;\n    h->value_valid = NULL;\n    h->value_tier = NULL;\n    h->quiet_bits = NULL;\n""",
)
replace_once(
    store_h,
    """    int32_t *new_values = (int32_t *)malloc((size_t)new_cap * sizeof(*new_values));\n    uint8_t *new_valid = (uint8_t *)calloc((size_t)new_cap, sizeof(*new_valid));\n    uint8_t *new_cert = (uint8_t *)calloc((size_t)new_cap, sizeof(*new_cert));\n    if (!new_values || !new_valid || !new_cert) {\n        free(new_states);\n        free(new_values);\n        free(new_valid);\n        free(new_cert);\n        return -1;\n    }\n""",
    """    int32_t *new_values = (int32_t *)malloc((size_t)new_cap * sizeof(*new_values));\n    int32_t *new_psqt = (int32_t *)malloc((size_t)new_cap * sizeof(*new_psqt));\n    uint8_t *new_valid = (uint8_t *)calloc((size_t)new_cap, sizeof(*new_valid));\n    uint8_t *new_tier = (uint8_t *)calloc((size_t)new_cap, sizeof(*new_tier));\n    uint8_t *new_cert = (uint8_t *)calloc((size_t)new_cap, sizeof(*new_cert));\n    if (!new_values || !new_psqt || !new_valid || !new_tier || !new_cert) {\n        free(new_states);\n        free(new_values);\n        free(new_psqt);\n        free(new_valid);\n        free(new_tier);\n        free(new_cert);\n        return -1;\n    }\n""",
)
replace_once(
    store_h,
    """        memcpy(new_states, h->states, (size_t)h->payload_cap * sizeof(*new_states));\n        memcpy(new_values, h->values, (size_t)h->payload_cap * sizeof(*new_values));\n        memcpy(new_valid, h->value_valid, (size_t)h->payload_cap * sizeof(*new_valid));\n        memcpy(new_cert, h->quiet_bits, (size_t)h->payload_cap * sizeof(*new_cert));\n""",
    """        memcpy(new_states, h->states, (size_t)h->payload_cap * sizeof(*new_states));\n        memcpy(new_values, h->values, (size_t)h->payload_cap * sizeof(*new_values));\n        memcpy(new_psqt, h->psqt_values, (size_t)h->payload_cap * sizeof(*new_psqt));\n        memcpy(new_valid, h->value_valid, (size_t)h->payload_cap * sizeof(*new_valid));\n        memcpy(new_tier, h->value_tier, (size_t)h->payload_cap * sizeof(*new_tier));\n        memcpy(new_cert, h->quiet_bits, (size_t)h->payload_cap * sizeof(*new_cert));\n""",
)
replace_once(
    store_h,
    """    free(h->states);\n    free(h->values);\n    free(h->value_valid);\n    free(h->quiet_bits);\n    h->states = new_states;\n    h->values = new_values;\n    h->value_valid = new_valid;\n    h->quiet_bits = new_cert;\n""",
    """    free(h->states);\n    free(h->values);\n    free(h->psqt_values);\n    free(h->value_valid);\n    free(h->value_tier);\n    free(h->quiet_bits);\n    h->states = new_states;\n    h->values = new_values;\n    h->psqt_values = new_psqt;\n    h->value_valid = new_valid;\n    h->value_tier = new_tier;\n    h->quiet_bits = new_cert;\n""",
)
replace_once(
    store_h,
    """    return 0;\n}\n\nstatic void cae_nnue_dag_store_reset(CaeNnueDagHandle *h) {\n""",
    """    return 0;\n}\n\n/* Configure a freshly-created store. Refuse a live graph: lazy-eval settings are\n * part of a context snapshot, not a mutable search knob. */\nstatic int cae_nnue_dag_configure_lazy(\n    CaeNnueDagHandle *h, int enabled, int32_t margin)\n{\n    if (!h || (enabled != 0 && enabled != 1) || margin < 0) return -1;\n    if (h->dag.node_count != 0) return -1;\n    h->lazy_enabled = enabled;\n    h->lazy_margin = margin;\n    return 0;\n}\n\nstatic void cae_nnue_dag_store_reset(CaeNnueDagHandle *h) {\n""",
)
replace_once(
    store_h,
    """        memset(h->value_valid, 0, (size_t)h->payload_cap * sizeof(*h->value_valid));\n        memset(h->quiet_bits, 0, (size_t)h->payload_cap * sizeof(*h->quiet_bits));\n    }\n    h->state_inits = h->state_makes = h->nnue_evals = h->node_reuses = 0;\n""",
    """        memset(h->value_valid, 0, (size_t)h->payload_cap * sizeof(*h->value_valid));\n        memset(h->value_tier, 0, (size_t)h->payload_cap * sizeof(*h->value_tier));\n        memset(h->quiet_bits, 0, (size_t)h->payload_cap * sizeof(*h->quiet_bits));\n    }\n    h->state_inits = h->state_makes = h->nnue_evals = h->node_reuses = 0;\n    h->full_nodes = h->psqt_only_nodes = h->no_value_nodes = 0;\n    h->psqt_probes = h->lazy_lower_bounds = h->lazy_upper_bounds = 0;\n    h->lazy_upgrades = 0;\n""",
)
replace_once(
    store_h,
    """        sizeof(CaeNnueState) + sizeof(int32_t) + sizeof(uint8_t) + sizeof(uint8_t));\n""",
    """        sizeof(CaeNnueState) + 2 * sizeof(int32_t)\n        + 2 * sizeof(uint8_t) + sizeof(uint8_t));\n""",
)

replace_function(
    store_h,
    "static int32_t cae_nnue_dag_publish_new(\n",
    r'''static int32_t cae_nnue_dag_publish_new(
    CaeNnueDagHandle *h,
    const CaeDagPosition *position,
    const CaeNnueState *state,
    int32_t psqt_value,
    int32_t value,
    uint8_t tier)
{
    if (cae_nnue_dag_grow_payload(h, h->dag.node_count + 1) != 0)
        return CAE_DAG_NO_NODE;
    int32_t node_id = cae_position_dag_insert_position(&h->dag, position);
    if (node_id == CAE_DAG_NO_NODE) return CAE_DAG_NO_NODE;
    h->states[node_id] = *state;
    h->psqt_values[node_id] = psqt_value;
    h->values[node_id] = value;
    h->value_tier[node_id] = tier;
    h->value_valid[node_id] = (uint8_t)(tier == CAE_DAG_VALUE_FULL);
    h->quiet_bits[node_id] = 0;
    if (tier == CAE_DAG_VALUE_FULL) h->full_nodes++;
    else if (tier == CAE_DAG_VALUE_PSQT_ONLY) h->psqt_only_nodes++;
    else h->no_value_nodes++;
    return node_id;
}''',
)

replace_function(
    store_h,
    "static int cae_nnue_dag_intern_position(\n",
    r'''static int cae_nnue_dag_intern_position(
    CaeNnueDagHandle *h, const CBoard *board, int32_t *out_node, int *out_created)
{
    CaeDagPosition position;
    cae_dag_position_from_cboard(board, &position);
    int32_t existing = cae_position_dag_find_position(&h->dag, &position);
    if (existing != CAE_DAG_NO_NODE) {
        h->node_reuses++;
        *out_node = existing;
        *out_created = 0;
        return CAE_VALUE_OK;
    }

    CaeNnueState state;
    int status = cae_nnue_state_init(h->weights, board, &state);
    if (status != CAE_VALUE_OK) return status;

    int32_t psqt_value = 0;
    int32_t value = 0;
    uint8_t tier = CAE_DAG_VALUE_NONE;
    if (!state.pos.in_check) {
        if (h->lazy_enabled) {
            status = cae_nnue_state_psqt_value(h->weights, &state, &psqt_value);
            if (status != CAE_VALUE_OK) return status;
            tier = CAE_DAG_VALUE_PSQT_ONLY;
        } else {
            status = cae_nnue_state_evaluate(h->weights, &state, &value);
            if (status != CAE_VALUE_OK) return status;
            tier = CAE_DAG_VALUE_FULL;
        }
    }

    int32_t node_id = cae_nnue_dag_publish_new(
        h, &position, &state, psqt_value, value, tier);
    if (node_id == CAE_DAG_NO_NODE) return CAE_NNUE_DAG_ERR_NO_MEMORY;
    h->state_inits++;
    if (tier == CAE_DAG_VALUE_FULL) h->nnue_evals++;
    *out_node = node_id;
    *out_created = 1;
    return CAE_VALUE_OK;
}''',
)

replace_function(
    store_h,
    "static int cae_nnue_dag_intern_child(\n",
    r'''static int cae_nnue_dag_intern_child(
    CaeNnueDagHandle *h, int32_t parent_id, int action, const CBoard *child_board,
    int32_t *out_node, int *out_created)
{
    CaeDagPosition child_pos;
    cae_dag_position_from_cboard(child_board, &child_pos);

    int32_t existing = cae_position_dag_find_position(&h->dag, &child_pos);
    if (existing != CAE_DAG_NO_NODE) {
        int link_rc = cae_position_dag_link(&h->dag, parent_id, action, existing);
        if (link_rc < 0)
            return link_rc == -1 ? CAE_NNUE_DAG_ERR_NO_MEMORY : CAE_NNUE_DAG_ERR_LINK;
        h->node_reuses++;
        *out_node = existing;
        *out_created = 0;
        return CAE_VALUE_OK;
    }

    CaeNnueState state;
    int status = cae_nnue_state_make(
        h->weights, &h->states[parent_id], child_board, &state);
    if (status != CAE_VALUE_OK) return status;

    int32_t psqt_value = 0;
    int32_t value = 0;
    uint8_t tier = CAE_DAG_VALUE_NONE;
    if (!state.pos.in_check) {
        if (h->lazy_enabled) {
            status = cae_nnue_state_psqt_value(h->weights, &state, &psqt_value);
            if (status != CAE_VALUE_OK) return status;
            tier = CAE_DAG_VALUE_PSQT_ONLY;
        } else {
            status = cae_nnue_state_evaluate(h->weights, &state, &value);
            if (status != CAE_VALUE_OK) return status;
            tier = CAE_DAG_VALUE_FULL;
        }
    }

    if (cae_position_dag_reserve_edge(&h->dag) != 0)
        return CAE_NNUE_DAG_ERR_NO_MEMORY;

    int32_t node_id = cae_nnue_dag_publish_new(
        h, &child_pos, &state, psqt_value, value, tier);
    if (node_id == CAE_DAG_NO_NODE) return CAE_NNUE_DAG_ERR_NO_MEMORY;
    int link_rc = cae_position_dag_link(&h->dag, parent_id, action, node_id);
    if (link_rc != 1)
        return link_rc == -1 ? CAE_NNUE_DAG_ERR_NO_MEMORY : CAE_NNUE_DAG_ERR_LINK;

    h->state_makes++;
    if (tier == CAE_DAG_VALUE_FULL) h->nnue_evals++;
    *out_node = node_id;
    *out_created = 1;
    return CAE_VALUE_OK;
}''',
)

replace_once(
    store_h,
    "\n#endif /* CAE_NNUE_DAG_STORE_H */\n",
    r'''
/* Upgrade one PSQT_ONLY node exactly once. Bounds are never written here: the
 * only persistent transition is the position-intrinsic FULL value. */
static int cae_nnue_dag_force_full(
    CaeNnueDagHandle *h, int32_t node_id, int32_t *out_value, int *out_did_eval)
{
    if (!h || node_id < 0 || node_id >= h->dag.node_count)
        return CAE_VALUE_ERR_NOT_LOADED;
    *out_did_eval = 0;
    const uint8_t tier = h->value_tier[node_id];
    if (tier == CAE_DAG_VALUE_NONE) return CAE_VALUE_ERR_IN_CHECK;
    if (tier == CAE_DAG_VALUE_FULL) {
        *out_value = h->values[node_id];
        return CAE_VALUE_OK;
    }
    if (tier != CAE_DAG_VALUE_PSQT_ONLY) return CAE_VALUE_ERR_BAD_POS;

    int32_t value = 0;
    const int rc = cae_nnue_state_evaluate(h->weights, &h->states[node_id], &value);
    if (rc != CAE_VALUE_OK) return rc;
    h->values[node_id] = value;
    h->value_valid[node_id] = 1;
    h->value_tier[node_id] = CAE_DAG_VALUE_FULL;
    h->psqt_only_nodes--;
    h->full_nodes++;
    h->nnue_evals++;
    h->lazy_upgrades++;
    *out_did_eval = 1;
    *out_value = value;
    return CAE_VALUE_OK;
}

static inline int32_t cae_nnue_dag_clamp_i64(int64_t v, int32_t clamp_abs)
{
    if (v > (int64_t)clamp_abs) return clamp_abs;
    if (v < -(int64_t)clamp_abs) return -clamp_abs;
    return (int32_t)v;
}

/* The ONE search accessor for a DAG static value. The configured margin is an
 * empirical envelope around FULL-PSQT; LOWER/UPPER exist only for this visit and
 * only when they clear the caller's window. If neither bound decides the visit,
 * the node upgrades to FULL and returns EXACT. */
static int cae_nnue_dag_value_for_window(
    CaeNnueDagHandle *h, int32_t node_id,
    int32_t alpha, int32_t beta, int32_t clamp_abs,
    int32_t *out_value, int *out_kind, int *out_did_eval)
{
    if (!h || node_id < 0 || node_id >= h->dag.node_count || clamp_abs <= 0)
        return CAE_VALUE_ERR_NOT_LOADED;
    *out_did_eval = 0;
    const uint8_t tier = h->value_tier[node_id];
    if (tier == CAE_DAG_VALUE_NONE) return CAE_VALUE_ERR_IN_CHECK;
    if (tier == CAE_DAG_VALUE_FULL) {
        *out_kind = CAE_DAG_WINDOW_EXACT;
        *out_value = cae_nnue_dag_clamp_i64(h->values[node_id], clamp_abs);
        return CAE_VALUE_OK;
    }
    if (tier != CAE_DAG_VALUE_PSQT_ONLY) return CAE_VALUE_ERR_BAD_POS;

    h->psqt_probes++;
    const int64_t p = (int64_t)h->psqt_values[node_id];
    const int64_t m = (int64_t)h->lazy_margin;
    const int32_t lower = cae_nnue_dag_clamp_i64(p - m, clamp_abs);
    const int32_t upper = cae_nnue_dag_clamp_i64(p + m, clamp_abs);
    if (lower >= beta) {
        h->lazy_lower_bounds++;
        *out_kind = CAE_DAG_WINDOW_LOWER;
        *out_value = lower;
        return CAE_VALUE_OK;
    }
    if (upper <= alpha) {
        h->lazy_upper_bounds++;
        *out_kind = CAE_DAG_WINDOW_UPPER;
        *out_value = upper;
        return CAE_VALUE_OK;
    }

    int32_t raw = 0;
    const int rc = cae_nnue_dag_force_full(h, node_id, &raw, out_did_eval);
    if (rc != CAE_VALUE_OK) return rc;
    *out_kind = CAE_DAG_WINDOW_EXACT;
    *out_value = cae_nnue_dag_clamp_i64(raw, clamp_abs);
    return CAE_VALUE_OK;
}

#endif /* CAE_NNUE_DAG_STORE_H */
''',
)


# ---------------------------------------------------------------------------
# 3. FastQ: every static read goes through the tier/window accessor.
# ---------------------------------------------------------------------------
fastq_h = "chess_anti_engine/nnue/_fastq_search.h"
replace_function(
    fastq_h,
    "static int cae_fastq_intern_child(\n",
    r'''static int cae_fastq_intern_child(
    CaeFastqCtx *q, int32_t parent_id, int action, const CBoard *child_board,
    int32_t *out_node)
{
    int created = 0;
    const uint64_t evals_before = q->store->nnue_evals;
    const int status = cae_nnue_dag_intern_child(
        q->store, parent_id, action, child_board, out_node, &created);
    if (status != CAE_VALUE_OK) return status;
    q->stats->nnue_evals += q->store->nnue_evals - evals_before;
    if (created) {
        q->stats->nodes_created++;
        if (cboard_in_check(child_board)) q->stats->nodes_created_in_check++;
    } else if (*out_node < q->dag_watermark) {
        q->stats->hits_cross_call++;
    } else {
        q->stats->hits_within_call++;
    }
    return CAE_VALUE_OK;
}''',
)

# Insert helpers before the real definition (the forward declaration ends with ';').
replace_once(
    fastq_h,
    """static int cae_fastq_node(\n    CaeFastqCtx *q, const CBoard *b, int32_t node_id, int ply,\n    int32_t alpha, int32_t beta, int32_t *out_value)\n{\n""",
    r'''static int cae_fastq_static_for_window(
    CaeFastqCtx *q, int32_t node_id, int32_t alpha, int32_t beta,
    int32_t *out_value, int *out_kind)
{
    const uint64_t before = q->store->nnue_evals;
    int did_eval = 0;
    const int rc = cae_nnue_dag_value_for_window(
        q->store, node_id, alpha, beta, CAE_RESOLVER_EVAL_CLAMP,
        out_value, out_kind, &did_eval);
    (void)did_eval;
    q->stats->nnue_evals += q->store->nnue_evals - before;
    return rc;
}

static int cae_fastq_force_full_static(
    CaeFastqCtx *q, int32_t node_id, int32_t *out_value)
{
    const uint64_t before = q->store->nnue_evals;
    int did_eval = 0;
    int32_t raw = 0;
    const int rc = cae_nnue_dag_force_full(q->store, node_id, &raw, &did_eval);
    (void)did_eval;
    q->stats->nnue_evals += q->store->nnue_evals - before;
    if (rc == CAE_VALUE_OK)
        *out_value = cae_resolver_clamp(raw);
    return rc;
}

static int cae_fastq_node(
    CaeFastqCtx *q, const CBoard *b, int32_t node_id, int ply,
    int32_t alpha, int32_t beta, int32_t *out_value)
{
''',
)

# Replace the non-check stand-pat block and tail of the tactical loop as one unit.
replace_once(
    fastq_h,
    r'''    if (!q->store->value_valid[node_id]) {
        /* Not in check, so the store must hold a static value; if it does not,
         * the node was published by something that disagrees with this search
         * about what a node is. Fail loudly rather than inventing a number. */
        status = CAE_VALUE_ERR_BAD_POS;
        goto done;
    }
    /* ⚑ CLAMPED ON READ, THE SAME WAY cae_qsearch_node CLAMPS ITS STAND-PAT.
     * The DAG stores the RAW NNUE value — that is the store's contract and the
     * qsearch-dag arm depends on it — so the clamp belongs at every reader, and
     * a reader that skips it is a reader whose search values can leave the
     * evaluation range. Measured: with a high-magnitude synthetic pack, 147 of
     * 444 non-check corpus rows evaluate past CAE_RESOLVER_EVAL_CLAMP and reach
     * 89044, which is inside the mate band the §8 harness classifies on. Not
     * reachable with the production net (max |v| = 4546), which is exactly why
     * it needs a test rather than an argument. */
    const int32_t stand_pat = cae_resolver_clamp(q->store->values[node_id]);

    if (stand_pat >= beta) {
        q->stats->stand_pat_cutoffs++;
        *out_value = stand_pat;
        goto done;
    }
    if (alpha < stand_pat) alpha = stand_pat;
    best = stand_pat;

    if (ply >= q->cfg.max_qply) {
        *out_value = stand_pat;
        goto done;
    }

    /* The certificate. A quiet node has nothing to search by construction. */
    if (cae_fastq_is_quiet(cae_fastq_node_certificate(q, node_id, b))) {
        q->stats->quiet_returns++;
        *out_value = stand_pat;
        goto done;
    }

    {
        CaeFastqMove tactical[CAE_FASTQ_MAX_MOVES];
        const int n_tactical = cae_fastq_tactical_moves(b, moves, n_moves, tactical);
        for (int i = 0; i < n_tactical; i++) {
            const CaeFastqMove *mv = &tactical[i];
''',
    r'''    int32_t stand_pat = 0;
    int stand_kind = CAE_DAG_WINDOW_EXACT;
    status = cae_fastq_static_for_window(
        q, node_id, alpha, beta, &stand_pat, &stand_kind);
    if (status != CAE_VALUE_OK) goto done;

    /* A PSQT lower bound is usable only because it ALREADY clears beta. It is a
     * cutoff result, not a static value, and is never stored in the node. */
    if (stand_kind == CAE_DAG_WINDOW_LOWER) {
        q->stats->stand_pat_cutoffs++;
        *out_value = stand_pat;
        goto done;
    }
    if (stand_kind == CAE_DAG_WINDOW_EXACT && stand_pat >= beta) {
        q->stats->stand_pat_cutoffs++;
        *out_value = stand_pat;
        goto done;
    }
    if (stand_kind == CAE_DAG_WINDOW_EXACT && alpha < stand_pat) alpha = stand_pat;

    /* A max-qply or quiet-certificate return is a STATIC VALUE, not a cutoff;
     * an UPPER bound is therefore insufficient and must upgrade first. */
    if (ply >= q->cfg.max_qply) {
        if (stand_kind == CAE_DAG_WINDOW_UPPER)
            status = cae_fastq_force_full_static(q, node_id, &stand_pat);
        if (status == CAE_VALUE_OK) *out_value = stand_pat;
        goto done;
    }

    if (cae_fastq_is_quiet(cae_fastq_node_certificate(q, node_id, b))) {
        q->stats->quiet_returns++;
        if (stand_kind == CAE_DAG_WINDOW_UPPER)
            status = cae_fastq_force_full_static(q, node_id, &stand_pat);
        if (status == CAE_VALUE_OK) *out_value = stand_pat;
        goto done;
    }

    best = (stand_kind == CAE_DAG_WINDOW_EXACT) ? stand_pat : -CAE_FASTQ_INF;
    int searched_tactical = 0;
    int move_cutoff = 0;
    {
        CaeFastqMove tactical[CAE_FASTQ_MAX_MOVES];
        const int n_tactical = cae_fastq_tactical_moves(b, moves, n_moves, tactical);
        for (int i = 0; i < n_tactical; i++) {
            const CaeFastqMove *mv = &tactical[i];
''',
)
replace_once(
    fastq_h,
    r'''            if (value > best) best = value;
            if (best > alpha) alpha = best;
            if (alpha >= beta) { q->stats->move_cutoffs++; break; }
        }
        *out_value = best;
    }
''',
    r'''            searched_tactical = 1;
            if (value > best) best = value;
            if (best > alpha) alpha = best;
            if (alpha >= beta) {
                q->stats->move_cutoffs++;
                move_cutoff = 1;
                break;
            }
        }
    }
    if (move_cutoff) {
        *out_value = best;
    } else if (stand_kind == CAE_DAG_WINDOW_UPPER) {
        int32_t exact_static = 0;
        status = cae_fastq_force_full_static(q, node_id, &exact_static);
        if (status == CAE_VALUE_OK)
            *out_value = searched_tactical && best > exact_static ? best : exact_static;
    } else {
        *out_value = best;
    }
''',
)
# Path-ceiling non-check fallback must also be exact.
replace_once(
    fastq_h,
    r'''        *out_value = (!in_check && q->store->value_valid[node_id])
                         ? cae_resolver_clamp(q->store->values[node_id])
                         : cae_resolver_clamp(beta);
        return CAE_VALUE_OK;
''',
    r'''        if (!in_check) {
            int32_t exact_static = 0;
            const int rc = cae_fastq_force_full_static(q, node_id, &exact_static);
            if (rc != CAE_VALUE_OK) return rc;
            *out_value = exact_static;
        } else {
            *out_value = cae_resolver_clamp(beta);
        }
        return CAE_VALUE_OK;
''',
)
# Delta pruning can use an upper bound safely; rename the comment's premise.
replace_once(
    fastq_h,
    """            if ((int64_t)stand_pat + mv->victim + q->cfg.delta_margin\n                <= (int64_t)alpha) {\n""",
    """            /* If stand_pat is an UPPER PSQT bound this is still sound:\n             * exact_static <= stand_pat, so clearing alpha with the upper bound\n             * implies the exact static value clears it too. */\n            if ((int64_t)stand_pat + mv->victim + q->cfg.delta_margin\n                <= (int64_t)alpha) {\n""",
)


# ---------------------------------------------------------------------------
# 4. Qsearch DAG: same search body, tier-aware stand pat and exact final returns.
# ---------------------------------------------------------------------------
arm_h = "chess_anti_engine/nnue/_arm_providers.h"
replace_once(
    arm_h,
    """static int g_arm_dag_node_cap         = CAE_QSEARCH_DEFAULT_DAG_NODE_CAP;\n\n/* FastQ's §6 knobs.\n""",
    """static int g_arm_dag_node_cap         = CAE_QSEARCH_DEFAULT_DAG_NODE_CAP;\nstatic int g_dag_lazy_enabled         = CAE_DAG_LAZY_DEFAULT_ENABLED;\nstatic int g_dag_lazy_margin          = CAE_DAG_LAZY_DEFAULT_MARGIN;\n\ntypedef struct CaeDagLazyConfig {\n    int enabled;\n    int margin;\n} CaeDagLazyConfig;\n\nstatic void cae_dag_lazy_get_config(CaeDagLazyConfig *out) {\n    pthread_mutex_lock(&g_arm_config_lock);\n    out->enabled = g_dag_lazy_enabled;\n    out->margin = g_dag_lazy_margin;\n    pthread_mutex_unlock(&g_arm_config_lock);\n}\n\nstatic int cae_dag_lazy_set_config(int enabled, int margin, char *err, size_t errlen) {\n    if (enabled != 0 && enabled != 1) {\n        cae_nnue_err(err, errlen, \"lazy enabled must be 0 or 1, got %d\", enabled);\n        return -1;\n    }\n    if (margin < 0) {\n        cae_nnue_err(err, errlen, \"lazy margin must be >= 0, got %d\", margin);\n        return -1;\n    }\n    pthread_mutex_lock(&g_arm_config_lock);\n    g_dag_lazy_enabled = enabled;\n    g_dag_lazy_margin = margin;\n    pthread_mutex_unlock(&g_arm_config_lock);\n    return 0;\n}\n\n/* FastQ's §6 knobs.\n""",
)

replace_function(
    arm_h,
    "static void cae_qsearch_note_intern(\n",
    r'''static void cae_qsearch_note_intern(
    CaeQsearchCtx *q, int32_t node_id, int created, uint64_t evals_before)
{
    q->stats->nnue_evals += q->store->nnue_evals - evals_before;
    if (created) {
        q->stats->dag_nodes_interned++;
        return;
    }
    if (node_id < q->dag_watermark) q->stats->dag_hits_cross_call++;
    else q->stats->dag_hits_within_call++;
}''',
)
# Both intern call sites capture the store counter before construction.
replace_once(
    arm_h,
    r'''        int created = 0;
        int rc = cae_nnue_dag_intern_child(
            q->store, parent_ref.node_id, action, b, &ref.node_id, &created);
        if (rc != CAE_VALUE_OK) return rc;
        cae_qsearch_note_intern(q, ref.node_id, created);
''',
    r'''        int created = 0;
        const uint64_t evals_before = q->store->nnue_evals;
        int rc = cae_nnue_dag_intern_child(
            q->store, parent_ref.node_id, action, b, &ref.node_id, &created);
        if (rc != CAE_VALUE_OK) return rc;
        cae_qsearch_note_intern(q, ref.node_id, created, evals_before);
''',
)
replace_once(
    arm_h,
    r'''        int created = 0;
        int rc = cae_nnue_dag_intern_position(q->store, board, &ref.node_id, &created);
        if (rc != CAE_VALUE_OK) return rc;
        cae_qsearch_note_intern(q, ref.node_id, created);
''',
    r'''        int created = 0;
        const uint64_t evals_before = q->store->nnue_evals;
        int rc = cae_nnue_dag_intern_position(q->store, board, &ref.node_id, &created);
        if (rc != CAE_VALUE_OK) return rc;
        cae_qsearch_note_intern(q, ref.node_id, created, evals_before);
''',
)

replace_function(
    arm_h,
    "static int cae_qsearch_stand_pat(\n",
    r'''static int cae_qsearch_stand_pat(
    CaeQsearchCtx *q, const CBoard *b, const CaeQNodeRef *ref,
    int32_t alpha, int32_t beta, int32_t *out_value, int *out_kind)
{
    *out_kind = CAE_DAG_WINDOW_EXACT;
    if (q->substrate == CAE_QSUB_DAG) {
        if (ref->node_id < 0 || ref->node_id >= q->store->dag.node_count)
            return CAE_VALUE_ERR_NOT_LOADED;
        const uint64_t before = q->store->nnue_evals;
        int did_eval = 0;
        const int rc = cae_nnue_dag_value_for_window(
            q->store, ref->node_id, alpha, beta, CAE_RESOLVER_EVAL_CLAMP,
            out_value, out_kind, &did_eval);
        (void)did_eval;
        q->stats->nnue_evals += q->store->nnue_evals - before;
        return rc;
    }

    int status = ref->state
        ? cae_nnue_state_evaluate(q->arm->weights, ref->state, out_value)
        : cae_nnue_evaluate_cboard(q->arm->weights, b, out_value);
    if (status != CAE_VALUE_OK) return status;
    q->stats->nnue_evals++;
    *out_value = cae_resolver_clamp(*out_value);
    return CAE_VALUE_OK;
}''',
)
# Insert exact-upgrade helper before qsearch node implementation.
replace_once(
    arm_h,
    """/* Stand-pat quiescence over captures, promotions, and — for the first\n * qsearch_check_plies plies — checking moves. Fail-soft negamax. */\nstatic int cae_qsearch_node(\n""",
    r'''static int cae_qsearch_force_full_static(
    CaeQsearchCtx *q, int32_t node_id, int32_t *out_value)
{
    const uint64_t before = q->store->nnue_evals;
    int did_eval = 0;
    int32_t raw = 0;
    const int rc = cae_nnue_dag_force_full(q->store, node_id, &raw, &did_eval);
    (void)did_eval;
    q->stats->nnue_evals += q->store->nnue_evals - before;
    if (rc == CAE_VALUE_OK) *out_value = cae_resolver_clamp(raw);
    return rc;
}

/* Stand-pat quiescence over captures, promotions, and — for the first
 * qsearch_check_plies plies — checking moves. Fail-soft negamax. */
static int cae_qsearch_node(
''',
)

replace_function(
    arm_h,
    "static int cae_qsearch_node(\n    CaeQsearchCtx *q, const CBoard *b, CaeQNodeRef ref,\n",
    r'''static int cae_qsearch_node(CaeQsearchCtx *q, const CBoard *b, CaeQNodeRef ref,
                            int qply, int depth, int32_t alpha, int32_t beta,
                            int32_t *out_value) {
    const CaeArmCtx *arm = q->arm;
    q->stats->qnodes++;
    if ((uint32_t)qply > q->stats->qmax_ply_seen) q->stats->qmax_ply_seen = (uint32_t)qply;

    int32_t stand_pat = 0;
    int stand_kind = CAE_DAG_WINDOW_EXACT;
    int status = cae_qsearch_stand_pat(
        q, b, &ref, alpha, beta, &stand_pat, &stand_kind);
    if (status != CAE_VALUE_OK) return status;

    /* LOWER is produced only when it clears beta, so it is already a legal
     * fail-high cutoff. UPPER is produced only when it is <= the incoming alpha;
     * it contributes no alpha raise and may not be returned as a static value. */
    if (stand_kind == CAE_DAG_WINDOW_LOWER) {
        *out_value = stand_pat;
        return CAE_VALUE_OK;
    }
    if (stand_kind == CAE_DAG_WINDOW_EXACT) {
        if (stand_pat >= beta) { *out_value = stand_pat; return CAE_VALUE_OK; }
        if (stand_pat > alpha) alpha = stand_pat;
    }

    if (qply >= arm->qsearch_max_ply || depth >= arm->resolver_max_depth) {
        q->stats->qply_cutoffs++;
        if (stand_kind == CAE_DAG_WINDOW_UPPER) {
            status = cae_qsearch_force_full_static(q, ref.node_id, &stand_pat);
            if (status != CAE_VALUE_OK) return status;
        }
        *out_value = stand_pat;
        return CAE_VALUE_OK;
    }

    if (q->node_cap > 0 && ++q->nodes_used > (uint64_t)q->node_cap) {
        q->stats->dag_budget_trips++;
        if (stand_kind == CAE_DAG_WINDOW_UPPER) {
            status = cae_qsearch_force_full_static(q, ref.node_id, &stand_pat);
            if (status != CAE_VALUE_OK) return status;
        }
        *out_value = stand_pat;
        return CAE_VALUE_OK;
    }

    int moves[CBOARD_MAX_LEGAL_MOVES];
    int n = cboard_legal_move_indices(b, moves, 0);
    int try_checks = (qply < arm->qsearch_check_plies);
    int32_t best_move = -CAE_QSEARCH_INF;
    int searched = 0;

    for (int i = 0; i < n; i++) {
        PolicyMove pm = POLICY_LUT[b->turn][moves[i]];
        if (pm.from_sq < 0 || pm.to_sq < 0) continue;
        int tactical = cae_qsearch_is_tactical(b, pm.from_sq, pm.to_sq, pm.promotion);
        if (!tactical && !try_checks) continue;

        CBoard child;
        memcpy(&child, b, sizeof(CBoard));
        cboard_push_index(&child, moves[i]);
        int gives_check = cboard_in_check(&child);
        if (!tactical && !gives_check) continue;

        int32_t child_value = 0;
        int rc = cae_qsearch_child(q, &child, ref, moves[i], qply + 1, depth + 1,
                                   -beta, -alpha, &child_value);
        if (rc != CAE_VALUE_OK) return rc;
        child_value = -child_value;
        searched = 1;
        if (child_value > best_move) best_move = child_value;
        if (child_value > alpha) alpha = child_value;
        if (alpha >= beta) {
            *out_value = best_move;
            return CAE_VALUE_OK;
        }
    }

    /* Non-cutoff return must be exact. An UPPER stand-pat bound cannot win a
     * max() by itself; upgrade it, then compare against exact child returns. */
    if (stand_kind == CAE_DAG_WINDOW_UPPER) {
        status = cae_qsearch_force_full_static(q, ref.node_id, &stand_pat);
        if (status != CAE_VALUE_OK) return status;
    }
    *out_value = searched && best_move > stand_pat ? best_move : stand_pat;
    return CAE_VALUE_OK;
}''',
)

# DAG arm contexts snapshot lazy config exactly once after store construction.
replace_once(
    arm_h,
    r'''        if (!ctx->dag
            || cae_nnue_dag_store_init(ctx->dag, w, CAE_ARM_DAG_INITIAL_NODES) != 0) {
            free(ctx->dag);
            cae_nnue_release(w);
            free(ctx);
            cae_nnue_err(err, errlen, "out of memory building the position DAG");
            return NULL;
        }
''',
    r'''        if (!ctx->dag
            || cae_nnue_dag_store_init(ctx->dag, w, CAE_ARM_DAG_INITIAL_NODES) != 0) {
            free(ctx->dag);
            cae_nnue_release(w);
            free(ctx);
            cae_nnue_err(err, errlen, "out of memory building the position DAG");
            return NULL;
        }
        CaeDagLazyConfig lazy_cfg;
        cae_dag_lazy_get_config(&lazy_cfg);
        if (cae_nnue_dag_configure_lazy(
                ctx->dag, lazy_cfg.enabled, lazy_cfg.margin) != 0) {
            cae_nnue_dag_store_release(ctx->dag);
            free(ctx->dag);
            cae_nnue_release(w);
            free(ctx);
            cae_nnue_err(err, errlen, "invalid lazy-eval DAG configuration");
            return NULL;
        }
''',
)
# FastQ root creation: count full work by store delta, not by node creation.
replace_once(
    arm_h,
    r'''    int32_t root_id = CAE_DAG_NO_NODE;
    int created = 0;
    int status = cae_nnue_dag_intern_position(ctx->dag, board, &root_id, &created);
    if (status == CAE_VALUE_OK) {
        if (created) {
            local.nodes_created++;
            if (cboard_in_check(board)) local.nodes_created_in_check++;
            else local.nnue_evals++;
''',
    r'''    int32_t root_id = CAE_DAG_NO_NODE;
    int created = 0;
    const uint64_t evals_before = ctx->dag->nnue_evals;
    int status = cae_nnue_dag_intern_position(ctx->dag, board, &root_id, &created);
    if (status == CAE_VALUE_OK) {
        local.nnue_evals += ctx->dag->nnue_evals - evals_before;
        if (created) {
            local.nodes_created++;
            if (cboard_in_check(board)) local.nodes_created_in_check++;
''',
)


# ---------------------------------------------------------------------------
# 5. Python DAG probe API + shared store stats.
# ---------------------------------------------------------------------------
dag_api_h = "chess_anti_engine/nnue/_nnue_dag_api.h"
# Shared stats builder: append lazy state/counters without changing old keys.
replace_once(
    dag_api_h,
    r'''        "node_reuses", (unsigned long long)h->node_reuses,
        "dag_memory_bytes", (long long)dag_bytes,
        "nnue_payload_bytes", (long long)payload_bytes,
        "memory_bytes", (long long)(dag_bytes + payload_bytes));
''',
    r'''        "node_reuses", (unsigned long long)h->node_reuses,
        "dag_memory_bytes", (long long)dag_bytes,
        "nnue_payload_bytes", (long long)payload_bytes,
        "memory_bytes", (long long)(dag_bytes + payload_bytes),
        "full_nodes", (unsigned long long)h->full_nodes,
        "psqt_only_nodes", (unsigned long long)h->psqt_only_nodes,
        "no_value_nodes", (unsigned long long)h->no_value_nodes,
        "psqt_probes", (unsigned long long)h->psqt_probes,
        "lazy_lower_bounds", (unsigned long long)h->lazy_lower_bounds,
        "lazy_upper_bounds", (unsigned long long)h->lazy_upper_bounds,
        "lazy_upgrades", (unsigned long long)h->lazy_upgrades,
        "lazy_enabled", h->lazy_enabled,
        "lazy_margin", h->lazy_margin);
''',
)
# Expand the Py_BuildValue format by nine key/value pairs.
replace_once(
    dag_api_h,
    r'''        "{s:i,s:i,s:i,s:i,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:L,s:L,s:L}",
''',
    r'''        "{s:i,s:i,s:i,s:i,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:L,s:L,s:L,"
        "s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:i,s:i}",
''',
)
# Add diagnostic configuration + window/PSQT accessors before the method macro.
replace_once(
    dag_api_h,
    "#define CAE_NNUE_DAG_METHODS \\\n",
    r'''static PyObject *py_dag_set_lazy(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    int enabled, margin;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &enabled, &margin)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (cae_nnue_dag_configure_lazy(h, enabled, margin) != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "lazy config needs enabled in {0,1}, margin >= 0, and an empty DAG");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_dag_psqt_value(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    int node_id;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &node_id)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (node_id < 0 || node_id >= h->dag.node_count) {
        PyErr_SetString(PyExc_ValueError, "node_id out of range");
        return NULL;
    }
    if (h->states[node_id].pos.in_check) Py_RETURN_NONE;
    int32_t value = 0;
    const int rc = cae_nnue_state_psqt_value(h->weights, &h->states[node_id], &value);
    if (rc != CAE_VALUE_OK) { raise_status(rc); return NULL; }
    return PyLong_FromLong((long)value);
}

static PyObject *py_dag_value_for_window(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *capsule;
    int node_id, alpha, beta;
    if (!PyArg_ParseTuple(args, "Oiii", &capsule, &node_id, &alpha, &beta)) return NULL;
    CaeNnueDagHandle *h = cae_nnue_dag_from_capsule(capsule);
    if (!h) return NULL;
    if (alpha >= beta) {
        PyErr_SetString(PyExc_ValueError, "alpha must be < beta");
        return NULL;
    }
    int32_t value = 0;
    int kind = CAE_DAG_WINDOW_EXACT, did_eval = 0;
    const int rc = cae_nnue_dag_value_for_window(
        h, node_id, alpha, beta, CAE_RESOLVER_EVAL_CLAMP,
        &value, &kind, &did_eval);
    if (rc != CAE_VALUE_OK) { raise_status(rc); return NULL; }
    return Py_BuildValue("(iii)", kind, value, did_eval);
}

#define CAE_NNUE_DAG_METHODS \
''',
)
replace_once(
    dag_api_h,
    r'''    {"dag_reset", py_dag_reset, METH_VARARGS, NULL},
''',
    r'''    {"dag_reset", py_dag_reset, METH_VARARGS, NULL}, \
    {"dag_set_lazy", py_dag_set_lazy, METH_VARARGS, NULL}, \
    {"dag_psqt_value", py_dag_psqt_value, METH_VARARGS, NULL}, \
    {"dag_value_for_window", py_dag_value_for_window, METH_VARARGS, NULL},
''',
)


# ---------------------------------------------------------------------------
# 6. _nnue_ext: global default-off config for NEW DAG-backed arm contexts.
# ---------------------------------------------------------------------------
ext_c = "chess_anti_engine/nnue/_nnue_ext.c"
replace_once(
    ext_c,
    """/* FastQ's counter block. Lives beside arm_stats_dict because BOTH eval surfaces\n""",
    r'''PyDoc_STRVAR(set_dag_lazy_config_doc,
"set_dag_lazy_config(enabled, margin) -> dict\n\n"
"Configure S1 lazy evaluation for NEW DAG-backed arm contexts only. Default is\n"
"OFF. The pair is snapshotted when nnue-qsearch-dag / nnue-fastq opens; changing\n"
"it does not retune an existing handle. Realized values are read through\n"
"arm_dag_stats(handle). margin is in raw NNUE internal units and must be\n"
"calibrated before enabling on production data.");

static PyObject *py_set_dag_lazy_config(PyObject *Py_UNUSED(self), PyObject *args) {
    int enabled, margin;
    if (!PyArg_ParseTuple(args, "ii", &enabled, &margin)) return NULL;
    char err[256] = {0};
    if (cae_dag_lazy_set_config(enabled, margin, err, sizeof(err)) != 0) {
        PyErr_SetString(PyExc_ValueError, err);
        return NULL;
    }
    CaeDagLazyConfig cfg;
    cae_dag_lazy_get_config(&cfg);
    return Py_BuildValue("{s:i,s:i}",
                         "enabled", cfg.enabled, "margin", cfg.margin);
}

/* FastQ's counter block. Lives beside arm_stats_dict because BOTH eval surfaces
''',
)
replace_once(
    ext_c,
    r'''    {"set_arm_config", py_set_arm_config, METH_VARARGS, set_arm_config_doc},
''',
    r'''    {"set_arm_config", py_set_arm_config, METH_VARARGS, set_arm_config_doc},
    {"set_dag_lazy_config", py_set_dag_lazy_config, METH_VARARGS,
     set_dag_lazy_config_doc},
''',
)
replace_once(
    ext_c,
    r'''    PyModule_AddIntConstant(m, "QSEARCH_DAG_NODE_CAP",
                            (long)CAE_QSEARCH_DEFAULT_DAG_NODE_CAP);
''',
    r'''    PyModule_AddIntConstant(m, "QSEARCH_DAG_NODE_CAP",
                            (long)CAE_QSEARCH_DEFAULT_DAG_NODE_CAP);
    PyModule_AddIntConstant(m, "DAG_LAZY_ENABLED",
                            (long)CAE_DAG_LAZY_DEFAULT_ENABLED);
    PyModule_AddIntConstant(m, "DAG_LAZY_MARGIN",
                            (long)CAE_DAG_LAZY_DEFAULT_MARGIN);
    PyModule_AddIntConstant(m, "DAG_WINDOW_EXACT", (long)CAE_DAG_WINDOW_EXACT);
    PyModule_AddIntConstant(m, "DAG_WINDOW_LOWER", (long)CAE_DAG_WINDOW_LOWER);
    PyModule_AddIntConstant(m, "DAG_WINDOW_UPPER", (long)CAE_DAG_WINDOW_UPPER);
''',
)
# Correct the FastQ identity wording for lazy-enabled contexts without touching
# the unrelated #476 Py_BuildValue bug/fix.
replace_once(
    ext_c,
    r'''"`nnue_evals + nodes_created_in_check == nodes_created` is the evaluate-once\n"
"identity; the in-check term is there because an in-check node is published with\n"
"no static value, the NNUE evaluation being undefined in check.\n");
''',
    r'''"With DAG lazy evaluation OFF (the default), `nnue_evals +\n"
"nodes_created_in_check == nodes_created` remains FastQ's old identity. With\n"
"lazy evaluation ON, arm_dag_stats() owns the stronger tier identities:\n"
"`nnue_evals == full_nodes` and FULL+PSQT_ONLY+NONE == node_count.\n");
''',
)


# ---------------------------------------------------------------------------
# 7. Typing surface.
# ---------------------------------------------------------------------------
pyi = "chess_anti_engine/nnue/_nnue_ext.pyi"
replace_once(
    pyi,
    """QSEARCH_DAG_NODE_CAP: int\n\n# FastQ-4+\n""",
    """QSEARCH_DAG_NODE_CAP: int\nDAG_LAZY_ENABLED: int\nDAG_LAZY_MARGIN: int\nDAG_WINDOW_EXACT: int\nDAG_WINDOW_LOWER: int\nDAG_WINDOW_UPPER: int\n\n# FastQ-4+\n""",
)
replace_once(
    pyi,
    """def arm_eval(\n    name: str, weights_path: str, boards: list[CBoard], /\n) -> tuple[list[int], dict[str, int]]: ...\n""",
    """def arm_eval(\n    name: str, weights_path: str, boards: list[CBoard], /\n) -> tuple[list[int], dict[str, int]]: ...\ndef set_dag_lazy_config(enabled: int, margin: int, /) -> dict[str, int]: ...\n""",
)
replace_once(
    pyi,
    """def dag_reset(handle: object, /) -> None: ...\n""",
    """def dag_reset(handle: object, /) -> None: ...\ndef dag_set_lazy(handle: object, enabled: int, margin: int, /) -> None: ...\ndef dag_psqt_value(handle: object, node_id: int, /) -> int | None: ...\ndef dag_value_for_window(\n    handle: object, node_id: int, alpha: int, beta: int, /\n) -> tuple[int, int, int]: ...\n""",
)


# ---------------------------------------------------------------------------
# 8. Tests: exact tier mechanics + qsearch/FastQ integration on a PSQT-only net.
# ---------------------------------------------------------------------------
test = r'''from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_parse
from tests.test_nnue_native_eval import write_synthetic_pack


@pytest.fixture(scope="module")
def lazy_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """PSQT-only net: FULL == PSQT, so margin 0 is a mathematically exact oracle."""
    rng = np.random.default_rng(20260826)
    mag = 2000
    halfka = rng.integers(
        -mag, mag + 1,
        size=nnue_parse.HALFKA_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    threats = rng.integers(
        -mag, mag + 1,
        size=nnue_parse.THREAT_DIMS * nnue_parse.PSQT_BUCKETS,
        dtype=np.int32,
    )
    path = tmp_path_factory.mktemp("nnue-lazy") / "psqt-only.pack"
    write_synthetic_pack(
        path,
        blobs={"ft_psqt": [(0, halfka)], "threat_psqt": [(0, threats)]},
    )
    return path


@pytest.fixture(autouse=True)
def _restore_configs() -> Iterator[None]:
    _nnue_ext.set_dag_lazy_config(0, 0)
    _nnue_ext.set_arm_config(
        _nnue_ext.RESOLVER_MAX_DEPTH,
        _nnue_ext.QSEARCH_MAX_PLY,
        _nnue_ext.QSEARCH_CHECK_PLIES,
        0,
    )
    _nnue_ext.fastq_set_config(
        _nnue_ext.FASTQ_MAX_QPLY,
        _nnue_ext.FASTQ_NODE_CAP,
        _nnue_ext.FASTQ_DELTA_MARGIN,
        _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
    )
    yield
    _nnue_ext.set_dag_lazy_config(0, 0)


def _cboard(fen: str = chess.STARTING_FEN) -> CBoard:
    return CBoard.from_board(chess.Board(fen))


def _corpus() -> list[CBoard]:
    out: list[CBoard] = []
    rng = random.Random(20260826)
    for _ in range(18):
        board = chess.Board()
        for ply in range(28):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply >= 5 and ply % 3 == 0:
                out.append(CBoard.from_board(board.copy(stack=True)))
    return out


def test_lazy_eval_ships_off() -> None:
    assert _nnue_ext.DAG_LAZY_ENABLED == 0
    assert _nnue_ext.DAG_LAZY_MARGIN == 0


def test_standalone_dag_has_monotone_psqt_to_full_upgrade(lazy_pack: Path) -> None:
    weights = _nnue_ext.load(str(lazy_pack))
    dag = _nnue_ext.dag_open(weights)
    _nnue_ext.dag_set_lazy(dag, 1, 0)
    node, value, created = _nnue_ext.dag_intern_root(dag, _cboard())
    assert created and value is None
    psqt = _nnue_ext.dag_psqt_value(dag, node)
    assert psqt is not None
    before = _nnue_ext.dag_stats(dag)
    assert before["full_nodes"] == before["nnue_evals"] == 0
    assert before["psqt_only_nodes"] == 1

    kind, score, did_eval = _nnue_ext.dag_value_for_window(
        dag, node, psqt - 100, psqt,
    )
    assert (kind, score, did_eval) == (_nnue_ext.DAG_WINDOW_LOWER, psqt, 0)
    bounded = _nnue_ext.dag_stats(dag)
    assert bounded["psqt_only_nodes"] == 1
    assert bounded["lazy_lower_bounds"] == 1
    assert bounded["nnue_evals"] == 0

    kind, score, did_eval = _nnue_ext.dag_value_for_window(
        dag, node, -200_000, 200_000,
    )
    assert kind == _nnue_ext.DAG_WINDOW_EXACT
    assert score == psqt  # PSQT-only synthetic net => FULL exactly equals PSQT.
    assert did_eval == 1
    after = _nnue_ext.dag_stats(dag)
    assert after["psqt_only_nodes"] == 0
    assert after["full_nodes"] == after["nnue_evals"] == 1
    assert after["lazy_upgrades"] == 1

    # FULL is terminal: no later window can cause another propagation.
    _nnue_ext.dag_value_for_window(dag, node, psqt, psqt + 100)
    final = _nnue_ext.dag_stats(dag)
    assert final["nnue_evals"] == 1
    assert final["full_nodes"] == 1


def test_upper_bound_is_ephemeral_and_does_not_upgrade(lazy_pack: Path) -> None:
    dag = _nnue_ext.dag_open(_nnue_ext.load(str(lazy_pack)))
    _nnue_ext.dag_set_lazy(dag, 1, 0)
    node, _, _ = _nnue_ext.dag_intern_root(dag, _cboard())
    psqt = _nnue_ext.dag_psqt_value(dag, node)
    assert psqt is not None
    kind, score, did_eval = _nnue_ext.dag_value_for_window(
        dag, node, psqt, psqt + 100,
    )
    assert (kind, score, did_eval) == (_nnue_ext.DAG_WINDOW_UPPER, psqt, 0)
    stats = _nnue_ext.dag_stats(dag)
    assert stats["lazy_upper_bounds"] == 1
    assert stats["full_nodes"] == stats["nnue_evals"] == 0
    assert _nnue_ext.dag_value(dag, node) is None


def test_lazy_config_is_snapshotted_by_the_live_arm(lazy_pack: Path) -> None:
    _nnue_ext.set_dag_lazy_config(1, 321)
    handle = _nnue_ext.arm_open("nnue-qsearch-dag", str(lazy_pack))
    _nnue_ext.set_dag_lazy_config(0, 0)
    realized = _nnue_ext.arm_dag_stats(handle)
    assert realized["lazy_enabled"] == 1
    assert realized["lazy_margin"] == 321


def test_margin_zero_is_bit_exact_on_a_psqt_only_net_for_qsearch(lazy_pack: Path) -> None:
    boards = _corpus()
    assert len(boards) >= 100
    _nnue_ext.set_arm_config(12, 3, 0, 0)

    _nnue_ext.set_dag_lazy_config(0, 0)
    oracle = _nnue_ext.arm_open("nnue-qsearch-dag", str(lazy_pack))
    expected = _nnue_ext.arm_handle_eval(oracle, boards)

    _nnue_ext.set_dag_lazy_config(1, 0)
    lazy = _nnue_ext.arm_open("nnue-qsearch-dag", str(lazy_pack))
    actual = _nnue_ext.arm_handle_eval(lazy, boards)
    assert actual == expected

    stats = _nnue_ext.arm_dag_stats(lazy)
    assert stats["nnue_evals"] == stats["full_nodes"]
    assert stats["full_nodes"] + stats["psqt_only_nodes"] + stats["no_value_nodes"] == stats["node_count"]
    assert stats["lazy_lower_bounds"] + stats["lazy_upper_bounds"] > 0
    assert stats["psqt_only_nodes"] > 0


def test_margin_zero_is_bit_exact_on_a_psqt_only_net_for_fastq(lazy_pack: Path) -> None:
    boards = _corpus()[:120]
    _nnue_ext.fastq_set_config(4, 0, 200, 1)  # budget OFF: compare search semantics only.

    _nnue_ext.set_dag_lazy_config(0, 0)
    baseline = _nnue_ext.arm_open("nnue-fastq", str(lazy_pack))
    expected = _nnue_ext.arm_handle_eval(baseline, boards)

    _nnue_ext.set_dag_lazy_config(1, 0)
    lazy = _nnue_ext.arm_open("nnue-fastq", str(lazy_pack))
    actual = _nnue_ext.arm_handle_eval(lazy, boards)
    assert actual == expected

    store = _nnue_ext.arm_dag_stats(lazy)
    fq = _nnue_ext.fastq_stats(lazy)
    assert store["nnue_evals"] == store["full_nodes"]
    assert fq["nnue_evals"] == store["nnue_evals"]
    assert store["lazy_lower_bounds"] + store["lazy_upper_bounds"] > 0


def test_reset_clears_tiers_and_work_but_keeps_lazy_snapshot(lazy_pack: Path) -> None:
    _nnue_ext.set_dag_lazy_config(1, 77)
    handle = _nnue_ext.arm_open("nnue-qsearch-dag", str(lazy_pack))
    _nnue_ext.arm_handle_eval(handle, _corpus()[:10])
    before = _nnue_ext.arm_dag_stats(handle)
    assert before["node_count"] > 0
    _nnue_ext.arm_dag_reset(handle)
    after = _nnue_ext.arm_dag_stats(handle)
    assert after["node_count"] == 0
    assert after["full_nodes"] == after["psqt_only_nodes"] == after["no_value_nodes"] == 0
    assert after["nnue_evals"] == after["psqt_probes"] == after["lazy_upgrades"] == 0
    assert after["lazy_enabled"] == 1
    assert after["lazy_margin"] == 77
'''
write("tests/test_nnue_lazy_eval.py", test)


# ---------------------------------------------------------------------------
# 9. A small real-net runner. It does not invent a production margin.
# ---------------------------------------------------------------------------
bench = r'''#!/usr/bin/env python3
"""Measure S1 DAG lazy evaluation on a fixed deterministic position corpus.

This is a MEASUREMENT tool, not a tuner. `--margin` is explicit. Run lazy-off and
lazy-on against the same arm/corpus; it reports value mismatches, full-propagation
skip rate and wall time. Production activation still requires the calibration
rule in docs/nnue_speed_plan.md; this script never chooses a margin for you.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext


def corpus(n: int, seed: int) -> list[CBoard]:
    rng = random.Random(seed)
    out: list[CBoard] = []
    while len(out) < n:
        b = chess.Board()
        for ply in range(80):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if ply >= 6 and ply % 2 == 0:
                out.append(CBoard.from_board(b.copy(stack=True)))
                if len(out) >= n:
                    break
    return out


def run(arm: str, pack: Path, boards: list[CBoard], *, lazy: bool, margin: int):
    _nnue_ext.set_dag_lazy_config(int(lazy), margin if lazy else 0)
    h = _nnue_ext.arm_open(arm, str(pack))
    t0 = time.perf_counter()
    values = _nnue_ext.arm_handle_eval(h, boards)
    dt = time.perf_counter() - t0
    return values, dt, _nnue_ext.arm_dag_stats(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--arm", choices=("nnue-qsearch-dag", "nnue-fastq"), default="nnue-fastq")
    ap.add_argument("--margin", type=int, required=True, help="calibrated raw NNUE-unit margin")
    ap.add_argument("--positions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.margin < 0:
        raise SystemExit("--margin must be >= 0")

    boards = corpus(args.positions, args.seed)
    off_v, off_s, off = run(args.arm, args.pack, boards, lazy=False, margin=0)
    on_v, on_s, on = run(args.arm, args.pack, boards, lazy=True, margin=args.margin)
    mismatches = sum(a != b for a, b in zip(off_v, on_v, strict=True))
    bounds = int(on["lazy_lower_bounds"]) + int(on["lazy_upper_bounds"])
    probes = int(on["psqt_probes"])
    result = {
        "arm": args.arm,
        "positions": len(boards),
        "margin": args.margin,
        "value_mismatches": mismatches,
        "lazy_off_seconds": off_s,
        "lazy_on_seconds": on_s,
        "speedup": off_s / on_s if on_s else None,
        "lazy_off_full_evals": int(off["nnue_evals"]),
        "lazy_on_full_evals": int(on["nnue_evals"]),
        "psqt_probes": probes,
        "bound_returns": bounds,
        "bound_return_frac_of_psqt_probes": bounds / probes if probes else 0.0,
        "full_eval_reduction": 1.0 - int(on["nnue_evals"]) / max(1, int(off["nnue_evals"])),
        "lazy_store": on,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
'''
write("scripts/nnue_lazy_bench.py", bench)


doc = r'''# S1 NNUE lazy evaluation implementation

This PR implements the **mechanism** preregistered as S1 in
`docs/nnue_speed_plan.md`. It deliberately does **not** choose or enable a
production margin.

## Invariant

A canonical DAG node stores only position-intrinsic facts:

```
NONE (in check)
PSQT_ONLY  ->  FULL
```

The upgrade is monotone. `FULL` is the exact raw native-NNUE static value. No
alpha/beta result, cutoff result, mate score, terminal verdict or backed-up
search value is written into the DAG.

Every search static read goes through `cae_nnue_dag_value_for_window`. Given
PSQT value `p`, configured margin `m`, and a visit window `(alpha, beta)`:

- `p - m >= beta` -> ephemeral LOWER bound; the visit cuts off without FC propagation;
- `p + m <= alpha` -> ephemeral UPPER bound; search may continue without FC propagation;
- otherwise -> upgrade once to FULL and return EXACT.

An UPPER bound is **never returned as a static value**. If max-qply, a quiet
certificate, a node tripwire, path ceiling, or an ordinary non-cutoff return
would expose the node's own static score, the node upgrades first. This is the
rule that keeps laziness an evaluation optimization rather than a silent search
semantic change when the configured margin is valid.

## Default and counters

Lazy evaluation ships OFF (`DAG_LAZY_ENABLED == 0`). New DAG-backed arm contexts
snapshot `set_dag_lazy_config(enabled, margin)` at construction; changing the
global afterwards does not retune a live handle. The realized pair is reported
by `arm_dag_stats()`.

The store exports two exact identities after each reset:

```
state_inits + state_makes == node_count
nnue_evals == full_nodes
full_nodes + psqt_only_nodes + no_value_nodes == node_count
```

and reports `psqt_probes`, `lazy_lower_bounds`, `lazy_upper_bounds`, and
`lazy_upgrades` separately.

## Calibration / activation gate

No production margin is committed here. Before enabling S1, follow the frozen
rule in `docs/nnue_speed_plan.md`: calibrate the FULL-vs-PSQT residual on a real
representative corpus, hold out a validation slice, and require the documented
miss-rate and predicted/real skip-rate gates. `scripts/nnue_lazy_bench.py` accepts
an **explicit** candidate margin and compares lazy-on against lazy-off on the same
deterministic corpus; it never selects the margin.

The synthetic test net is PSQT-only, so `margin=0` is an exact envelope. That
lets CI prove both LOWER and UPPER skip paths are exercised while qsearch-DAG and
FastQ remain bit-identical to their lazy-off answers. It is a correctness test of
the mechanism, not evidence that margin 0 is safe for the production net.
'''
write("docs/nnue_lazy_eval.md", doc)

print("S1 lazy-eval patch applied")
