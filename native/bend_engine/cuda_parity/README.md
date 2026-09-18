# DeepFin CUDA package parity gate

This is PR 3 in the Bend-engine sequence. It is intentionally a **manual GPU
gate** because GitHub-hosted CI does not provide the NVIDIA/CUDA environment
used by DeepFin.

It compares the same real DeepFin AOTInductor package through two paths:

```text
CBoard fixtures -> BF16 encoded batch -> Python AOT package
                               \-> Bend -> native C++ AOTI package
```

Both paths expand a compact 1858 policy head to DeepFin's dense 4672 search
policy and convert policy/WDL outputs to float32. The harness compares two
independent bit-level summaries (XOR and index-weighted sum) plus tensor counts
and raw policy width.

The native executable contains no Python runtime. Python is the test oracle and
fixture builder only.

## Run on the DeepFin CUDA host

Install Bend first if needed:

```bash
native/bend_engine/install_bend.sh
export PATH="$PWD/build/bend_toolchain/bin:$PATH"
```

Then choose one real fixed-batch DeepFin package and its input width:

```bash
uv run python native/bend_engine/cuda_parity/run_parity.py \
  --package data/aot_models_512/chess_b16.pt2 \
  --input-planes 175 \
  --device-index 0
```

The package filename normally supplies the fixed batch size. For a differently
named package, pass `--batch N`.

Use `--input-planes 146` for the v1 34-extra-plane encoding and
`--input-planes 175` for v2_threats (63 extras). The harness deliberately
requires this value rather than guessing from the package.

A passing run ends with `CUDA parity PASS`. A mismatch prints each differing
field.

## Scope

This gate validates:

- actual CBoard-produced neural inputs;
- BF16 input transport;
- a real DeepFin CUDA `.pt2` package;
- C++ `AOTIModelPackageLoader` on CUDA;
- compact-policy to dense-policy expansion parity;
- WDL output parity;
- Bend receiving native inference summaries.

It does **not** yet validate native hot weight rebinding with
`load_constants`. Run this against a package containing the checkpoint you
intend to compare. Live checkpoint rebinding is a later gate if the standalone
engine/client needs one package to serve multiple published nets.
