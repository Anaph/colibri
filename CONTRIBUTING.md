# Contributing

Keep changes focused and preserve Colibri's dependency-free default CPU path.

## Branches

- **`main`** is the stable branch. It's what users clone, and it stays known-good.
- **`dev`** is the integration branch. **Open your PR against `dev`.** Reviewed PRs
  land there first; once a batch is tested and stable, the maintainer fast-forwards
  it into `main`. This keeps `main` clean instead of taking every PR one at a time.

Every PR — on either branch — is reviewed for a clean build (0 warnings), the
engine validation (REF-mode token match where applicable), and its own targeted
validation before merge.

## Local checks

Run the lightweight checks locally:

```sh
make check
```

`make -C c check` remains available for scripts that already run from the
engine directory.

This performs one portable CPU build and the test suite. It does not download
a model. The test suite needs cmake ≥ 3.24 and a C++ compiler (GoogleTest glue
only — all test logic is C); the first configure fetches a pinned gtest over
the network unless a system GTest is installed.

Benchmark reports should include the commit, exact commands, hardware and
storage details, warm-up policy, run count, and median throughput.
