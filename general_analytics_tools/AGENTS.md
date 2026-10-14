# Agent instructions for ad hoc analytics

Read [README.md](README.md) and the repository's [CONTRIBUTING.md](../CONTRIBUTING.md)
before using or changing this toolkit. It orchestrates native ss104 functionality;
do not duplicate or weaken the core solver, filter, measurement or parity contracts.

- Work from a verified revision; preserve user edits and prior dated artifacts.
- Use offline, immutable official input snapshots with source/vintage/units and
  retrieval times. Missing values stay missing. CPI is not core PCE; wages,
  payrolls, household employment and unemployment are different measurements.
- Explicitly list complete conditioning channels. Label other conditional-quarter
  measurements latent. Separate population HP endpoint revisions from observation news.
- Only public-default fixed parameters are supported here. Do not import the
  unconverged M7 mode, start estimation or pretend numerical parity validates calibration.
- Use the bounded CPU CLI, a unique ignored output directory, and separate rendering.
  Scenario rate changes must be verified actual impact responses, not raw shock sizes.
- Retain initial, deterministic, measurement-gap and numerical-remainder terms.
  Never relax the 1e-8 plotted-unit HD gate or remove a diagnostic label.
- Maintain the six-panel style, readable HD legends/residual strips, source/vintage,
  units, captions and actual-history/latent-nowcast/forecast distinctions.
- No uncertainty bands without a supported uncertainty calculation. Use one
  consolidated identification caveat and explicit decision-relevant limitations.
- Tests use small clearly synthetic fixtures; real data smoke runs use ignored,
  configurable inputs/outputs. Run the relevant tests, full normal gates when
  practical, and visually inspect output. Do not commit private data or generated plots.

The user's explicit instructions determine publication permissions; this file
does not independently authorize pushes, PRs, merging or messages to other people.
