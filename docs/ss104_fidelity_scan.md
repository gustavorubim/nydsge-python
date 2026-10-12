# nydsge-python fidelity scan vs FRBNY DSGE (Model 1002)

**2026-10-12 port update:** the stub and missing-equation findings below describe
the original scan revision. M1-M6 now implement public ss104 at fixed default
parameters, including all 29 regimes and full conditional mean forecasts.
The independent source Julia oracle, numerical errors and certification boundary
are recorded in [ss104_m6.md](ss104_m6.md) and [ss104_parity.json](ss104_parity.json).
Published-release reproduction still requires matching vintages/posteriors and
the nonpublic inputs described in this historical scan. No estimation was added
to the ss104 workflow.

Scope: `main @ 6716227` (PRs #1–#3), fresh clone `/workspace/nydsge-scan`. Read-only: nothing was pushed, committed, or opened.
Reference code: public DSGE.jl `master @ c09dfe8` (2026-07-23, "migrate code to work on julia v1.12"), plus tag v1.3.0 (45c2662).
Reference releases: NY Fed DSGE Model Forecast, Sep 18 2026 (latest; I found no newer release as of this scan) and Jun 2026. Model doc: `docs/DSGE_Model_Documentation_1002.pdf` (2021-06-17).
Working notes: `NOTES.md` (untracked). Scratch: `/workspace/nydsge-scan-ref/{DSGE.jl,jl,parity}`.

## 1. Verdict

* **ss10 implementation: faithful.** Re-verified against freshly installed DSGE.jl master (Julia 1.12.7) at the MCMC posterior median and at defaults, n_ant = 0 and 6. Γ0, Γ1, Ψ, Π, C, TTT, RRR, CCC, ZZ, DD, QQ, EE and the steady state agree to ≤1e-11. All 63 free-parameter prior log-densities agree to 1e-13, and the fixed/free flags agree. PRs #1–#3 did not break parity.
* **Not faithful to what FRBNY runs today.** The production spec is **ss104**: ss97 ← ss93 ← ss87 ← ss64 ← ss9, dispatched last in `DSGE.jl/src/models/representative/m1002/subspecs.jl:134` and defined at 6889–7021. It adds:
  * COVID shocks;
  * a regime-switching flexible-AIT rule with time-varying credibility and a temporary ZLB;
  * an AR(1) inflation measurement error;
  * SPD/SME expected-FFR observables;
  * conditional-quarter measurement errors;
  * full conditioning on GDP and core PCE nowcasts.
  
  Production also uses features that are not public: SPF short-run inflation expectations as a conditioning input (named in the Sep-2026 release) and anticipated markup shocks (dev commits visible only in a stray log in the repo). Our "ss104" (`src/nydsge/models/m1002.py:1138-1147`) is a stub that routes to the ss10 matrices (`m1002.py:1249, 1434-1455`; `solve.py:442-470` drops `tvis`).
* **The gap to published results is mostly spec, not data.** Mean Q4/Q4 GDP 2026/27/28: ours 2.5/2.5/2.7 vs NY Fed 1.2/0.1/0.5. Core PCE: ours 3.2/2.3/1.8 vs 3.3/2.1/1.8. Inflation is close. Turning on the data features we can reproduce publicly moves GDP by ≤0.2 pp:
  * SPF nowcast;
  * BofA spread basis;
  * SME expected-FFR path with 6 anticipated shocks;
  * population padding.
  
  The ~1.3–2.4 pp GDP gap therefore sits in the ss104 structure and posterior, the unpublished inputs, and vintage. `docs/nyfed_comparison.md:116` attributes the gap mainly to anticipated shocks/SME; that is **not supported**.

## 2. Reference spec (ss104), with citations

All paths are relative to `DSGE.jl/src/models/representative/m1002/` unless noted.

| Item | FRBNY reference |
|---|---|
| Dispatcher / ss104 definition | `subspecs.jl:134`, `6889-7021` (docstring: "clean the forecasting process in the post-covid era (July 2024), based off ss97"). ss105 exists (7030) but is not dispatched. |
| init settings | `m1002.jl:409-426`: anticipated AIT rm shocks 1–6, expected_ffr 1–6, forecast_horizons 40, add_iid_cond_obs_gdp/corepce_meas_err, add_ait_rm, add_taylor_rm, spread_adjust 2, add_altpolicy_pgap/ygap |
| COVID states/shocks (≥59) | states `m1002.jl:169-214` (ziid_t, biidc_t, φ_t, Eφ_t; e_meas_π ≥87); eqcond `eqcond.jl:63-81` (φ in Euler), `272-284` (φ in MRS), `379-390` (ziid), `409-422` (biidc), `479-486` (φ AR1); params `m1002.jl:731-750` |
| Inflation meas. error AR(1) (≥87) | `measurement.jl:161-172`; ρ_meas_π `m1002.jl:643`, σ_meas_π `720`; `subspecs_builder.jl:526` add_meas_pi!, `609` rm_iid_pce_meas_err! |
| Flexible AIT | `altpolicy/flexible_ait.jl:15-75`; pgap/ygap `eqcond.jl:727-809`; ait_rm into eq_mp `eqcond.jl:895-904`; AIT anticipated shocks `eqcond.jl:572-590`; ρ_ait_rm `m1002.jl:648`, σ_ait_rm `725`, σ_ait_r_m `920` |
| ss104 policy settings | φ_π 4, φ_y 3, Thalf 10, pgap_init 0.125, ygap_init 12, ρ_smooth 0.9; ZLB 2020Q4–2021Q4 (gensys2 temporary ZLB); tvcred 2020Q4→2026Q4 from 0→1; tworule_zlb; adjusted_historical_expectations; spd_expect; cond_type :full; remove_rm_t_shocks 8; remove_pistar_shocks 5; σ_ait_r_m1..6 regime 2 from model regime 9 (`subspecs.jl:6889-7021`; builder `subspecs_builder.jl:20-175, 285-467, 484`) |
| Expected FFR obs | `subspecs_builder.jl:683-789`; `measurement.jl:368-419`; data from "SPD" `observables.jl:488-504`; σ_exp_rm_i = 0.0375+0.00625·i (`m1002.jl:1005-1014`) |
| Cond. meas. errors | `measurement.jl:127-130, 176-179`; σ_condgdp `m1002.jl:884` (ss104: 2.0), σ_condcorepce `904` (1.25) |
| Other model settings | `m1002.jl:1195-1290`: antshocks biidc/φ/ziid (≥59); obs_gdp1 from MEDIANANTGDP & pgap/ygap pseudo-obs (≥62); cond_full_names [gdp, corepce, spread, nominalrate, longrate]; cond_semi_names [spread, nominalrate, longrate]; shockdec_startdate 2007Q1; use_population_forecast true |
| Long-horizon obs with regimes | `measurement.jl:203-217` (k_periods_ahead_expected_sums using regime TTTs) |
| Data | spread = BAA <2016Q4, BAMLC8A0C15PYEY ≥2016Q4 (`observables.jl:228-254`); longinflation = ASACX10 − 0.5 (`258-290`); OIS anticipated obs (`387-404`, internal) |
| Sampler | DSGE.jl default `:MH` (`defaults.jl:113`); NY Fed public statements (Liberty Street Aug 2019; Staff Report 893) describe SMC. **Unverified** whether 2026 releases use SMC. |
| Doc PDF | Documents COVID shocks and flexible AIT, but with ρR .75, φp 15, φy 3 and linear credibility over 24q from 2020Q4. The ss104 code (2024) supersedes it. |

## 3. Equation-level diff: our ss10 vs ss104

The ss10 core block is identical (verified numerically, §5). ss104 adds the following; each item is missing in ours unless stated.

1. φ_t labor-supply shock in the Euler equations and the MRS, plus anticipated Eφ_t. (`eqcond.jl:63-81, 272-284, 479-486`)
2. ziid_t iid TFP shock and biidc_t iid consumption/Euler shock, each with a 1-period anticipated version. (`eqcond.jl:379-422`)
3. Regime-dependent σ's for the COVID shocks. (`subspecs_builder.jl:484`)
4. e_meas_π AR(1) measurement error on core PCE and GDP-deflator inflation, which replaces the iid errors. (`measurement.jl:161-172`, `subspecs_builder.jl:526,609`)
   * Ours carries the `rho_meas_pi`/`sigma_*` parameter specs (`m1002.py:219-345`) but nothing uses them.
5. pgap_t/ygap_t states and the flexible-AIT rule with ρ_smooth, a credibility-weighted two-rule expectation, and the ait_rm shock plus 6 anticipated AIT shocks. (`eqcond.jl:572-590, 727-809, 895-904`; `flexible_ait.jl`)
6. remove_rm_t_shocks 8. This alters the eq_rm row: master writes Γ1[eq_rm, rm_t] = ρ_rm without the noant factor, which only matters when this setting is on.
7. remove_pistar_shocks 5. (`eqcond.jl:~536`)
8. Regime-switching system: one TTT/RRR/CCC per regime, gensys2 temporary ZLB, time-varying information sets.
   * Ours: `solve.py:442-470` does `del tvis`.
9. Measurement:
   * obs_exp_nominalrate1..6 with iid errors. The ss104 stub in ours already builds these rows (`models/m1002_measurement.py:170-179`), but there is no SPD/SME data loader and no regime awareness.
   * Conditional-quarter iid errors exist in ours (`m1002_measurement.py:61,153-158`) at the ss104 σ values.
   * obs_gdp1 anticipated-GDP infrastructure exists (`m1002_measurement.py:181-192`), but there is no MEDIANANTGDP data.

Instantiated dimensions:
* Ours ss10: 68 endogenous, 16 augmented, 24 shocks, 19 observables, 95 params (63 free).
* Ours ss104: 68/24/32/25, with no COVID states, no meas_π, no AIT and no regimes.
* DSGE.jl ss104 (instantiated, Julia master, forecast start 2026Q3):
  * 84 endogenous states, 110 augmented, 48 shocks, 28 observables, 130 params (80 free), 29 regimes.
  * Observables add obs_gdp1, obs_pgap, obs_ygap and obs_exp_nominalrate1–6.
  * Shocks add φ/biidc/ziid (and their 1-period anticipated versions), meas_π_sh, pgap_sh, ygap_sh, rm_ait_sh plus rm_ait_shl1–6, and condgdp_sh/condcorepce_sh.
  * Settings: sampling_method = MH (in code), forecast_horizons = 40, cond_full_names = [gdp, corepce, spread, nominalrate, longrate], cond_semi_names = [spread, nominalrate, longrate], n_mon_anticipated_shocks = 6, shockdec_startdate = 2007Q1.
* Ours ss104 vs DSGE.jl ss104: missing 16 endogenous states, 16 shocks, 3 observables (obs_gdp1, obs_pgap, obs_ygap) and 28 regimes. The 3 missing observables are infrastructure we have only partly or not at all.

## 4. Data and measurement

| Item | Ours | FRBNY | Impact |
|---|---|---|---|
| Spread splice | BAA through 2023Q3, then BofA (`data.py:1389-1392`). FRED's BAMLC8A0C15PYEY only goes back to 2023Q4 (13 q). | BAA <2016Q4, BofA ≥2016Q4 | Level break at 2023Q4. Mean BofA−BAA over the overlap is −18.5 bp (sd 3 bp). Toggle (BAA shifted −18.5 bp over 2016Q4–2023Q3): ≈0 effect on forecasts. **Low.** |
| Expected FFR | None in the c1c2 config (`n_mon_anticipated_shocks: 0`, `configs/quarterly_economy_c1c2.json:21`) | SME/SPD 1–6q (public SME since 2021Q4; OIS/internal before) | Toggle n_ant=6 + SME Jul-2026: GDP −0.19/+0.05/+0.21, core PCE −0.06/−0.06/−0.03, FFR path 3.63 vs 3.81–3.85 endogenous. n_ant=6 *without* data is unidentified (FFR 4.3–4.8). **Medium.** |
| Nowcast conditioning | Semi-style. Conditions on nominal rate, spread, longrate, longinflation (`economy.py:880-900`; config) | cond_type :full → GDP and core PCE nowcast (SPF/staff) + spread, nominalrate, longrate, with cond meas. errors. The Sep-2026 post also cites SPF short-run inflation. | Toggle SPF Aug-26 Q3 (GDP 2.5, core PCE 2.7): GDP −0.08/+0.07/+0.04, core PCE −0.12/−0.08/−0.03. **Medium.** |
| longinflation conditioning | Ours conditions on it | Not in cond_full/semi names | Minor; switch off for fidelity. **Low.** |
| Population | HP-filtered, padding with trend extrapolation (`data.py:537-565`); GDP aggregate uses a constant 1.055% growth; the population forecast is optional and `economy.py` never passes it | use_population_forecast = true | <0.001 pp. **Low.** |
| Core PCE vs PCE | Core PCE (PCEPILFE) for obs_corepce; GDP deflator separate | Same in ss10/ss104 | Match. |
| Partial quarters | Snapshot `fred_levels_20261006.csv` has PCEPILFE, PCE, CE16OV, AWHNONAG through 2026Q3 and DFF through 2026Q4 (partial) | Release vintage cutoffs; partial quarters only via conditioning | Make sure the partial quarter is either conditioning-only or dropped. **Medium** (hygiene). |
| COVID quarters | Masked 2020Q2–Q4 (config) | Modelled with COVID shocks and regime σ's | Structural, not a data issue. Masking is a reasonable ss10 approximation. |

## 5. Estimation, forecast, conditioning

* **Parity (verified):** see §1. Max abs diffs (Python vs DSGE.jl master @ c09dfe8, Julia 1.12.7), with rows and columns permuted by state, shock and equation names. Harness: `/workspace/nydsge-scan-ref/parity/{py_export.py,jl_export.jl,compare.py}`.

| Case | Γ0 | Γ1 | C, Ψ, Π | TTT | RRR | CCC | ZZ | DD, QQ, EE | steady state |
|---|---|---|---|---|---|---|---|---|---|
| posterior median, n_ant 6 (84 states) | 1.5e-13 | 4.2e-13 | 0 | 8.3e-12 | 2.2e-12 | 0 | 1.1e-12 | 0 | 6.2e-13 |
| DSGE.jl defaults, n_ant 6 | 2.0e-14 | 5.8e-14 | 0 | 2.7e-12 | 1.6e-12 | 0 | 1.8e-13 | 0 | 6.1e-14 |
| posterior median, n_ant 0 (62 endo / 78 states; c1c2 config) | 1.5e-13 | 4.2e-13 | 0 | 8.6e-12 | 2.4e-12 | 0 | see below | see below | 6.2e-13 |

* **n_ant 0 measurement layer.** DSGE.jl master's `compute_system` throws for n_mon_anticipated_shocks=0: a BoundsError in `one_to_k_periods_ahead_expectations` with k=0. This is an upstream bug; v1.3.0 worked, and the existing oracles in `tools/oracle_julia` come from v1.3.0. Instead I cross-checked our n0 ZZ against Julia's n6 ZZ on all 13 core observables × 78 common states: 1.0e-12. DD, EE and the common-shock QQ match exactly. The rm_tl1..6 loadings on long-rate/long-inflation rows are expected in n6 only.
* **Test suite.** The data, Kalman, c1c2-wiring, expected-FFR, MH-Jacobian, model1002 and public-sources tests all pass at 6716227 (165 tests). This covers the PR #1–#3 loaders, the presample handling and the Jacobian.
* **Presample and masking at the c1c2 config.**
  * Estimation (`cli.py:2201-2208, 5620`) loads from date_presample_start and excludes presample periods from the likelihood. This matches DSGE.jl (`defaults.jl:46-48`: presample 1959Q3, main sample 1960Q1).
  * The forecast/economy path (`economy.py:304-308`, `start_date` 1960-Q1) filters from 1960Q1 with no presample run-in. Effect on 2026+ forecasts is negligible (initial conditions decay). **Low.**
  * The pre-ZLB QQ treatment for anticipated shocks (`kalman.py:196-232`) follows DSGE.jl ss10's approach in concept but was not checked numerically.
  * Masking 2020Q2–Q4 (`economy.py:309-312`) has no DSGE.jl counterpart; FRBNY models these quarters with COVID shocks (ss≥59). This is a deliberate ss10 approximation.
* **Priors:** 63 free logpdfs equal (sum −86.99103117577 both). Bounds equal. Transform names differ only on fixed parameters, because ModelConstructors stores fixed parameters as Untransformed; this is immaterial.
* **Sampler:** ours is RW Metropolis-Hastings (`estimate.py:724`). This matches the DSGE.jl default but **not** the NY Fed's stated SMC. MH vs SMC should give the same posterior if both converge, so the issue is robustness/multimodality, not bias. **Medium.**
* **Regime-switching priors and parameters** (COVID σ's by regime, AIT σ's, ρ_meas_π…): absent. **Critical**, part of ss104.
* **Forecast:**
  * no enforce_zlb (`forecast.py`; only `build_zlb_conditional_observations` at `forecast.py:318`), vs DSGE.jl forecast_zlb_value 0.13/4 (`defaults.jl:167`); **Low** at the current ~3.6% rates;
  * horizon 20 (`configs/quarterly_economy_c1c2.json:13`) vs 40; **Low**;
  * shockdec start: the FRBNY default is 2007Q1;
  * we report means with 68% bands; the published table shows means with 68% bands — matched in kind.

## 6. Results vs NY Fed (Sep 18 2026)

Q4/Q4; published = means.

| | 2026 | 2027 | 2028 | 2029 |
|---|---|---|---|---|
| NY Fed GDP | 1.2 | 0.1 | 0.5 | 1.2 |
| NY Fed core PCE | 3.3 | 2.1 | 1.8 | 1.8 |
| NY Fed r* | 1.9 | 1.8 | 1.4 | 1.2 |
| Ours E0 GDP | 2.48 | 2.48 | 2.73 | – |
| Ours E0 core PCE | 3.22 | 2.26 | 1.82 | – |

Attribution by toggles (c1c2 masked mode, mean paths; scripts `/workspace/nydsge-scan-ref/parity/toggle_exp.py`, `toggle_sme.py`, results `toggle_results.csv`):

| Experiment | Core PCE 26/27/28 | GDP 26/27/28 |
|---|---|---|
| E0 baseline | 3.22/2.26/1.82 | 2.48/2.48/2.73 |
| + SPF Aug-26 Q3 nowcast | 3.10/2.18/1.79 | 2.40/2.55/2.77 |
| + spread on BofA basis 2016Q4–2023Q3 | ≈E0 | ≈E0 |
| n_ant=6 (σ 0.2) + SME Jul-26 path | 3.16/2.20/1.79 | 2.29/2.53/2.94 |
| n_ant=6, no SME data | – | 1.78/2.17/3.32 (FFR 4.3–4.8; unidentified) |
| population forecast padding | <0.001 | <0.001 |

Gap remaining after the public data features: GDP ≈ +1.1 / +2.4 / +2.3 pp; core PCE within ±0.2 pp.

Candidates I could not toggle without implementing ss104:
* the flexible-AIT credibility path (now ≈1) with φ_π 4, φ_y 3, ρ 0.9;
* the COVID-shock posterior and regime σ's;
* the meas_π AR error;
* SPF short-run inflation conditioning;
* the anticipated markup shocks;
* the FRBNY posterior itself.

## 7. Prioritized change list

| # | Pri | What differs | Ours | FRBNY ref | Economic impact | Effort | Fix |
|---|---|---|---|---|---|---|---|
| 1 | Critical | Regime-switching solve (gensys2 temp ZLB, regime TTTs, TVIS) | `solve.py:442-470` (`del tvis`) | `subspecs_builder.jl:20-175`; `measurement.jl:203-217` | Prerequisite for AIT/credibility/ZLB history; drives expectations and the GDP path | L (2–3 wk) | Regime-indexed system container; gensys2 port; regime-aware Kalman filter and forecast; parity tests vs DSGE.jl ss104 |
| 2 | Critical | Flexible-AIT rule + pgap/ygap + tvcred + two-rule | absent; stub `m1002.py:1138-1147` | `flexible_ait.jl:15-75`; `eqcond.jl:727-809, 895-904`; `subspecs.jl:6889-7021` | Policy reaction and expectations post-2020; likely the largest GDP-gap source | L | Implement on top of #1 with ss104 parameter values |
| 3 | Critical | COVID shocks φ/ziid/biidc (+anticipated), regime σ's | absent | `eqcond.jl:63-81, 272-284, 379-422, 479-486`; `m1002.jl:731-750` | Replaces 2020 masking; changes the posterior and smoothed states | M–L | Add states and shocks; remove masking for ss104 |
| 4 | High | meas_π AR(1) error replacing iid PCE error | params only (`m1002.py:219-345`) | `measurement.jl:161-172`; `subspecs_builder.jl:526,609` | Inflation nowcast and filtering | S–M | Add state + ZZ/TTT rows; gate on subspec ≥87 |
| 5 | High | Full conditioning on GDP/core PCE nowcast with cond meas. errors | semi-style list `economy.py:880-900` | cond_full_names `m1002.jl:1195-1290`; `measurement.jl:127-130,176-179` | −0.1 pp core PCE 2026 (toggle) | S | Add SPF/staff nowcast inputs; cond_type full in config |
| 6 | High | Expected-FFR (SME/SPD 1–6) observables + anticipated shocks, data loader | measurement rows exist (`m1002_measurement.py:170-179`), no data; config n_ant 0 | `observables.jl:488-504`; `subspecs_builder.jl:683-789` | Pins the rate path (3.63 vs 3.8); GDP ±0.2 | M | Public SME medians loader (2021Q4+), n_ant 6, document the pre-2021Q4 OIS gap |
| 7 | Medium | Sampler: MH vs SMC | `estimate.py:724` | NY Fed SR893/LSE; `defaults.jl:113` (:MH) | Robustness for the larger ss104 posterior | M | Add an adaptive tempered SMC; cross-check against MH |
| 8 | Medium | Partial-quarter observations in the snapshot | `data/public/fred_levels_20261006.csv` | release cutoffs | Biased last observation | S | Drop incomplete quarters or route them to conditioning |
| 9 | Medium | Docs claims outdated | `docs/nyfed_comparison.md:32,116` | ss104 public | Misleading | S | Update with ss104 and the toggle evidence |
| 10 | Low | Spread splice 2016Q4–2023Q3 on BAA | `data.py:1389-1392` | `observables.jl:228-254` | ≈0 | S | Optionally apply the −18.5 bp overlap adjustment; document |
| 11 | Low | longinflation in conditioning set | config | cond names lack it | small | S | Drop for ss104 fidelity |
| 12 | Low | No enforce_zlb; horizon 20 vs 40; population forecast unused | `forecast.py`; config:13; `data.py:537-565` | `defaults.jl:167`; `m1002.jl:409-426,1195-1290` | small at current rates | S | Add options |
| 13 | Low | Unpublished FRBNY inputs (SPF short-run inflation obs, anticipated markup shocks) | – | Sep-2026 post; dev commits not public | unknown | – | Track; cannot replicate exactly |

## 8. Verified to match

* ss10 Γ0/Γ1/C/Ψ/Π, TTT/RRR/CCC, ZZ/DD/QQ/EE and the steady state vs DSGE.jl master. Cases: posterior median at n_ant 6 and 0; defaults at n_ant 6.
* Prior families and hyperparameters, log-densities, fixed/free flags, bounds.
* The ss10 code path in DSGE.jl master equals v1.3.0 except in gated (≥ss59) blocks, so the existing v1.3.0 oracles are still valid.
* Core PCE/GDP deflator observables; longinflation transform; spread series choice after 2023Q4; cond measurement-error σ values in the stub.

## 9. Could not verify

* Whether the 2026 releases use SMC or MH, and the FRBNY posterior draws (not published).
* The exact form of production SPF short-run inflation conditioning and the anticipated markup shocks (unpublished).
* Pre-2021Q4 OIS-based expected FFR data (internal).
* BofA spread for 2016Q4–2023Q3 (FRED only from 2023Q4 in our snapshot).
* An end-to-end ss104 forecast in DSGE.jl (would need the FRBNY data vintage and posterior).
* Fed release vintage dates for every series.

## 10. Suggested PR sequencing

1. Docs + hygiene (#9, #8, #11, #12). Small and safe.
2. Data: public SME expected-FFR loader + n_ant=6 config; SPF nowcast input and full conditioning (#5, #6).
3. meas_π AR(1) behind a subspec gate, with parity vs DSGE.jl ss87 (#4).
4. Regime-switching infrastructure (gensys2, regime-indexed system, Kalman/forecast), with parity vs DSGE.jl on a toy regime setup (#1).
5. COVID shocks (#3), then flexible AIT/tvcred/two-rule (#2). Do a full ss104 parity check vs DSGE.jl at fixed parameters (the Julia harness in `/workspace/nydsge-scan-ref/parity` can be reused).
6. SMC sampler and ss104 re-estimation (#7); re-run the toggle attribution against the latest release.
