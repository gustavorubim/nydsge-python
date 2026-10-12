# ss104 port checklist

Port of NY Fed DSGE.jl Model1002 subspec ss104 into nydsge-python. Spec: docs/ss104_fidelity_scan.md.
Rules: branch ss104-port only, no force-push, main untouched, no long MCMC before M7.

- [ ] M1: regime-switching solve (gensys per regime + gensys2 for temporary ZLB/alt policy), regime-aware Kalman filter/smoother, time-varying information sets
- [ ] M2: COVID shocks with regime volatilities
- [ ] M3: AR(1) inflation measurement error
- [ ] M4: flexible AIT, pgap/ygap, time-varying credibility 2020Q4-2026Q4
- [ ] M5: SME/SPD expected FFR observables 1-6Q
- [ ] M6: full ss104 parity + conditional forecast
- [ ] M7: final estimation + NY Fed forecast comparison doc

## Progress log
