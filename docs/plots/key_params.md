# Saved parameter comparison

ss104: `data/estimates/m1002_ss104_mode.json`; status `time_limit`; best finite estimate, not a certified converged mode.

ss10: `data/estimates/m1002_ss10_c1c2_mode_maskcovid.json`; status `converged`; recommended COVID-masked C1/C2 estimate used in both forecast plots.

ss104 NY Fed/default line uses DSGE.jl c09dfe8 ss104 defaults, the published code's calibration rather than the FRBNY production posterior.

| Parameter | ss104 saved estimate | ss104 code defaults | ss10 C1/C2 | Units / status |
|---|---:|---:|---:|---|
| h (Habit persistence) | 0.678461 | 0.534700 | 0.302293 | Dimensionless; estimated |
| zeta_p (Price rigidity) | 0.959533 | 0.894000 | 0.943120 | Dimensionless; estimated |
| zeta_w (Wage rigidity) | 0.957852 | 0.929100 | 0.907425 | Dimensionless; estimated |
| psi1 (Taylor inflation response) | 1.165750 | 1.367900 | 1.413684 | Dimensionless; estimated |
| psi2 (Taylor output response) | 0.033391 | 0.038800 | 0.087159 | Dimensionless; estimated |
| rho (Taylor policy smoothing) | 0.647267 | 0.712600 | 0.768775 | Dimensionless; estimated |
| sigma_ait_r_m1@2 | 0.139849 | 0.200000 | N/A | Quarterly model-unit shock SD; estimated parameter regime 2 |
| sigma_ait_r_m2@2 | 0.141078 | 0.200000 | N/A | Quarterly model-unit shock SD; estimated parameter regime 2 |
| sigma_ait_r_m3@2 | 0.140569 | 0.200000 | N/A | Quarterly model-unit shock SD; estimated parameter regime 2 |
| sigma_ait_r_m4@2 | 0.146710 | 0.200000 | N/A | Quarterly model-unit shock SD; estimated parameter regime 2 |
| sigma_ait_r_m5@2 | 0.156165 | 0.200000 | N/A | Quarterly model-unit shock SD; estimated parameter regime 2 |
| sigma_ait_r_m6@2 | 0.173246 | 0.200000 | N/A | Quarterly model-unit shock SD; estimated parameter regime 2 |

AIT policy coefficients and credibility are calibrated, not estimated. The estimated `rho` above is the historical Taylor-rule coefficient; the flexible AIT rule uses its own smoothing setting.

| Fixed ss104 policy setting | Value | Units |
|---|---:|---|
| flexible_ait_phi_pi | 4.000000 | Dimensionless |
| flexible_ait_phi_y | 3.000000 | Dimensionless |
| flexible_ait_rho_smooth | 0.900000 | Dimensionless |
| ait_Thalf | 10.000000 | Quarters |
| gdp_Thalf | 10.000000 | Quarters |
| AIT credibility | 0 to 1, 2020Q4–2026Q4 | Weight; 1/24 increase per quarter |

`rho_ait_rm` = 0.2135 and `sigma_ait_rm@2` = 0.238 remain at defaults because they have no defined prior; they were not estimated. ss10 has no AIT block.
