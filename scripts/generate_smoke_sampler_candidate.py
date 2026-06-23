from pathlib import Path

from nydsge.estimate import save_sampler_result
from nydsge.vv import load_sampler_fixture_result

repo = Path(__file__).resolve().parents[1]
oracle = repo / "tests/fixtures/smoke/oracle_sampler/m1002_ss10_sampler.h5"
candidate = repo / "tests/fixtures/smoke/candidate/sampler.npz"
result = load_sampler_fixture_result(oracle)
save_sampler_result(result, candidate)
print(candidate)
