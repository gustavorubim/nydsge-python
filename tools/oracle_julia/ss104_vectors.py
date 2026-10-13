"""Deterministic ss104 vectors and lossless compaction of source Julia exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from nydsge.models import Model1002
from nydsge.ss104_estimation import (
    ss104_estimation_bounds,
    ss104_estimation_vector,
    ss104_parameter_catalog,
    update_ss104_parameters,
)


def parameter_vectors():
    for vector_id in range(1, 4):
        model = Model1002("ss104")
        catalog = ss104_parameter_catalog(model)
        x = ss104_estimation_vector(catalog)
        x += 1e-3 * np.sin(vector_id * np.arange(1, len(x) + 1))
        for i, (lower, upper) in enumerate(ss104_estimation_bounds(catalog)):
            x[i] = np.clip(
                x[i], -np.inf if lower is None else lower, np.inf if upper is None else upper
            )
        update_ss104_parameters(model, catalog, x)
        yield {
            "id": f"vector_{vector_id}",
            "recipe": "x[j] + 0.001*sin(vector_id*(j+1)), clipped to estimation bounds",
            "coordinates": list(catalog),
            "estimation_values": x.tolist(),
            "parameters": {k: p.value for k, p in ss104_parameter_catalog(model).items()},
        }


def compact_export(source, destination):
    """Keep all matrices/means; sample covariance periods 0, 2, 4 and the final one.

    Float64 values remain exact. Identical datasets share HDF5 hard links.
    The default fixture separately tests complete covariance histories.
    """
    with h5py.File(source) as src, h5py.File(destination, "w") as dst:
        seen = {}

        def copy(name, dataset):
            if not isinstance(dataset, h5py.Dataset):
                return
            values = dataset[...]
            if name.endswith("_covariances"):
                values = values[[0, 2, 4, len(values) - 1]]
            if name == "metadata/json":
                metadata = json.loads(dataset.asstr()[()])
                metadata["compact_covariance_periods"] = "0, 2, 4, final (zero-based)"
                values = json.dumps(metadata)
            parent, key = name.rsplit("/", 1)
            group = dst.require_group(parent)
            if name == "metadata/json":
                group.create_dataset(key, data=values, dtype=h5py.string_dtype("utf-8"))
            elif isinstance(values, np.ndarray) and values.dtype.kind != "O":
                identity = (
                    values.dtype.str,
                    values.shape,
                    hashlib.sha256(values.tobytes()).digest(),
                )
                if identity in seen:
                    group[key] = dst[seen[identity]]
                else:
                    group.create_dataset(key, data=values, compression="gzip", compression_opts=9)
                    seen[identity] = name
            else:
                group.create_dataset(key, data=values, dtype=dataset.dtype)

        src.visititems(copy)
        for label, count in (("history", 27), ("conditional", 28)):
            dst.create_dataset(f"workflow/{label}/covariance_periods", data=[0, 2, 4, count - 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    vectors = subparsers.add_parser("vectors")
    vectors.add_argument("directory", type=Path)
    compact = subparsers.add_parser("compact")
    compact.add_argument("source", type=Path)
    compact.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "vectors":
        args.directory.mkdir(parents=True, exist_ok=True)
        for vector in parameter_vectors():
            (args.directory / f"{vector['id']}.json").write_text(
                json.dumps(vector, indent=2) + "\n", encoding="utf-8"
            )
    else:
        compact_export(args.source, args.destination)
