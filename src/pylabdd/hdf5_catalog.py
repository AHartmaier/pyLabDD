from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from matplotlib import pyplot as plt


def _json_value(value: Any) -> Any:
    """Convert HDF5/NumPy values to JSON-compatible Python values."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _attrs(obj: h5py.Group | h5py.Dataset | h5py.File) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in obj.attrs.items()}


def _shape_json(shape: tuple[int, ...]) -> str:
    return json.dumps(list(shape), separators=(",", ":"))


def _filename_parameters(path: Path) -> dict[str, int | float]:
    parameters: dict[str, int | float] = {}
    for key, value in re.findall(r"(?:^|_)(tau|T|H|D|L)(-?\d+(?:\.\d+)?)", path.stem):
        number = float(value)
        parameters[key] = int(number) if number.is_integer() else number
    return parameters


def inspect_hdf5(path: str | Path) -> dict[str, Any]:
    """Return all attributes and structural metadata without loading datasets."""
    file_path = Path(path).resolve()
    result: dict[str, Any] = {
        "path": str(file_path),
        "name": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "modified_ns": file_path.stat().st_mtime_ns,
        "filename_parameters": _filename_parameters(file_path),
        "attributes": {},
        "datasets": {},
        "groups": [],
    }

    with h5py.File(file_path, "r") as h5:
        result["attributes"]["/"] = _attrs(h5)

        def inspect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            object_path = f"/{name}"
            result["attributes"][object_path] = _attrs(obj)
            if isinstance(obj, h5py.Group):
                result["groups"].append(object_path)
                return
            result["datasets"][object_path] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "compression": obj.compression,
                "chunks": list(obj.chunks) if obj.chunks else None,
            }

        h5.visititems(inspect)

    return result


class HDF5Catalog:
    """Index, search, load, and plot collections of HDF5 result files."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> HDF5Catalog:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                filename_parameters TEXT NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS attributes (
                file_path TEXT NOT NULL,
                object_path TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                value_text TEXT,
                value_num REAL,
                PRIMARY KEY (file_path, object_path, key),
                FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS datasets (
                file_path TEXT NOT NULL,
                object_path TEXT NOT NULL,
                shape TEXT NOT NULL,
                dtype TEXT NOT NULL,
                compression TEXT,
                chunks TEXT,
                field_names TEXT,
                PRIMARY KEY (file_path, object_path),
                FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS attribute_lookup
                ON attributes(object_path, key, value_num, value_text);
            CREATE INDEX IF NOT EXISTS dataset_lookup ON datasets(object_path);
            """
        )

    @staticmethod
    def find_files(directory: str | Path, recursive: bool = True) -> list[Path]:
        root = Path(directory)
        pattern = "**/*.h5" if recursive else "*.h5"
        return sorted(root.glob(pattern))

    def scan(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        replace: bool = True,
    ) -> list[dict[str, Any]]:
        """Inspect HDF5 files, return their metadata, and update the database."""
        records: list[dict[str, Any]] = []
        if replace:
            self.connection.execute("DELETE FROM attributes")
            self.connection.execute("DELETE FROM datasets")
            self.connection.execute("DELETE FROM files")

        for path in self.find_files(directory, recursive):
            try:
                record = inspect_hdf5(path)
                self._store(record)
            except (OSError, ValueError) as exc:
                stat = path.stat()
                record = {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                    "filename_parameters": _filename_parameters(path),
                    "attributes": {},
                    "datasets": {},
                    "groups": [],
                    "error": str(exc),
                }
                self._store(record)
            records.append(record)
        self.connection.commit()
        return records

    def _store(self, record: Mapping[str, Any]) -> None:
        file_path = str(record["path"])
        self.connection.execute("DELETE FROM attributes WHERE file_path = ?", (file_path,))
        self.connection.execute("DELETE FROM datasets WHERE file_path = ?", (file_path,))
        self.connection.execute(
            """
            INSERT OR REPLACE INTO files
                (path, name, size_bytes, modified_ns, filename_parameters, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                record["name"],
                record["size_bytes"],
                record["modified_ns"],
                json.dumps(record["filename_parameters"]),
                record.get("error"),
            ),
        )

        attributes = record.get("attributes", {})
        for object_path, object_attrs in attributes.items():
            for key, value in object_attrs.items():
                numeric = float(value) if isinstance(value, (int, float)) else None
                text = value if isinstance(value, str) else None
                self.connection.execute(
                    "INSERT INTO attributes VALUES (?, ?, ?, ?, ?, ?)",
                    (file_path, object_path, key, json.dumps(value), text, numeric),
                )

        for object_path, dataset in record.get("datasets", {}).items():
            parent = str(Path(object_path).parent)
            field_names = attributes.get(parent, {}).get("field_names")
            self.connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    file_path,
                    object_path,
                    _shape_json(tuple(dataset["shape"])),
                    dataset["dtype"],
                    dataset["compression"],
                    json.dumps(dataset["chunks"]),
                    json.dumps(field_names),
                ),
            )

    def metadata_dict(self) -> dict[str, dict[str, Any]]:
        """Return the indexed metadata as a dictionary keyed by absolute path."""
        records: dict[str, dict[str, Any]] = {}
        for row in self.connection.execute("SELECT * FROM files ORDER BY path"):
            records[row["path"]] = {
                "path": row["path"],
                "name": row["name"],
                "size_bytes": row["size_bytes"],
                "modified_ns": row["modified_ns"],
                "filename_parameters": json.loads(row["filename_parameters"]),
                "attributes": {},
                "datasets": {},
                "error": row["error"],
            }
        for row in self.connection.execute("SELECT * FROM attributes ORDER BY file_path, object_path, key"):
            records[row["file_path"]]["attributes"].setdefault(row["object_path"], {})[
                row["key"]
            ] = json.loads(row["value_json"])
        for row in self.connection.execute("SELECT * FROM datasets ORDER BY file_path, object_path"):
            records[row["file_path"]]["datasets"][row["object_path"]] = {
                "shape": json.loads(row["shape"]),
                "dtype": row["dtype"],
                "compression": row["compression"],
                "chunks": json.loads(row["chunks"]),
                "field_names": json.loads(row["field_names"]),
            }
        return records

    def query(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find files by `/metadata` attributes and optional dataset/field name."""
        sql = "SELECT DISTINCT f.* FROM files AS f"
        join_params: list[Any] = []
        where_params: list[Any] = []
        where = ["f.error IS NULL"]
        for index, (key, value) in enumerate((filters or {}).items()):
            alias = f"a{index}"
            sql += (
                f" JOIN attributes AS {alias} ON {alias}.file_path = f.path"
                f" AND {alias}.object_path = '/metadata' AND {alias}.key = ?"
            )
            join_params.append(key)
            if isinstance(value, (int, float)):
                where.append(f"{alias}.value_num = ?")
                where_params.append(float(value))
            else:
                where.append(f"{alias}.value_text = ?")
                where_params.append(str(value))
        if dataset:
            sql += " JOIN datasets AS d ON d.file_path = f.path"
            requested = dataset.strip("/")
            if "/" in requested:
                group, field = requested.split("/", 1)
                where.append(
                    "(d.object_path = ? OR (d.object_path = ? AND d.field_names LIKE ?))"
                )
                where_params.extend((f"/{requested}", f"/{group}/fields", f'%"{field}"%'))
            else:
                where.append("(d.object_path = ? OR d.field_names LIKE ?)")
                where_params.extend(("/" + requested, f'%"{requested}"%'))
        sql += " WHERE " + " AND ".join(where) + " ORDER BY f.path"
        return [dict(row) for row in self.connection.execute(sql, join_params + where_params)]

    def sql(self, statement: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run a read-only-style SQL query against the catalog."""
        return [dict(row) for row in self.connection.execute(statement, parameters)]

    @staticmethod
    def _resolve_dataset(h5: h5py.File, dataset: str) -> tuple[h5py.Dataset, int | None]:
        requested = dataset.strip("/")
        if requested in h5 and isinstance(h5[requested], h5py.Dataset):
            return h5[requested], None
        parts = requested.split("/")
        if len(parts) == 2 and parts[0] in h5:
            group = h5[parts[0]]
            if isinstance(group, h5py.Group) and "fields" in group:
                names = [_json_value(value) for value in group.attrs.get("field_names", [])]
                if parts[1] in names:
                    return group["fields"], names.index(parts[1])
        matches: list[tuple[h5py.Dataset, int]] = []
        for group_name in ("global", "gb", "pileup"):
            if group_name not in h5:
                continue
            group = h5[group_name]
            names = [_json_value(value) for value in group.attrs.get("field_names", [])]
            if requested in names:
                matches.append((group["fields"], names.index(requested)))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous field {dataset!r}; use 'group/{dataset}'.")
        raise KeyError(f"Dataset or named field {dataset!r} not found.")

    @classmethod
    def load_dataset(cls, file: str | Path, dataset: str) -> np.ndarray:
        """Load a direct dataset or packed named field such as `global/psr_av`."""
        with h5py.File(file, "r") as h5:
            source, field_index = cls._resolve_dataset(h5, dataset)
            if field_index is None:
                return source[()]
            if source.ndim == 2:
                return source[:, field_index]
            if source.ndim == 3:
                return source[:, field_index, :]
            raise ValueError(f"Cannot select a named field from shape {source.shape}.")

    @classmethod
    def plot_dataset(
        cls,
        file: str | Path,
        dataset: str,
        *,
        nplot: int = 10,
        ax: Any = None,
    ) -> Any:
        """Plot a one-dimensional series or frames of a two-dimensional field."""
        path = Path(file)
        values = cls.load_dataset(path, dataset)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6), dpi=150)
        with h5py.File(path, "r") as h5:
            time = h5["time"][()] if "time" in h5 else None
            xval = h5["xval"][()] if "xval" in h5 else None

        if values.ndim == 1:
            x = time if time is not None and len(time) == len(values) else np.arange(len(values))
            ax.plot(x, values, label=path.stem)
            ax.set_xlabel("time" if x is time else "index")
        elif values.ndim == 2:
            step = max(1, len(values) // nplot)
            x = xval if xval is not None and len(xval) == values.shape[1] else np.arange(values.shape[1])
            for frame in range(0, len(values), step):
                label = f"t={time[frame]:g}" if time is not None else f"frame={frame}"
                ax.plot(x, values[frame], color=plt.cm.viridis(frame / max(1, len(values) - 1)), label=label)
            ax.set_xlabel("x" if x is xval else "index")
        else:
            raise ValueError("Plot a named field from multi-dimensional packed datasets.")
        ax.set_ylabel(dataset)
        ax.legend()
        return ax

    def plot_group(
        self,
        dataset: str,
        filters: Mapping[str, Any] | None = None,
        *,
        label_by: Sequence[str] = ("temperature", "grain_size", "tau0", "length_grain_boundary"),
        ax: Any = None,
    ) -> Any:
        """Plot a one-dimensional named field for all files matching a query."""
        files = self.query(filters, dataset=dataset)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6), dpi=150)
        metadata = self.metadata_dict()
        for item in files:
            path = item["path"]
            values = self.load_dataset(path, dataset)
            if values.ndim != 1:
                raise ValueError("Group plots require a one-dimensional dataset or named field.")
            with h5py.File(path, "r") as h5:
                time = h5["time"][()] if "time" in h5 else np.arange(len(values))
            meta = metadata[path]["attributes"].get("/metadata", {})
            label = ", ".join(f"{key}={meta[key]}" for key in label_by if key in meta)
            ax.plot(time, values, label=label or Path(path).stem)
        ax.set_xlabel("time")
        ax.set_ylabel(dataset)
        if files:
            ax.legend()
        return ax


def _parse_filters(items: Iterable[str]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Filter must have KEY=VALUE form: {item!r}")
        try:
            filters[key] = json.loads(value)
        except json.JSONDecodeError:
            filters[key] = value
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Index, search, and plot HDF5 result files.")
    parser.add_argument("--database", default="hdf5_catalog.sqlite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser("index")
    index.add_argument("directory")
    query = subparsers.add_parser("query")
    query.add_argument("filters", nargs="*", help="metadata filters in KEY=VALUE form")
    query.add_argument("--dataset")
    plot = subparsers.add_parser("plot")
    plot.add_argument("dataset")
    plot.add_argument("filters", nargs="*", help="metadata filters in KEY=VALUE form")
    plot.add_argument("--output")
    args = parser.parse_args()

    with HDF5Catalog(args.database) as catalog:
        if args.command == "index":
            records = catalog.scan(args.directory)
            print(f"Indexed {len(records)} HDF5 files in {args.database}")
        elif args.command == "query":
            print(json.dumps(catalog.query(_parse_filters(args.filters), dataset=args.dataset), indent=2))
        else:
            catalog.plot_group(args.dataset, _parse_filters(args.filters))
            if args.output:
                plt.savefig(args.output, dpi=300, bbox_inches="tight")
            else:
                plt.show()


if __name__ == "__main__":
    main()
