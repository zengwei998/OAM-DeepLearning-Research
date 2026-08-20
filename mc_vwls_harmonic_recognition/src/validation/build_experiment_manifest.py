"""
Build a reproducible experiment-file manifest.

Scanned directories:
    data/generated/
    data/manifest/
    results/csv/
    results/tables/
    results/figures/
    results/validation/

Outputs:
    results/validation/experiment_manifest.csv
    results/validation/experiment_manifest_report.txt

Recorded fields:
    relative path
    file size
    modification time
    SHA-256
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRECTORIES = (
    ROOT / "data" / "generated",
    ROOT / "data" / "manifest",
    ROOT / "results" / "csv",
    ROOT / "results" / "tables",
    ROOT / "results" / "figures",
    ROOT / "results" / "validation",
)

MANIFEST_CSV_PATH = (
    ROOT
    / "results"
    / "validation"
    / "experiment_manifest.csv"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "validation"
    / "experiment_manifest_report.txt"
)

HASH_BLOCK_SIZE = 8 * 1024 * 1024

EXCLUDED_OUTPUT_PATHS = {
    MANIFEST_CSV_PATH.resolve(),
    REPORT_PATH.resolve(),
}


def calculate_sha256(
    path: Path,
) -> str:
    """
    Calculate the SHA-256 digest of one file.
    """

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file_handle:
        while True:
            block = file_handle.read(
                HASH_BLOCK_SIZE
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def format_size(
    size_bytes: int,
) -> str:
    """
    Format a byte count using binary units.
    """

    size = float(
        size_bytes
    )

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    unit_index = 0

    while (
        size >= 1024.0
        and unit_index
        < len(units) - 1
    ):
        size /= 1024.0
        unit_index += 1

    return (
        f"{size:.3f} "
        f"{units[unit_index]}"
    )


def discover_files(
    directories: Iterable[Path],
) -> List[Path]:
    """
    Recursively discover all regular files in the requested directories.
    """

    discovered_files: List[
        Path
    ] = []

    for directory in directories:
        if not directory.exists():
            raise FileNotFoundError(
                f"Scan directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Scan path is not a directory: {directory}"
            )

        for path in directory.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            resolved_path = path.resolve()

            if resolved_path in EXCLUDED_OUTPUT_PATHS:
                continue

            discovered_files.append(
                path
            )

    unique_files = sorted(
        {
            path.resolve()
            for path in discovered_files
        },
        key=lambda item: (
            item.relative_to(
                ROOT.resolve()
            ).as_posix().lower()
        ),
    )

    return [
        Path(path)
        for path in unique_files
    ]


def make_manifest_row(
    path: Path,
) -> Dict[str, object]:
    """
    Build one manifest row.
    """

    resolved_path = path.resolve()

    relative_path = resolved_path.relative_to(
        ROOT.resolve()
    )

    stat = resolved_path.stat()

    modified_datetime = datetime.fromtimestamp(
        stat.st_mtime
    )

    sha256_digest = calculate_sha256(
        resolved_path
    )

    return {
        "relative_path": relative_path.as_posix(),
        "file_name": resolved_path.name,
        "suffix": resolved_path.suffix.lower(),
        "size_bytes": int(
            stat.st_size
        ),
        "size_human": format_size(
            stat.st_size
        ),
        "modified_local_time": modified_datetime.isoformat(
            timespec="seconds"
        ),
        "sha256": sha256_digest,
    }


def save_manifest_csv(
    rows: List[Dict[str, object]],
) -> None:
    """
    Save the manifest table.
    """

    if not rows:
        raise ValueError(
            "No manifest rows were generated."
        )

    MANIFEST_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with MANIFEST_CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def summarize_by_directory(
    rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, int]]:
    """
    Aggregate file count and total size by top-level scanned directory.
    """

    summary: Dict[
        str,
        Dict[str, int],
    ] = {}

    for row in rows:
        relative_path = Path(
            str(
                row["relative_path"]
            )
        )

        parts = (
            relative_path.parts
        )

        if len(parts) < 2:
            category = str(
                relative_path
            )
        else:
            category = (
                f"{parts[0]}/"
                f"{parts[1]}"
            )

        if category not in summary:
            summary[category] = {
                "file_count": 0,
                "size_bytes": 0,
            }

        summary[
            category
        ][
            "file_count"
        ] += 1

        summary[
            category
        ][
            "size_bytes"
        ] += int(
            row["size_bytes"]
        )

    return summary


def summarize_by_suffix(
    rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, int]]:
    """
    Aggregate file count and total size by suffix.
    """

    summary: Dict[
        str,
        Dict[str, int],
    ] = {}

    for row in rows:
        suffix = str(
            row["suffix"]
        )

        if suffix == "":
            suffix = "[no suffix]"

        if suffix not in summary:
            summary[suffix] = {
                "file_count": 0,
                "size_bytes": 0,
            }

        summary[
            suffix
        ][
            "file_count"
        ] += 1

        summary[
            suffix
        ][
            "size_bytes"
        ] += int(
            row["size_bytes"]
        )

    return summary


def build_report(
    rows: List[Dict[str, object]],
) -> str:
    """
    Build a human-readable manifest report.
    """

    total_size_bytes = int(
        sum(
            int(
                row["size_bytes"]
            )
            for row in rows
        )
    )

    directory_summary = summarize_by_directory(
        rows
    )

    suffix_summary = summarize_by_suffix(
        rows
    )

    largest_files = sorted(
        rows,
        key=lambda row: int(
            row["size_bytes"]
        ),
        reverse=True,
    )[:20]

    lines = [
        "Experiment manifest report",
        f"Project root: {ROOT}",
        (
            "Manifest generated at local time: "
            f"{datetime.now().isoformat(timespec='seconds')}"
        ),
        f"Total files: {len(rows)}",
        (
            "Total size: "
            f"{total_size_bytes} bytes "
            f"({format_size(total_size_bytes)})"
        ),
        "",
        "[SCANNED DIRECTORIES]",
    ]

    for directory in SCAN_DIRECTORIES:
        lines.append(
            str(
                directory.relative_to(
                    ROOT
                )
            )
        )

    lines.extend(
        [
            "",
            "[SUMMARY BY DIRECTORY]",
        ]
    )

    for category in sorted(
        directory_summary
    ):
        values = directory_summary[
            category
        ]

        lines.append(
            (
                f"{category}: "
                f"files={values['file_count']}, "
                f"size={values['size_bytes']} bytes "
                f"({format_size(values['size_bytes'])})"
            )
        )

    lines.extend(
        [
            "",
            "[SUMMARY BY FILE TYPE]",
        ]
    )

    for suffix in sorted(
        suffix_summary
    ):
        values = suffix_summary[
            suffix
        ]

        lines.append(
            (
                f"{suffix}: "
                f"files={values['file_count']}, "
                f"size={values['size_bytes']} bytes "
                f"({format_size(values['size_bytes'])})"
            )
        )

    lines.extend(
        [
            "",
            "[20 LARGEST FILES]",
        ]
    )

    for rank, row in enumerate(
        largest_files,
        start=1,
    ):
        lines.append(
            (
                f"{rank:02d}. "
                f"{row['relative_path']} | "
                f"{row['size_bytes']} bytes | "
                f"{row['size_human']} | "
                f"SHA256={row['sha256']}"
            )
        )

    lines.extend(
        [
            "",
            "[OUTPUTS]",
            f"Manifest CSV: {MANIFEST_CSV_PATH}",
            f"Report: {REPORT_PATH}",
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:
    print("=" * 78)
    print("BUILD EXPERIMENT MANIFEST")
    print("=" * 78)

    print("Scan directories:")

    for directory in SCAN_DIRECTORIES:
        print(
            " ",
            directory,
        )

    files = discover_files(
        SCAN_DIRECTORIES
    )

    print("")
    print(
        "Files discovered:",
        len(files),
    )

    manifest_rows: List[
        Dict[str, object]
    ] = []

    total_files = len(
        files
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        print(
            f"[{index:03d}/{total_files:03d}] "
            f"Hashing: "
            f"{path.relative_to(ROOT)}"
        )

        row = make_manifest_row(
            path
        )

        manifest_rows.append(
            row
        )

    save_manifest_csv(
        manifest_rows
    )

    report_text = build_report(
        manifest_rows
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print("")
    print(
        report_text
    )

    print("")
    print("=" * 78)
    print("EXPERIMENT MANIFEST COMPLETE")
    print("=" * 78)

    print(
        "Manifest:",
        MANIFEST_CSV_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()