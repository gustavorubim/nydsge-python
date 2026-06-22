from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimePurityFinding:
    path: Path
    line: int
    pattern: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(frozen=True)
class RuntimePurityReport:
    root: Path
    checked_files: int
    findings: tuple[RuntimePurityFinding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "checked_files": self.checked_files,
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
        }


PROHIBITED_RUNTIME_PATTERNS: tuple[str, ...] = (
    "julia" + "call",
    "py" + "julia",
    "import " + "julia",
    "from " + "julia",
    "julia" + ".install",
    "julia" + ".api",
    "julia" + ".exe",
    "subprocess" + ".run",
    "subprocess" + ".call",
    "subprocess" + ".Popen",
    "os" + ".system",
    "wsl" + ".exe",
    "bash" + ".exe",
)


def audit_runtime_purity(
    root: Path,
    *,
    patterns: tuple[str, ...] = PROHIBITED_RUNTIME_PATTERNS,
) -> RuntimePurityReport:
    runtime_root = root.resolve()
    if not runtime_root.exists():
        msg = f"Runtime purity root does not exist: {runtime_root}"
        raise FileNotFoundError(msg)
    if not runtime_root.is_dir():
        msg = f"Runtime purity root is not a directory: {runtime_root}"
        raise NotADirectoryError(msg)

    findings: list[RuntimePurityFinding] = []
    checked_files = 0
    display_root = runtime_root.parent if runtime_root.name == "nydsge" else runtime_root
    for path in sorted(runtime_root.rglob("*.py")):
        if not path.is_file():
            continue
        checked_files += 1
        relative_path = path.relative_to(display_root)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.casefold()
            for pattern in patterns:
                if pattern.casefold() in lowered:
                    findings.append(
                        RuntimePurityFinding(
                            path=relative_path,
                            line=line_number,
                            pattern=pattern,
                            text=line.strip(),
                        )
                    )
    return RuntimePurityReport(
        root=runtime_root,
        checked_files=checked_files,
        findings=tuple(findings),
    )
