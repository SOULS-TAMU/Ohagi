"""Validate release metadata without importing or changing numerical implementations."""

import argparse
import ast
from datetime import date
from email.parser import BytesParser
import json
from pathlib import Path
import re
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def field(text, pattern, label):
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    require(match is not None, "Missing " + label)
    return match.group(1).strip().strip("\"'")


def runtime_versions(root):
    versions = set()
    for path in (root / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if any(
                    isinstance(target, ast.Name) and target.id == "__version__"
                    for target in node.targets
                ):
                    versions.add(node.value.value)
    return versions


def check_artifacts(directory, name, version):
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    require(
        len(wheels) == len(sources) == 1,
        "Expected one wheel and one source distribution",
    )
    require(
        len(list(directory.iterdir())) == 2,
        "Unexpected files in the publishing directory",
    )
    with zipfile.ZipFile(wheels[0]) as archive:
        paths = archive.namelist()
        metadata = next(path for path in paths if path.endswith(".dist-info/METADATA"))
        wheel = BytesParser().parsebytes(archive.read(metadata))
        require(
            any(path.endswith("/licenses/LICENSE") for path in paths),
            "Wheel lacks LICENSE",
        )
        require(
            any(path.endswith("/licenses/NOTICE") for path in paths),
            "Wheel lacks NOTICE",
        )
    with tarfile.open(sources[0]) as archive:
        metadata = min(
            (p for p in archive.getmembers() if p.name.endswith("/PKG-INFO")),
            key=lambda p: len(p.name),
        )
        source = BytesParser().parsebytes(archive.extractfile(metadata).read())
    for metadata in (wheel, source):
        require(
            metadata["Name"] == name,
            "Artifact distribution name differs from the intended project",
        )
        require(
            metadata["Version"] == version,
            "Artifact version differs from package metadata",
        )
        require(
            metadata["License-Expression"] == "Apache-2.0",
            "Artifact license differs from Apache-2.0",
        )
        require(
            metadata["Requires-Python"] == ">=3.10",
            "Artifact Python requirement differs",
        )
        require(
            not any(" @ " in entry for entry in metadata.get_all("Requires-Dist", [])),
            "Publishing metadata contains direct dependency URLs",
        )
    return [wheels[0].name, sources[0].name]


def check(root, tag=None, publishing=False, index="pypi", artifacts=None):
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]
    require(
        re.fullmatch(
            r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?", version
        ),
        "Use a documented release version",
    )
    require(
        runtime_versions(root) == {version},
        "Runtime __version__ values differ from pyproject.toml",
    )
    require(
        project["license"] == "Apache-2.0", "Project license differs from Apache-2.0"
    )
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    bib = (root / "CITATION.bib").read_text(encoding="utf-8")
    require(
        field(cff, r"^version:\s*(.+)$", "CFF version") == version,
        "CFF version differs",
    )
    require(
        field(bib, r"^\s*version\s*=\s*\{([^}]+)\}", "BibTeX version") == version,
        "BibTeX version differs",
    )
    changes = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    release_date = field(
        changes,
        r"^##\s+" + re.escape(version) + r"\s+—\s+(.+)$",
        "versioned changelog entry",
    )
    if release_date != "Unreleased":
        date.fromisoformat(release_date)
        require(
            field(cff, r"^date-released:\s*(.+)$", "CFF release date") == release_date,
            "Citation and changelog release dates differ",
        )
    if publishing:
        require(
            release_date != "Unreleased",
            "Set the changelog and citation release date before publishing",
        )
    if tag:
        require(tag == "v" + version, "Release tag differs from the package version")
    name = (
        "kkthn"
        if index == "testpypi" and project["name"] == "kkt-hardnet"
        else project["name"]
    )
    files = check_artifacts(artifacts, name, version) if artifacts else []
    return {
        "distribution": name,
        "version": version,
        "tag": "v" + version,
        "release_date": release_date,
        "artifacts": files,
    }


def ensure_unpublished(record, index):
    host = "test.pypi.org" if index == "testpypi" else "pypi.org"
    url = f"https://{host}/pypi/{record['distribution']}/{record['version']}/json"
    try:
        with urllib.request.urlopen(url, timeout=30):
            raise ValueError(
                "This version is already published; choose a new version instead of overwriting release files"
            )
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError(
                "Package-index version check failed: " + str(error.code)
            ) from None


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "src/sample").mkdir(parents=True)
        files = {
            "pyproject.toml": '[project]\nname="sample"\nversion="1.2.3"\nlicense="Apache-2.0"\n',
            "src/sample/__init__.py": '__version__="1.2.3"\n',
            "CITATION.cff": 'version: "1.2.3"\ndate-released: "2026-11-05"\n',
            "CITATION.bib": "@software{sample,\n version = {1.2.3}\n}\n",
            "CHANGELOG.md": "# Changelog\n\n## 1.2.3 — 2026-11-05\n",
        }
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")
        check(root, tag="v1.2.3", publishing=True)
        for name in ["src/sample/__init__.py", "CITATION.cff", "CITATION.bib"]:
            path = root / name
            path.write_text(files[name].replace("1.2.3", "9.9.9"), encoding="utf-8")
            try:
                check(root)
            except ValueError:
                pass
            else:
                raise RuntimeError("Release guard accepted inconsistent " + name)
            path.write_text(files[name], encoding="utf-8")
        for arguments in [{"tag": "v9.9.9"}, {"publishing": True}]:
            if "publishing" in arguments:
                (root / "CHANGELOG.md").write_text(
                    files["CHANGELOG.md"].replace("2026-11-05", "Unreleased"),
                    encoding="utf-8",
                )
            try:
                check(root, **arguments)
            except ValueError:
                pass
            else:
                raise RuntimeError("Release guard accepted inconsistent release state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tag")
    parser.add_argument("--publishing", action="store_true")
    parser.add_argument("--index", choices=["pypi", "testpypi"], default="pypi")
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    result = check(
        args.root.resolve(),
        tag=args.tag,
        publishing=args.publishing,
        index=args.index,
        artifacts=args.dist,
    )
    if args.publishing:
        ensure_unpublished(result, args.index)
    (args.root / "release-check-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
