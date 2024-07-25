import base64
import hashlib
import json
from pathlib import Path

import pydantic
import requests
from github import Github


def get_file_from_github(
    gh_api: Github,
    owner: str,
    repo: str,
    path: str,
    timeout: int | None = None,
) -> str:
    repo_ = gh_api.get_repo(f"{owner}/{repo}")
    data = repo_.get_contents(path)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"Expected 1 file, found {len(data)}")
        data = data[0]

    content = data.content
    if not content:
        download_url = data.download_url
        r = requests.get(download_url, timeout=timeout)
        if r.status_code != requests.codes.ok:
            raise RuntimeError(
                f"Failed to make request: {r.status_code}\n{r.text}\n{r.headers}",
            )
        return r.content.decode("utf-8")
    return base64.b64decode(content).decode("utf-8")


class RegexEntry(pydantic.BaseModel):
    regex: str
    hash: str
    source: str


class RegexData(pydantic.BaseModel):
    regexes: list[RegexEntry]


def update_data_file(data_file: Path, entries: list[RegexEntry]) -> None:
    if data_file.exists():
        # read into model and append new entries
        model = RegexData.model_validate_json(data_file.read_text())
        model.regexes.extend(entries)
        data_file.write_text(model.model_dump_json(indent=4))
    else:
        model = RegexData(regexes=entries)
        data_file.write_text(model.model_dump_json(indent=4))


def get_from_deep_regex(data_file: Path, dataset_name: str) -> None:
    data_txt = get_file_from_github(
        gh_api=Github(),
        owner="nicholaslocascio",
        repo="deep-regex",
        path=f"datasets/{dataset_name}/targ.txt",
    )
    lines = data_txt.strip().splitlines()
    entries = []
    for line in lines:
        regex_str = line.strip()
        regex_hash = hashlib.sha256(regex_str.encode()).hexdigest()
        source = dataset_name
        entries.append(RegexEntry(regex=regex_str, hash=regex_hash, source=source))

    update_data_file(data_file, entries)


def get_kb13(data_file: Path) -> None:
    get_from_deep_regex(data_file, "KB13")


def get_nl_rx_synth(data_file: Path) -> None:
    get_from_deep_regex(data_file, "NL-RX-Synth")


def get_nl_rx_turk(data_file: Path) -> None:
    get_from_deep_regex(data_file, "NL-RX-Turk")


def get_regexeval_regexlib(data_file: Path) -> None:
    data_txt = get_file_from_github(
        gh_api=Github(),
        owner="s2e-lab",
        repo="RegexEval",
        path="DatasetCollection/RegexEval.json",
    )
    data_json = json.loads(data_txt)
    entries = []
    for entry in data_json:
        regex_str = entry["expression"]
        regex_hash = hashlib.sha256(regex_str.encode()).hexdigest()
        source = "regexeval_regexlib"
        entries.append(RegexEntry(regex=regex_str, hash=regex_hash, source=source))
    update_data_file(data_file, entries)


CURRENT_DIR = Path(__file__).parent
DATA_FILE = CURRENT_DIR / "regex_data.json"

if __name__ == "__main__":
    if DATA_FILE.exists():
        DATA_FILE.unlink()

    get_kb13(DATA_FILE)
    get_nl_rx_synth(DATA_FILE)
    get_nl_rx_turk(DATA_FILE)
    get_regexeval_regexlib(DATA_FILE)
