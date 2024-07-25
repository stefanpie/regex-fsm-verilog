import hashlib
import logging
import shutil
import tarfile
from collections.abc import Generator
from pathlib import Path

import greenery
import networkx as nx
import tqdm
from joblib import Parallel, delayed

from core import (
    Timeout,
    can_parse_regex,
    fsm_to_verilog,
    gen_tb_str_verilog_fsm,
    regex_to_fsm,
)
from regex_data import RegexData, RegexEntry
from tools import validate_design_with_iverilog, validate_design_with_yosys


def process_single_regex_entry(regex_entry: RegexEntry, logger: logging.Logger) -> None:
    regex_str = regex_entry.regex
    logger.debug(f"Processing regex: {regex_str}")

    fsm_hash = hashlib.sha256(regex_str.encode()).hexdigest()
    fsm_dir = DIR_BUILDS / fsm_hash
    if fsm_dir.exists():
        shutil.rmtree(fsm_dir)
    fsm_dir.mkdir()

    logger.debug(f"Generating FSM for regex: {regex_str}")
    logger.debug(f"  - FSM hash: {fsm_hash}")
    logger.debug(f"  - FSM directory: {fsm_dir}")

    regex_pattern = greenery.parse(regex_str)
    fsm = regex_pattern.to_fsm()

    v_str, conversion_data = fsm_to_verilog(regex_str, regex_pattern, fsm)
    v_fp = fsm_dir / "fsm.v"
    v_fp.write_text(v_str)
    logger.debug(f"  - FSM module: {v_fp}")

    logger.debug("  - Validating FSM with Yosys")
    validate_design_with_yosys(fsm_dir, "fsm", "fsm")

    logger.debug("  - Validating FSM with Icarus Verilog")
    v_tb_str = gen_tb_str_verilog_fsm(
        conversion_data.fsm,
        conversion_data.alphabet_size,
        conversion_data.state_size,
        conversion_data.alphabet_encoding,
        conversion_data.state_encoding,
        n_valid_samples_attempt=100,
    )
    validate_design_with_iverilog(fsm_dir, "fsm", "fsm", v_tb_str)


def filter_duplicate_regexes(regex_entries: list[RegexEntry], logger: logging.Logger) -> list[RegexEntry]:
    logger.info("Filtering duplicate regexes")
    regex_entries_dedupe: list[RegexEntry] = []
    regex_entries_set = set()
    for entry in tqdm.tqdm(regex_entries):
        if entry.hash not in regex_entries_set:
            regex_entries_dedupe.append(entry)
            regex_entries_set.add(entry.hash)
    return regex_entries_dedupe


def filter_parseable_regex_entries(regex_entries: list[RegexEntry], logger: logging.Logger) -> list[RegexEntry]:
    logger.info("Filtering regexes that can be parsed")
    regex_entries_parseable = [entry for entry in tqdm.tqdm(regex_entries) if can_parse_regex(entry.regex)]
    return regex_entries_parseable


def filter_long_regexes(regex_entries: list[RegexEntry], logger: logging.Logger) -> list[RegexEntry]:
    logger.info("Filtering regexes length > 8")
    regex_entries_big = [entry for entry in tqdm.tqdm(regex_entries) if len(entry.regex) > 8]
    return regex_entries_big


def filter_regex_timeout(regex_entries: list[RegexEntry], logger: logging.Logger, n_jobs: int) -> list[RegexEntry]:
    logger.info("Filtering regexes that timeout")

    def check_timeout(entry: RegexEntry) -> bool:
        try:
            with Timeout(5):
                regex_to_fsm(entry.regex)
        except TimeoutError:
            return False
        return True

    regex_entries_no_timeout_flag = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(check_timeout)(entry) for entry in tqdm.tqdm(regex_entries)
    )
    regex_entries_no_timeout = [
        entry for entry, is_timeout in zip(regex_entries, regex_entries_no_timeout_flag, strict=False) if is_timeout
    ]

    return regex_entries_no_timeout


def filter_small_state_regex_entries(
    regex_entries: list[RegexEntry],
    logger: logging.Logger,
    n_jobs: int,
) -> list[RegexEntry]:
    logger.info("Filtering out regexes with a small number of states")

    def num_states(entry: RegexEntry) -> int:
        return len(regex_to_fsm(entry.regex).states)

    regex_entries_num_states: Generator[int] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(num_states)(entry) for entry in tqdm.tqdm(regex_entries)
    )  # type: ignore
    regex_entries_many_states = [
        entry for entry, num_states in zip(regex_entries, regex_entries_num_states, strict=False) if num_states > 8
    ]

    return regex_entries_many_states


def filter_duplicate_regex_by_fsm_graph_hash(
    regex_entries: list[RegexEntry],
    logger: logging.Logger,
    n_jobs: int,
) -> list[RegexEntry]:
    logger.info("Filtering out duplicate regexes via FSM graph hashing")

    def hash_fsm(entry: RegexEntry) -> str:
        fsm = regex_to_fsm(entry.regex)
        g_fsm = nx.DiGraph()
        for state in fsm.states:
            g_fsm.add_node(
                state,
            )
        for state, transitions in fsm.map.items():
            for charclass, next_state in transitions.items():
                g_fsm.add_edge(state, next_state, charclass=charclass)
        g_hash = nx.weisfeiler_lehman_graph_hash(g_fsm, iterations=64, digest_size=64)
        return g_hash

    graph_hashes: Generator[str] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(hash_fsm)(entry) for entry in tqdm.tqdm(regex_entries)
    )  # type: ignore
    regex_entries_unique_graphs: list[RegexEntry] = []
    regex_entries_unique_graphs_set: set[str] = set()
    for entry, graph_hash in zip(regex_entries, graph_hashes, strict=False):
        if graph_hash not in regex_entries_unique_graphs_set:
            regex_entries_unique_graphs.append(entry)
            regex_entries_unique_graphs_set.add(graph_hash)
    return regex_entries_unique_graphs


DIR_SCRIPT = Path(__file__).parent
DIR_BUILDS = DIR_SCRIPT / "generated_designs"

FP_REGEX_DATA = DIR_SCRIPT / "regex_data.json"

N_JOBS = 32

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    if not FP_REGEX_DATA.exists():
        raise FileNotFoundError(f"Regex data file not found: {FP_REGEX_DATA}")
    regex_data = RegexData.model_validate_json(FP_REGEX_DATA.read_text())

    regex_entries = regex_data.regexes

    regex_entries = filter_duplicate_regexes(regex_entries, logger)
    logger.info(f"Number of regexes: {len(regex_entries)}")

    regex_entries = filter_parseable_regex_entries(regex_entries, logger)
    logger.info(f"Number of parseable regexes: {len(regex_entries)}")

    regex_entries = filter_long_regexes(regex_entries, logger)
    logger.info(f"Number of long regexes: {len(regex_entries)}")

    regex_entries = filter_regex_timeout(regex_entries, logger, N_JOBS)
    logger.info(f"Number of regexes that don't timeout: {len(regex_entries)}")

    regex_entries = filter_small_state_regex_entries(regex_entries, logger, N_JOBS)
    logger.info(f"Number of regexes with many states: {len(regex_entries)}")

    regex_entries = filter_duplicate_regex_by_fsm_graph_hash(regex_entries, logger, N_JOBS)
    logger.info(f"Number of unique regexes via FSM graph hashing: {len(regex_entries)}")

    if DIR_BUILDS.exists():
        shutil.rmtree(DIR_BUILDS)
    DIR_BUILDS.mkdir()

    # Main loop to generate designs
    logger.info("Generating designs...")
    Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_single_regex_entry)(regex_entry, logger) for regex_entry in tqdm.tqdm(regex_entries)
    )

    # Compress all generated designs into a single archive
    archive_fp = DIR_SCRIPT / "generated_designs.tar.gz"
    if archive_fp.exists():
        archive_fp.unlink()
    archive = tarfile.open(archive_fp, "w:gz", compresslevel=9)
    archive.add(DIR_BUILDS, arcname="generated_designs")
    archive.close()
    logger.info(f"Generated designs archived to: {archive_fp}")
