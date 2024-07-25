import json
import shutil
import subprocess
from pathlib import Path


def get_bin(prog_name: str) -> Path:
    match_which = shutil.which(prog_name)
    if match_which is None:
        raise FileNotFoundError(f"{prog_name} not found in PATH")
    bin_prog = Path(match_which)

    return bin_prog


def get_bin_yosys() -> Path:
    return get_bin("yosys")


def get_bin_iverilog() -> Path:
    return get_bin("iverilog")


def get_bin_vvp() -> Path:
    return get_bin("vvp")


def check_process_output(process: subprocess.CompletedProcess, extra_message: str | None = None) -> None:
    if process.returncode != 0:
        error_str = ""
        error_str += f"Process returned non-zero exit code: {process.returncode}\n"
        error_str += f"cmd: {process.args}\n"
        if extra_message is not None:
            error_str += f"{extra_message}\n"
        error_str += f"stdout: {process.stdout}\n"
        error_str += f"stderr: {process.stderr}"
        raise RuntimeError(
            error_str,
        )


class EmptyDesignError(ValueError):
    def __init__(self, design_name: str) -> None:
        self.design_name = design_name

    def __str__(self) -> str:
        return f"Design {self.design_name} does not contain any cells and any wires after synthesis"


def validate_design_with_yosys(
    design_dir: Path,
    design_name: str,
    top_name: str,
) -> None:
    bin_yosys = get_bin_yosys()

    script_yosys = ""
    script_yosys += f"read_verilog {design_name}.v\n"
    script_yosys += f"hierarchy -check -top {top_name}\n"
    script_yosys += "synth\n"
    script_yosys += f"tee -o validate__stat.json stat -json -top {top_name}\n"

    script_fp = design_dir / "validate.ys"
    script_fp.write_text(script_yosys)

    p_args: list = [
        bin_yosys,
        "-q",
        "-s",
        script_fp.name,
    ]

    p = subprocess.run(
        p_args,
        cwd=design_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    check_process_output(p, extra_message=f"{script_fp=}")

    data_stat = json.loads((design_dir / "validate__stat.json").read_text())

    data_design = data_stat["design"]
    data_num_cells = data_design["num_cells"]
    data_num_wires = data_design["num_wires"]

    if data_num_cells == 0 and data_num_wires == 0:
        raise EmptyDesignError(design_name)


def validate_design_with_iverilog(
    design_dir: Path,
    design_name: str,
    top_name: str,
    tb_str: str,
    cleanup: bool = True,
    print_output: bool = False,
):
    bin_iverilog = get_bin_iverilog()
    bin_vvp = get_bin_vvp()

    v_fp = design_dir / f"{design_name}.v"

    tb_fp = design_dir / f"{design_name}_tb.v"
    if tb_fp.exists():
        tb_fp.unlink()
    tb_fp.write_text(tb_str)

    vvp_output_fp = design_dir / f"{design_name}.vvp"

    p_args: list = [
        bin_iverilog,
        "-o",
        vvp_output_fp.name,
        v_fp.name,
        tb_fp.name,
    ]

    p = subprocess.run(
        p_args,
        cwd=design_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    check_process_output(p, extra_message=f"{tb_fp=}")

    p_args = [bin_vvp, vvp_output_fp.name]

    p = subprocess.run(
        p_args,
        cwd=design_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    check_process_output(p, extra_message=f"{tb_fp=}")

    if print_output:
        print(p.stdout)  # noqa: T201

    if "ERROR" in p.stdout:
        raise RuntimeError("Simulation failed")

    if cleanup:
        vvp_output_fp.unlink()
