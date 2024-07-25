import enum
import itertools
import random
import signal
import string
from collections.abc import Iterator
from dataclasses import dataclass

import greenery
import greenery.charclass
import jinja2
import networkx as nx


def can_parse_regex(regex_str: str) -> bool:
    try:
        greenery.parse(regex_str)
        return True
    except Exception:
        return False


def regex_to_fsm(regex_str: str) -> greenery.fsm.Fsm:
    regex = greenery.parse(regex_str)
    fsm = regex.to_fsm()
    return fsm


def accepts_with_trace(
    fsm: greenery.fsm.Fsm,
    string: str,
    /,
) -> tuple[bool, list[greenery.fsm.StateType], list[greenery.fsm.AlphaType]]:
    trace_state: list[greenery.fsm.StateType] = []
    trace_alphabet: list[greenery.fsm.AlphaType] = []
    state = fsm.initial
    for char in string:
        for charclass in fsm.map[state]:
            if charclass.accepts(char):
                state = fsm.map[state][charclass]
                trace_state.append(state)
                trace_alphabet.append(charclass)
                break
    accepted = state in fsm.finals
    return accepted, trace_state, trace_alphabet


class Timeout:
    def __init__(self, seconds: int = 1, error_message: str = "Timeout") -> None:
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self) -> None:
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback) -> None:
        signal.alarm(0)


def get_chars_custom(charclass: greenery.charclass.Charclass, r: random.Random, n: int = 10) -> list[str]:
    chars = []
    for _ in range(n):
        if not charclass.ord_ranges:
            chars.append(r.choice(string.digits + string.ascii_letters + string.punctuation))
            continue
        char_range = r.choice(charclass.ord_ranges)
        char_index = r.randint(char_range[0], char_range[1])
        char = chr(char_index)
        chars.append(char)
    return chars


def roundrobin(*iterables: list[Iterator]) -> Iterator:
    "Visit input iterables in a cycle until each is exhausted."
    # roundrobin('ABC', 'D', 'EF') → A D E B F C
    # Algorithm credited to George Sakkis
    iterators = map(iter, iterables)
    for num_active in range(len(iterables), 0, -1):
        iterators = itertools.cycle(itertools.islice(iterators, num_active))
        yield from map(next, iterators)


def strings_generator(
    fsm: greenery.fsm.Fsm,
    seed: int = 7,
) -> Iterator[str]:
    r = random.Random(seed)  # noqa: S311

    livestates = {state for state in fsm.states if fsm.islive(state)}
    if fsm.initial not in livestates:
        raise ValueError("Initial state is not a live state")

    g_fsm = nx.DiGraph()
    for state in fsm.states:
        g_fsm.add_node(state, is_inital=state == fsm.initial, is_final=state in fsm.finals, is_live=fsm.islive(state))
    for state, transitions in fsm.map.items():
        for charclass, next_state in transitions.items():
            g_fsm.add_edge(state, next_state, charclass=charclass)

    a = nx.nx_agraph.to_agraph(g_fsm)
    for node in a.nodes():
        if a.get_node(node).attr["is_live"] == "True":
            a.get_node(node).attr["color"] = "green"

    path_gens = []
    for state_final in fsm.finals:
        paths_generator = nx.all_simple_paths(g_fsm, source=fsm.initial, target=state_final)
        path_gens.append(paths_generator)
    paths = list(itertools.chain.from_iterable(path_gens))

    # for path in paths_gen:
    while True:
        path = r.choice(paths)
        path_str = []
        i = 0
        done = False
        while not done:
            state = path[i]
            state_has_self_loop = state in g_fsm.successors(state)
            sample_self_loop = r.choices([True, False], [0.2, 0.8])[0]
            if state_has_self_loop and sample_self_loop:
                next_state = state
                charclass = g_fsm.get_edge_data(state, next_state)["charclass"]
                char = get_chars_custom(charclass, r, 1)[0]
                path_str.append(char)
            elif i + 1 < len(path):
                next_state = path[i + 1]
                charclass = g_fsm.get_edge_data(state, next_state)["charclass"]
                char = get_chars_custom(charclass, r, 1)[0]
                path_str.append(char)
                i += 1
            else:
                done = True

        valid_string = "".join(path_str)
        yield valid_string


def gen_tb_str_verilog_fsm(
    fsm: greenery.fsm.Fsm,
    alphabet_size: int,
    state_size: int,
    alphabet_encoding: dict[greenery.fsm.AlphaType, str],
    state_encoding: dict[greenery.fsm.StateType, str],
    n_valid_samples_attempt: int = 10,
    gen_vcd: bool = False,
) -> str:
    if fsm.empty():
        raise ValueError("FSM is empty")

    test_data = {}
    valid_strings_iter = strings_generator(fsm)
    valid_strings = []
    for _ in range(n_valid_samples_attempt):
        try:
            is_valid = False
            while not is_valid:
                valid_string = next(valid_strings_iter)
                is_valid = fsm.accepts(valid_string)
            valid_strings.append(valid_string)

        except StopIteration:
            break

    for valid_string in valid_strings:
        _, trace_state, trace_alphabet = accepts_with_trace(fsm, valid_string)
        trace_state_encoded = [state_encoding[state] for state in trace_state]
        trace_alphabet_encoded = [alphabet_encoding[char] for char in trace_alphabet]
        test_data[valid_string] = {
            "trace_state": trace_state_encoded,
            "trace_alphabet": trace_alphabet_encoded,
        }

    tb = ""
    tb += "module tb_fsm;\n"
    tb += "    parameter PERIOD = 10;\n"
    tb += "    parameter PERIOD_HALF = 5;\n"
    tb += "\n"
    tb += "    reg clk = 0;\n"
    tb += "    reg rst = 0;\n"
    tb += f"    reg [{alphabet_size}-1:0] data_in;\n"
    tb += "    reg data_valid;\n"
    tb += f"    wire [{state_size}-1:0] state;\n"
    tb += "    wire accepted;\n"
    tb += "\n"
    tb += "    fsm uut (\n"
    tb += "        .clk(clk),\n"
    tb += "        .rst(rst),\n"
    tb += "        .data_in(data_in),\n"
    tb += "        .data_valid(data_valid),\n"
    tb += "        .state(state),\n"
    tb += "        .accepted(accepted)\n"
    tb += "    );\n"
    tb += "\n"
    if gen_vcd:
        tb += "    initial begin\n"
        tb += '        $dumpfile("tb_fsm.vcd");\n'
        tb += "        $dumpvars(0, tb_fsm);\n"
        tb += "    end\n"
        tb += "\n"
    tb += "    initial begin\n"
    tb += '        $display("Starting simulation");\n'
    tb += "        forever #(PERIOD/2) clk = ~clk;\n"
    tb += "    end\n"
    tb += "\n"
    tb += "    initial begin\n"
    tb += "        rst = 1;\n"
    tb += "        data_valid = 0;\n"
    tb += "        data_in = 0;\n"
    tb += "        #(PERIOD)\n"
    tb += "        rst = 0;\n"
    tb += "        data_valid = 0;\n"
    tb += "        data_in = 0;\n"
    tb += "        #(PERIOD)\n"
    tb += "\n"
    for test_string, trace_data in test_data.items():
        test_string_escaped = ""
        for c in test_string:
            if c in string.whitespace:
                test_string_escaped += "<ws>"
            else:
                test_string_escaped += c
        tb += f"        // Test string: {test_string_escaped}\n"
        tb += "        data_valid = 1;\n"
        for c, s in zip(trace_data["trace_alphabet"], trace_data["trace_state"], strict=False):
            tb += f"        data_in = {alphabet_size}'b{c};\n"
            tb += "        #(PERIOD)\n"
            tb += "        // Check state\n"
            tb += f'       $display("Expected state: {s}");\n'
            tb += '       $display("Actual state: %b", state);\n'
            tb += f'        if (state != {state_size}\'b{s}) $display("ERROR: State mismatch");\n'
        tb += "        data_valid = 0;\n"
        tb += "        #(PERIOD)\n"
        # print the accepted val
        tb += '        $display("Accepted: %b", accepted);\n'
        # if 0 print ERROR
        tb += '        if (!accepted) $display("ERROR: FSM did not accept the string");\n'
        # reset state
        tb += "        rst = 1;\n"
        tb += "        data_valid = 0;\n"
        tb += "        data_in = 0;\n"
        tb += "        #(PERIOD)\n"
        tb += "        rst = 0;\n"
        tb += "        data_valid = 0;\n"
        tb += "        data_in = 0;\n"
        tb += "        #(PERIOD)\n"
        tb += "\n"
    tb += '        $display("Simulation complete");\n'
    tb += "        $finish;\n"
    tb += "    end\n"
    tb += "endmodule\n"

    return tb


template_str_verilog_fsm = """
module fsm(
    input wire clk,
    input wire rst,
    input wire [{{alphabet_size}}-1:0] data_in,
    input wire data_valid,
    output wire [{{state_size}}-1:0] state,
    output wire accepted
);
    // state machine for the following regex pattern:
    // {{regex_pattern}}

    // states
    {% for state, state_enc in state_encoding.items() %}
    // {{state}}: {{state_size}}'b{{state_enc}};
    {% endfor %}

    // alphabet
    {% for charclass, char in alphabet_encoding.items() %}
    // {{charclass}}: {{alphabet_size}}'b{{char}};
    {% endfor %}

    reg [{{state_size}}-1:0] state_reg = {{state_size}}'b{{initial_state}};
    reg [{{state_size}}-1:0] state_next;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state_reg <= {{state_size}}'b{{initial_state}};
        end else if (data_valid) begin
            state_reg <= state_next;
        end else begin
            state_reg <= state_reg;
        end
    end

    always @(*) begin
        case (state_reg)
            {% for state in states %}
            {{state_size}}'b{{state}}: begin
                {% for transition in transitions[state] %}
                if (data_in == {{alphabet_size}}'b{{transition.input}}) begin
                    state_next = {{state_size}}'b{{transition.next_state}};
                end
                {% endfor %}
            end
            {% endfor %}
        endcase
    end

    assign state = state_reg;
    assign accepted = {% for final_state in final_states %}(state_reg == {{state_size}}'b{{final_state}}){% if not loop.last %} || {% endif %}{% endfor %};
endmodule
"""


template_verilog_fsm = jinja2.Template(
    template_str_verilog_fsm,
    trim_blocks=True,
    lstrip_blocks=True,
)


def bin_encode(n: int, width: int) -> str:
    if width < n.bit_length():
        raise ValueError("n is too large to fit in width bits")
    return bin(n).removeprefix("0b").zfill(width)


def onehot_encode(n: int, width: int) -> str:
    if n >= width:
        raise ValueError("n is too large to fit in width bits")
    str_onehot = ["0"] * width
    str_onehot[n] = "1"
    str_onehot = str_onehot[::-1]
    return "".join(str_onehot)


def gray_encode(n: int, width: int) -> str:
    if width < n.bit_length():
        raise ValueError("n is too large to fit in width bits")
    return bin(n ^ (n >> 1)).removeprefix("0b").zfill(width)


class Encoding(enum.Enum):
    BINARY = enum.auto()
    ONEHOT = enum.auto()
    GRAY = enum.auto()


@dataclass
class FsmToVerilogData:
    fsm: greenery.fsm.Fsm
    encoding_alphabet: Encoding
    encoding_state: Encoding
    alphabet_size: int
    state_size: int
    alphabet_encoding: dict[greenery.fsm.AlphaType, str]
    state_encoding: dict[greenery.fsm.StateType, str]
    alphabet_encoded: frozenset[str]
    state_encoded: frozenset[str]
    transitions_encoded: dict[str, list[dict[str, str]]]
    initial_state_encoded: str
    final_states_encoded: list[str]


def fsm_to_verilog(
    regex_str: str,
    regex_pattern: greenery.Pattern,
    fsm: greenery.fsm.Fsm,
    encoding_alphabet: Encoding = Encoding.BINARY,
    encoding_state: Encoding = Encoding.GRAY,
) -> tuple[str, FsmToVerilogData]:
    alphabet = fsm.alphabet
    states = fsm.states
    initial = fsm.initial
    finals = fsm.finals
    transitions = fsm.map

    match encoding_alphabet:
        case Encoding.BINARY:
            alphabet_size = len(alphabet).bit_length()
            alphabet_encoding = {char: bin_encode(i, alphabet_size) for i, char in enumerate(alphabet)}
        case Encoding.ONEHOT:
            alphabet_size = len(alphabet)
            alphabet_encoding = {char: onehot_encode(i, alphabet_size) for i, char in enumerate(alphabet)}
        case Encoding.GRAY:
            alphabet_size = len(alphabet).bit_length()
            alphabet_encoding = {char: gray_encode(i, alphabet_size) for i, char in enumerate(alphabet)}

    match encoding_state:
        case Encoding.BINARY:
            state_size = len(states).bit_length()
            state_encoding = {state: bin_encode(i, state_size) for i, state in enumerate(states)}
        case Encoding.ONEHOT:
            state_size = len(states)
            state_encoding = {state: onehot_encode(i, state_size) for i, state in enumerate(states)}
        case Encoding.GRAY:
            state_size = len(states).bit_length()
            state_encoding = {state: gray_encode(i, state_size) for i, state in enumerate(states)}

    alphabet_encoded = frozenset(alphabet_encoding.values())
    state_encoded = frozenset(state_encoding.values())

    transitions_encoded = {
        state_encoding[state]: [
            {
                "input": alphabet_encoding[a_input],
                "next_state": state_encoding[next_state],
            }
            for a_input, next_state in transitions[state].items()
        ]
        for state in states
    }

    initial_state_encoded = state_encoding[initial]
    final_states_encoded = [state_encoding[state] for state in finals]

    data = FsmToVerilogData(
        fsm=fsm,
        encoding_alphabet=encoding_alphabet,
        encoding_state=encoding_state,
        alphabet_size=alphabet_size,
        state_size=state_size,
        alphabet_encoding=alphabet_encoding,
        state_encoding=state_encoding,
        alphabet_encoded=alphabet_encoded,
        state_encoded=state_encoded,
        transitions_encoded=transitions_encoded,
        initial_state_encoded=initial_state_encoded,
        final_states_encoded=final_states_encoded,
    )

    v_str = template_verilog_fsm.render(
        alphabet_size=alphabet_size,
        state_size=state_size,
        initial_state=initial_state_encoded,
        final_states=final_states_encoded,
        alphabet=alphabet_encoded,
        states=state_encoded,
        alphabet_encoding=alphabet_encoding,
        state_encoding=state_encoding,
        transitions=transitions_encoded,
        regex_str=regex_str,
        regex_pattern=regex_pattern,
    )

    return v_str, data
