"""The arrival-count protocol of the single-launch decode, read from the CUDA sources as tokens (plain Python: no GPU, no torch).

No comparison of results can see most of what the protocol relies on: a plain store instead of the relaxed one, the arrival atomic behind a barrier or behind other work, an additional
read of the ``pos`` word, a fence, a spin-wait. They are rules about the text of ``rows_entry<true>`` (csrc/cfbi/cfbi_rows.cuh) and what it relies on, and this module reads that text the way
a build that defines no test macro reads it:

* comments are removed and the conditionals of the test macros are resolved (``default_text``: the branch of ``#ifdef M`` is dropped, the ``#else`` branch kept); a string that a rule asks for
  therefore cannot be satisfied by a comment or by a variant branch;
* the hooks that the default build compiles to nothing are expanded the way the default macros define them (``ROWS_DELAY(n)`` and ``ROWS_DECODE_ARRIVED(old, heads)`` vanish,
  ``ROWS_POS_INDEX(access, slot)`` is ``(slot)``; the definitions themselves are pinned) and the text is cut into tokens, so spacing, line breaks and the name of a hook argument do not matter;
* ``problems(texts)`` returns ``[(rule, message)]``: the rule that an edit breaks says which property of the protocol it endangers. ``R0`` (the whole decode entry equals ``GOLDEN``) is the
  complete pin: a deliberate change of the protocol updates ``GOLDEN`` and the model of ``test_decode_pos_protocol_model.py`` together; ``R1 .. R12`` name the properties one by one.

The model of the protocol reads its constants and its election from here (``constants``, ``election``), so that the model and the kernel cannot drift apart.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
FILES = ("cfbi_common.cuh", "cfbi_primitives.cuh", "cfbi_rows.cuh", "cfbi_launch.cu", "cfbi_binding.cpp")

# the macros of the variant builds, by prefix (the default build defines none of them)
TEST_PREFIXES = ("CFBI_ROWS_PROFILE", "CFBI_ROWS_DELAY_NS", "CFBI_ROWS_TEST_", "CFBI_NEG_CONTROL", "CFBI_K1_PROFILE", "CFBI_K2_PROFILE", "CFBI_UMMA_PROFILE", "CFBI_DYN_TEST_",
                 "CFBI_K2_TEST_", "CFBI_UMMA_NEG_", "CFBI_UMMA_RACE_", "CFBI_CONV_NEG_", "CFBI_LIN_NEG_", "CFBI_DECODE_TEST_")

STRING = r'"(?:[^"\\]|\\.)*"'
TOKEN = re.compile(STRING + r"|[A-Za-z_]\w*|\d[\w.]*|->|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||::|\+\+|--|\S")


def is_test_macro(name):
    return any(name.startswith(p) for p in TEST_PREFIXES)


def macros_in(texts):
    """The macro names of the variants that some text tests (derived, not listed): every ``#ifdef`` / ``#ifndef`` / ``defined()`` of a name with a test prefix."""
    found = set()
    for t in texts:
        found |= set(re.findall(r"#\s*(?:ifdef|ifndef)\s+(CFBI_\w+)", t)) | set(re.findall(r"defined\s*\(?\s*(CFBI_\w+)", t))
    return {m for m in found if is_test_macro(m)}


def default_text(text, macros):
    """The text as a build that defines none of ``macros`` reads it: the branch of ``#ifdef M`` is dropped and its ``#else`` branch kept (``#ifndef M`` the other way round), the
    directive lines of those conditionals are dropped, every other line stays. Only ``#ifdef`` / ``#ifndef`` / ``#else`` / ``#endif`` of a test macro are read; ``#elif`` of one is refused."""
    out, stack = [], []
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(\w+)?", line)
        skipping = any(f[1] for f in stack)
        if m:
            d, name = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef") and name in macros:
                stack.append(["test", d == "ifdef"])
                continue
            if d in ("ifdef", "ifndef", "if"):
                stack.append(["other", False])
            elif d == "endif":
                if stack.pop()[0] == "test":
                    continue
            elif d == "else" and stack and stack[-1][0] == "test":
                stack[-1][1] = not stack[-1][1]
                continue
            elif d == "elif" and stack and stack[-1][0] == "test":
                raise ValueError("#elif of a conditional on a test macro is not read by this preprocessor")
        if not skipping:
            out.append(line)
    assert not stack, "unbalanced preprocessor conditionals"
    return "\n".join(out)


def strip_comments(text):
    """The text without ``/* */`` and ``//`` comments (line structure kept; string literals are not touched)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            m = re.compile(STRING).match(text, i)
            j = m.end() if m else i + 1
            out.append(text[i:j])
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokens(text):
    """The tokens of the code in ``text`` (comments dropped; preprocessor lines are not code and are dropped as well)."""
    code = "\n".join(l for l in strip_comments(text).splitlines() if not l.lstrip().startswith("#"))
    return TOKEN.findall(code)


def norm(text):
    """One normal form of a snippet of source: its tokens, single-spaced."""
    return " ".join(tokens(text))


def load(csrc=CSRC):
    return {name: (Path(csrc) / name).read_text() for name in FILES}


def default_view(texts):
    macros = macros_in(texts.values())
    return {name: default_text(text, macros) for name, text in texts.items()}


def match_close(toks, i, open_, close):
    """The index of the token that closes the bracket opened at ``toks[i]``."""
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == open_:
            depth += 1
        elif toks[j] == close:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced %s%s" % (open_, close))


def macro_definitions(view_text):
    """The ``#define`` lines of a default view: name -> (parameters, body text)."""
    defs = {}
    for m in re.finditer(r"^\s*#\s*define\s+(\w+)\(([^)]*)\)\s*(.*)$", view_text, re.M):
        defs[m.group(1)] = ([a.strip() for a in m.group(2).split(",")], m.group(3).strip())
    return defs


def expand_default_hooks(toks):
    """The tokens as the default build reads them: ``ROWS_DELAY(..);`` and ``ROWS_DECODE_ARRIVED(..);`` are gone, ``ROWS_POS_INDEX(access, slot)`` is ``( slot )``."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("ROWS_DELAY", "ROWS_DECODE_ARRIVED") and i + 1 < len(toks) and toks[i + 1] == "(":
            j = match_close(toks, i + 1, "(", ")")
            i = j + 2 if j + 1 < len(toks) and toks[j + 1] == ";" else j + 1
        elif t == "ROWS_POS_INDEX" and i + 1 < len(toks) and toks[i + 1] == "(":
            j = match_close(toks, i + 1, "(", ")")
            inner = toks[i + 2:j]
            depth, comma = 0, None
            for k, x in enumerate(inner):
                depth += x == "("
                depth -= x == ")"
                if x == "," and depth == 0:
                    comma = k
                    break
            assert comma is not None, "ROWS_POS_INDEX takes (access, slot)"
            out.extend(["("] + inner[comma + 1:] + [")"])
            i = j + 1
        else:
            out.append(t)
            i += 1
    return out


def function_body(toks, head, after=0):
    """The tokens of the body of the function whose head (a snippet) is ``head``: from the ``{`` that follows its parameter list to the matching ``}`` (both excluded)."""
    h = tokens(head)
    for i in range(after, len(toks) - len(h)):
        if toks[i:i + len(h)] == h:
            j = i + len(h) - 1
            assert toks[j] == "(", "the head of a function ends at its parameter list"
            k = match_close(toks, j, "(", ")")
            assert toks[k + 1] == "{", "no body after %s" % head
            e = match_close(toks, k + 1, "{", "}")
            return toks[k + 2:e]
    return None


ENTRY_HEAD = "SeqPlan rows_entry("
KERNEL_HEAD = "rows_kernel("
BRANCH_PREFIX = norm("if constexpr (!DECODE) { return p.plan[n]; } else {").split()

# ---- the pinned text -----------------------------------------------------------------------------------------------------------------------------------------
GOLDEN = norm("""
    const SeqWindow w = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);
    if (w.len == 0) return SeqPlan{0, 0, 0, 0, 0, 0, 0, w.slot};
    if (threadIdx.x == 0) {
        const int old = atomicAdd(p.pos + (w.slot), 1 << POS_COUNT_SHIFT);
        pos_smem = old & POS_MASK;
        if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + (w.slot), (int)(((int64_t)(old & POS_MASK) + w.len) % BT));
    }
    __syncthreads();
    return SeqPlan{w.tok0, w.len, pos_smem, w.len, 0, 0, 0, w.slot};
""")
SKIP = norm("if (w.len == 0) return SeqPlan{0, 0, 0, 0, 0, 0, 0, w.slot};")
BLOCK_OPEN = norm("if (threadIdx.x == 0) {")
ARRIVAL = norm("const int old = atomicAdd(p.pos + (w.slot), 1 << POS_COUNT_SHIFT);")
FILL = norm("pos_smem = old & POS_MASK;")
ELECTION = norm("if ((old >> POS_COUNT_SHIFT) == p.H - 1)")
STORE = norm("st_relaxed_gpu(p.pos + (w.slot), (int)(((int64_t)(old & POS_MASK) + w.len) % BT));")
BARRIER = norm("__syncthreads();")
STORE_HELPER = norm('__device__ __forceinline__ void st_relaxed_gpu(int* ptr, int v) { asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory"); }')
FUSED_TAIL = norm("if (DECODE) { } else if (p.fuse_pos && tid == 0) { p.pos[(sp.slot)] = (int)(((int64_t)sp.pos0 + sp.len) % BT); }")
KERNEL_MAP = norm("const int n = blockIdx.x, h = blockIdx.y, H = p.H;")
ENTRY_CALL = norm("rows_entry<DECODE>(p, n, dec_pos0)")
HOOK_DEFAULTS = {"ROWS_DELAY": ["n"], "ROWS_DECODE_ARRIVED": ["old", "heads"], "ROWS_POS_INDEX": ["access", "slot"]}
HOOK_BODIES = {"ROWS_DELAY": "do {} while (0)", "ROWS_DECODE_ARRIVED": "do {} while (0)", "ROWS_POS_INDEX": "(slot)"}
LAUNCH = [norm("p.H = a.H;"), norm("p.pos = a.pos;"), norm("const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};"),
          norm("rows_kernel<true><<<dim3(a.N, a.H), NTHREADS, ROWS_DYNAMIC_SMEM, stream>>>(dp);")]
ALLOWED_CALLS = {"seq_window", "SeqPlan", "atomicAdd", "st_relaxed_gpu", "__syncthreads"}
FORBIDDEN = {"for", "while", "do", "goto", "volatile", "asm", "__threadfence", "__threadfence_block", "__threadfence_system", "__nanosleep", "atomicCAS", "atomicExch", "atomicOr",
             "atomicAnd", "atomicMax", "atomicMin", "atomicInc", "atomicDec", "atomicSub", "atomicXor", "__ldg", "__ldcg", "__ldcs", "__ldlu", "ld_relaxed_gpu"}


def evaluate(expr, names):
    assert re.fullmatch(r"[\w\s\-+*()<>]+", expr), expr
    return eval(expr, {"__builtins__": {}}, dict(names))


def constants(texts):
    """``BT``, ``POS_COUNT_SHIFT`` and ``POS_MASK`` as the sources define them (evaluated, not copied)."""
    view = default_view(texts)
    found = {}
    for name in ("BT", "POS_COUNT_SHIFT", "POS_MASK"):
        for text in view.values():
            m = re.search(r"constexpr\s+int\s+%s\s*=\s*([^;]+);" % name, strip_comments(text))
            if m:
                found[name] = evaluate(m.group(1).strip(), found)
                break
        else:
            raise KeyError("the constant %s is not defined" % name)
    return found


def decode_branch(texts):
    """The tokens of the ``DECODE = true`` branch of ``rows_entry`` as the default build reads them (hooks expanded); ``None`` if the function or the branch is not there."""
    toks = expand_default_hooks(tokens(default_view(texts)["cfbi_rows.cuh"]))
    body = function_body(toks, ENTRY_HEAD.replace("(", " ("))
    if body is None:
        return None
    if body[:len(BRANCH_PREFIX)] != BRANCH_PREFIX or body[-1] != "}":
        return None
    return body[len(BRANCH_PREFIX):-1]


def election(texts):
    """The election of the model: ``f(count, H)`` is true for the CTA that is to store, read from the condition of the elected store in the source. Refuses (ValueError) a condition that is
    not of the form ``(old >> POS_COUNT_SHIFT) == <expression of p.H and numbers>``."""
    branch = decode_branch(texts)
    s = " ".join(branch or [])
    m = re.search(r"if \( \( old >> POS_COUNT_SHIFT \) == (.+?) \) st_relaxed_gpu", s)
    if not m:
        raise ValueError("the elected store has no condition of the form (old >> POS_COUNT_SHIFT) == expression")
    rhs = m.group(1).replace("p . H", "H")
    if not re.fullmatch(r"[H\d\s\-+()]+", rhs):
        raise ValueError("the election compares the count with %r, which the model does not read" % rhs)
    return lambda count, H: count == evaluate(rhs, {"H": H})


def counter_limit_problems(texts):
    """The arrival counter must never wrap the word: heads * 2^POS_COUNT_SHIFT + POS_MASK < 2^31 for every H that the binding admits (``H < N`` in its size check)."""
    probs = []
    c = constants(texts)
    m = re.search(r"H\s*<\s*(\d+)\s*,\s*\"problem too large", strip_comments(texts["cfbi_binding.cpp"]))
    if not m:
        return [("R10", "the binding has no size check of the form H < <limit> that the counter capacity can be tied to")]
    limit = int(m.group(1))
    if (limit - 1) * (1 << c["POS_COUNT_SHIFT"]) + c["POS_MASK"] >= 2 ** 31 - (1 << c["POS_COUNT_SHIFT"]):
        probs.append(("R10", "H up to %d heads can overflow the arrival counter: %d x 2^%d" % (limit - 1, limit - 1, c["POS_COUNT_SHIFT"])))
    return probs


# the conditionals that remain in the default view of a file (those on build tunables, which no variant build changes); any other one would hide text from the compiler that the rules still read
ALLOWED_CONDITIONALS = {"cfbi_rows.cuh": {"ifndef CFBI_ROWS_PF_L1", "if CFBI_ROWS_PF_L1", "ifndef CFBI_ROWS_MINB"}, "cfbi_common.cuh": set(), "cfbi_launch.cu": set()}


def stray_conditionals(view_text, allowed):
    out = []
    for i, line in enumerate(view_text.splitlines(), 1):
        m = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif)\b\s*(.*)$", line)
        if m:
            key = re.sub(r"\s*(//.*|/\*.*)$", "", ("%s %s" % (m.group(1), m.group(2).strip())).strip())
            if key not in allowed:
                out.append((i, line.strip()))
    return out


def problems(texts):
    """What is wrong with the arrival-count protocol as the sources of ``texts`` (file name -> text) define it: ``[(rule, message)]``, empty when every rule holds."""
    out = []
    try:
        view = default_view(texts)
    except (ValueError, AssertionError) as e:
        return [("R12", "the sources cannot be read as the default build reads them: %s" % e)]
    rows = view["cfbi_rows.cuh"]
    for name, allowed in ALLOWED_CONDITIONALS.items():
        for ln, line in stray_conditionals(view[name], allowed):
            out.append(("R12", "%s:%d holds the conditional `%s`, which the rules cannot resolve (an `#if 0` would hide a protocol line from the compiler and leave it in the text)" % (name, ln, line)))
    # R11: the default hooks are what the expansion assumes
    defs = macro_definitions(rows)
    for name, params in HOOK_DEFAULTS.items():
        if name not in defs or defs[name][0] != params or norm(defs[name][1]) != norm(HOOK_BODIES[name]):
            out.append(("R11", "the default definition of %s is not the one that the rules expand (%s(%s) %s)" % (name, name, ", ".join(params), HOOK_BODIES[name])))
    # R5: constants
    c = None
    try:
        c = constants(texts)
    except KeyError as e:
        out.append(("R5", str(e)))
    if c is not None:
        if c["BT"] != 64:
            out.append(("R5", "BT is %d, the protocol is written for 64" % c["BT"]))
        if c["POS_COUNT_SHIFT"] != 8:
            out.append(("R5", "POS_COUNT_SHIFT is %d, expected 8" % c["POS_COUNT_SHIFT"]))
        if c["POS_MASK"] != c["BT"] - 1:
            out.append(("R5", "POS_MASK is %d, expected BT - 1" % c["POS_MASK"]))
    common = tokens(view["cfbi_common.cuh"])
    sa = [i for i, t in enumerate(common) if t == "static_assert"]
    cond = [" ".join(common[i + 2:common.index(",", i)]) for i in sa]
    if norm("POS_MASK < (1 << POS_COUNT_SHIFT)") not in cond:
        out.append(("R5", "the static_assert POS_MASK < (1 << POS_COUNT_SHIFT) is missing (the fill must fit below the arrival counter)"))
    # R6: the relaxed store helper
    rt = " ".join(expand_default_hooks(tokens(rows)))
    if STORE_HELPER not in rt:
        out.append(("R6", "st_relaxed_gpu is not the relaxed device-scope store (st.relaxed.gpu.global.b32, volatile, memory clobber)"))
    # the entry
    branch = decode_branch(texts)
    if branch is None:
        out.append(("R0", "rows_entry has no `if constexpr (!DECODE) { return p.plan[n]; } else { ... }` body that the rules can read (or its text holds a preprocessor conditional that they do not read)"))
        return out + counter_limit_problems(texts)
    s = " ".join(branch)
    if s != GOLDEN:
        out.append(("R0", "the decode entry is not the pinned protocol:\n  found:    %s\n  expected: %s" % (s, GOLDEN)))
    calls = {t for i, t in enumerate(branch[:-1]) if branch[i + 1] in ("(", "{") and re.match(r"[A-Za-z_]", t) and t not in ("if", "switch")}
    # R1: the arrival atomic is the only read of pos, the first statement of the guarded block, and the fill comes from its result
    if s.count("p . pos") != 2:
        out.append(("R1", "rows_entry reads or writes the pos word %d times in the decode branch, expected 2 (the arrival atomic and the elected store)" % s.count("p . pos")))
    if BLOCK_OPEN + " " + ARRIVAL not in s:
        out.append(("R1", "the arrival atomic is not the first statement of the thread-0 block: `%s`" % ARRIVAL))
    if ARRIVAL + " " + FILL not in s:
        out.append(("R1", "the fill of the open chunk is not taken from the arrival's result with the mask: `%s`" % FILL))
    unknown = calls - ALLOWED_CALLS
    if unknown:
        out.append(("R1", "the decode branch calls %s, which the protocol does not use" % sorted(unknown)))
    # R2: election and store
    if s.count("st_relaxed_gpu") != 1:
        out.append(("R2", "the decode branch has %d stores of the pos word, expected exactly one (the elected store)" % s.count("st_relaxed_gpu")))
    if ELECTION + " " + STORE not in s:
        out.append(("R2", "the elected store is not `%s %s`" % (ELECTION, STORE)))
    # R3: the skip precedes the arrival and depends on the sequence only
    i_skip, i_pos = s.find(SKIP), s.find("p . pos")
    if i_skip < 0 or (i_pos >= 0 and i_skip > i_pos):
        out.append(("R3", "the skip `%s` does not come before the first access to the pos word" % SKIP))
    sw = function_body(tokens(view["cfbi_common.cuh"]), "SeqWindow seq_window (")
    if sw is None:
        out.append(("R3", "seq_window is not found"))
    else:
        sws = " ".join(sw)
        if norm("if (w.slot < 0) w.len = 0;") not in sws:
            out.append(("R3", "seq_window does not make a negative slot an empty window"))
        for bad in ("threadIdx", "blockIdx", "gridDim", "blockDim"):
            if bad in sw:
                out.append(("R3", "seq_window depends on %s: the head CTAs of a sequence could take different decisions" % bad))
    # R4: no loop, no spin, no fence, one barrier
    for t in sorted(FORBIDDEN & set(branch)):
        out.append(("R4", "the decode branch uses %s (no spin-wait, loop, fence or other atomic belongs to the protocol)" % t))
    if branch.count("__syncthreads") != 1:
        out.append(("R4", "the decode branch has %d barriers, expected one (the hand-over of the fill)" % branch.count("__syncthreads")))
    # R9: the hand-over: pos_smem is written before the barrier, read after it
    if branch.count("pos_smem") != 2 or not (s.find(FILL) >= 0 and s.find(FILL) < s.find(BARRIER) < s.rfind("pos_smem")):
        out.append(("R9", "pos_smem is not written inside the thread-0 block before the barrier and read once after it"))
    if BARRIER in s and s.find(BARRIER) < s.find(ARRIVAL):
        out.append(("R9", "a barrier comes before the arrival atomic"))
    # R7: no other access to the pos word in the whole row engine; the kernel hands p to rows_entry only
    allrows = expand_default_hooks(tokens(rows))
    member = [i for i, t in enumerate(allrows) if t == "pos" and i > 0 and allrows[i - 1] in (".", "->")]
    if len(member) != 3:
        out.append(("R7", "cfbi_rows.cuh reaches a member `pos` %d times, expected 3 (the arrival atomic, the elected store and the fused store of the two-launch path)" % len(member)))
    if FUSED_TAIL not in " ".join(allrows):
        out.append(("R7", "the fused position store is not `%s` (it must be the DECODE = false branch of the kernel tail)" % FUSED_TAIL))
    kbody = function_body(allrows, "__global__ void __launch_bounds__ ( NTHREADS , CFBI_ROWS_MINB ) rows_kernel (")
    if kbody is None:
        out.append(("R7", "rows_kernel is not found"))
    else:
        ks = " ".join(kbody)
        passed = [i for i, t in enumerate(kbody) if t == "p" and i + 1 < len(kbody) and kbody[i + 1] != "."]
        if len(passed) != 1 or ENTRY_CALL not in ks:
            out.append(("R7", "rows_kernel passes the parameter block p to %d places, expected the one call %s" % (len(passed), ENTRY_CALL)))
        if KERNEL_MAP not in ks:
            out.append(("R8", "rows_kernel does not map the grid as `%s` (CTA (n, h) of sequence n and head h, H from the parameters)" % KERNEL_MAP))
    # R8: the launch geometry
    lt = " ".join(tokens(view["cfbi_launch.cu"]))
    for snippet in LAUNCH:
        if snippet not in lt:
            out.append(("R8", "the launcher does not contain `%s`" % snippet))
    launches = lt.count("rows_kernel < true > << <")      # a launch (the occupancy table names the kernel without launching it)
    if launches != 1:
        out.append(("R8", "the launcher has %d launches of rows_kernel<true>, expected one" % launches))
    return out + counter_limit_problems(texts)
