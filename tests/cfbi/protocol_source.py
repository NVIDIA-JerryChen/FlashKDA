"""The arrival-count protocol of the single-launch decode, read from the CUDA sources as tokens (plain Python: no GPU, no torch).

No comparison of results can see most of what the protocol relies on: a plain store instead of the relaxed one, the arrival atomic behind a barrier or behind other work, an additional
read of the ``pos`` word, a fence, a spin-wait. They are rules about the text of ``rows_entry<true>`` (csrc/cfbi/cfbi_rows.cuh) and what it relies on, and this module reads that text the way
a build that defines no test macro reads it:

* the text is first cut into LOGICAL lines the way the first three phases of translation do it (``logical_lines``: a backslash before a line break joins the two lines, also when blanks sit between
  them, a comment is one space and a block comment takes its line breaks with it, string and character literals are left alone; a line keeps the number of its first physical line), so that a
  directive spelled over several lines is one directive, here as for the compiler;
* the conditionals of the test macros are resolved on those lines (``default_text``: the branch of ``#ifdef M`` is dropped, the ``#else`` branch kept); a string that a rule asks for
  therefore cannot be satisfied by a comment or by a variant branch;
* the hooks that the default build compiles to nothing are expanded the way the default macros define them (``ROWS_DELAY(n)`` and ``ROWS_DECODE_ARRIVED(old, heads)`` vanish,
  ``ROWS_POS_INDEX(access, slot)`` is ``(slot)``; the definitions themselves are pinned) and the text is cut into tokens, so spacing, line breaks and the name of a hook argument do not matter;
* ``problems(texts)`` returns ``[(rule, message)]``: the rule that an edit breaks says which property of the protocol it endangers. ``R0`` (the whole decode entry equals ``GOLDEN``) is the
  complete pin: a deliberate change of the protocol updates ``GOLDEN`` and the model of ``test_decode_pos_protocol_model.py`` together; ``R1 .. R12`` name the properties one by one. R12 also keeps the pins honest about the preprocessor: the tokens that the pins read are what the compiler sees only if no macro
  rewrites them, so every effective ``#define`` / ``#undef`` of every source file (and every ``-D`` / ``-U`` flag of the build) is read, per directive, and one that names an identifier of
  a pinned snippet is a problem unless it is a modeled hook or the tunable ``CFBI_ROWS_MINB``; a quoted ``#include`` of a file that is not read (and a forced include of the build flags) could bring a macro
  that no pin sees and is a problem as well.

The model of the protocol reads its constants and its election from here (``constants``, ``election``), so that the model and the kernel cannot drift apart.
"""
import functools
import posixpath
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
FILES = ("cfbi_common.cuh", "cfbi_primitives.cuh", "cfbi_rows.cuh", "cfbi_launch.cu", "cfbi_binding.cpp")

# the macros of the variant builds, by prefix (the default build defines none of them)
TEST_PREFIXES = ("CFBI_ROWS_PROFILE", "CFBI_ROWS_DELAY_NS", "CFBI_ROWS_TEST_", "CFBI_NEG_CONTROL", "CFBI_K1_PROFILE", "CFBI_K2_PROFILE", "CFBI_UMMA_PROFILE", "CFBI_DYN_TEST_",
                 "CFBI_K2_TEST_", "CFBI_UMMA_NEG_", "CFBI_UMMA_RACE_", "CFBI_CONV_NEG_", "CFBI_LIN_NEG_", "CFBI_DECODE_TEST_", "CFBI_ROWS_NEG_E_")

STRING = r'"(?:[^"\\]|\\.)*"'
TOKEN = re.compile(STRING + r"|[A-Za-z_]\w*|\d[\w.]*|->|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||::|\+\+|--|\S")


def is_test_macro(name):
    return any(name.startswith(p) for p in TEST_PREFIXES)


# a directive starts with a ``#`` or its digraph ``%:`` (blanks may stand before it, between it and the keyword and, as form feeds and vertical tabs too, inside the line)
HASH = r"[ \t\v\f]*(?:#|%:)[ \t\v\f]*"
DIRECTIVE_START = re.compile(r"^[ \t\v\f]*(?:#|%:)")
CONDITIONAL = re.compile(r"^" + HASH + r"(ifdef|ifndef|if|elifdef|elifndef|elif|else|endif)\b[ \t\v\f]*(\w+)?")
_SPLICE = re.compile(r"\\[ \t\f\v]*\n")                  # a backslash, blanks (which gcc accepts too) and a line break
_LEX = re.compile(r'"(?:[^"\\\n]|\\[^\n])*"?|(?<![A-Za-z0-9_])\'(?:\\[^\n]|[^\'\\\n])+\'|//[^\n]*|/\*.*?(?:\*/|\Z)|\n|[^"\'/\n]+|["\'/]', re.S)


@functools.lru_cache(maxsize=512)
def logical_lines(text):
    """``((first physical line, line), ...)``: the text as the first three phases of translation read it. A backslash before a line break (also with blanks between them: gcc joins those too) joins
    the two lines, inside comments and literals as everywhere; a comment is replaced by one space and a block comment takes the line breaks inside it with it, so a directive can run over a
    comment that spans lines; string and character literals are not touched (a ``//`` in a string starts no comment); the lines that are left are cut at the remaining line breaks. A line is
    numbered by the physical line of its first character, an empty one by its own. Digit separators (``1'000``) are no character literals."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces, starts, line, pos = [], [], 1, 0      # starts[k]: the physical line that begins after the (k+1)-th kept line break
    for m in _SPLICE.finditer(text):
        piece = text[pos:m.start()]
        pieces.append(piece)
        for _ in range(piece.count("\n")):
            line += 1
            starts.append(line)
        line += 1                                    # the spliced break advances the physical line and cuts nothing
        pos = m.end()
    piece = text[pos:]
    pieces.append(piece)
    for _ in range(piece.count("\n")):
        line += 1
        starts.append(line)
    spliced = "".join(pieces)

    def at(k):
        return 1 if k == 0 else starts[k - 1]
    out, parts, start, breaks = [], [], None, 0
    for m in _LEX.finditer(spliced):
        tok = m.group()
        if tok == "\n":
            out.append((at(breaks) if start is None else start, "".join(parts)))
            parts, start = [], None
            breaks += 1
            continue
        if start is None:
            start = at(breaks)
        if tok.startswith("/*"):
            parts.append(" ")
            breaks += tok.count("\n")
        elif tok.startswith("//"):
            parts.append(" ")
        else:
            parts.append(tok)
    if parts:
        out.append((start, "".join(parts)))
    return tuple(out)


def macros_in(texts):
    """The macro names of the variants that some text tests (derived, not listed): every ``#ifdef`` / ``#ifndef`` / ``defined()`` of a name with a test prefix, read on logical lines (no comment,
    a directive that is spelled over several lines counts)."""
    found = set()
    for t in texts:
        code = "\n".join(line for _, line in logical_lines(t))
        found |= set(re.findall(HASH + r"(?:ifdef|ifndef)[ \t\v\f]+(CFBI_\w+)", code)) | set(re.findall(r"defined\s*\(?\s*(CFBI_\w+)", code))
    return {m for m in found if is_test_macro(m)}


def default_lines(text, macros):
    """``[(first physical line in ``text``, logical line)]`` of the text as a build that defines none of ``macros`` reads it (see ``default_text``); the numbers are those of the file, not of the view,
    and the lines are the logical lines of ``logical_lines`` (no comment is left, a directive is one line)."""
    out, stack = [], []
    for number, line in logical_lines(text):
        m = CONDITIONAL.match(line)
        skipping = any(f[1] for f in stack)
        if m:
            d, name = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef") and name in macros:
                stack.append(["test", d == "ifdef"])
                continue
            if d in ("ifdef", "ifndef", "if"):
                stack.append(["other", False])
            elif d == "endif":
                if not stack:
                    raise ValueError("%d: #endif without a conditional" % number)
                if stack.pop()[0] == "test":
                    continue
            elif d == "else" and stack and stack[-1][0] == "test":
                stack[-1][1] = not stack[-1][1]
                continue
            elif d in ("elif", "elifdef", "elifndef") and stack and stack[-1][0] == "test":
                raise ValueError("#%s of a conditional on a test macro is not read by this preprocessor" % d)
        if not skipping:
            out.append((number, line))
    assert not stack, "unbalanced preprocessor conditionals"
    return out


def _joined(numbered_lines):
    """The text of ``[(line number, line)]``."""
    return "\n".join(line for _, line in numbered_lines)


def default_text(text, macros):
    """The text as a build that defines none of ``macros`` reads it: the branch of ``#ifdef M`` is dropped and its ``#else`` branch kept (``#ifndef M`` the other way round), the
    directive lines of those conditionals are dropped, every other logical line stays (without its comments). Only ``#ifdef`` / ``#ifndef`` / ``#else`` / ``#endif`` of a test macro are read;
    ``#elif`` of one is refused."""
    return _joined(default_lines(text, macros))


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
    """The tokens of the code in ``text`` (comments dropped; the logical lines of the preprocessor, a directive that runs over several lines included, are not code and are dropped as well)."""
    code = "\n".join(line for _, line in logical_lines(text) if not DIRECTIVE_START.match(line))
    return TOKEN.findall(code)


def norm(text):
    """One normal form of a snippet of source: its tokens, single-spaced."""
    return " ".join(tokens(text))


SOURCE_SUFFIXES = (".cu", ".cuh", ".h", ".cpp")
FLAG_FILES = {"flags:_ext.py": ROOT / "flash_kda" / "cfbi" / "_ext.py", "flags:setup.py": ROOT / "setup.py"}      # where the nvcc flags of the build are written (texts under the keys "flags:...")


def load(csrc=CSRC):
    """Every source file of ``csrc`` (relative path -> text: the macros of any of them can rewrite what the pins read) and, for the repository's own tree, the files that hold the build flags."""
    root = Path(csrc)
    texts = {str(f.relative_to(root)): f.read_text() for f in sorted(root.rglob("*")) if f.is_file() and f.suffix in SOURCE_SUFFIXES}
    missing = [n for n in FILES if n not in texts]
    if missing:
        raise FileNotFoundError("the sources %s are not in %s" % (missing, root))
    if root == CSRC:
        texts.update({key: path.read_text() for key, path in FLAG_FILES.items() if path.exists()})
    return texts


def source_texts(texts):
    """The C++ sources of ``texts`` (without the flag files)."""
    return dict((k, v) for k, v in texts.items() if not k.startswith("flags:"))


_VIEWS = {}


def _views(texts):
    srcs = source_texts(texts)
    key = tuple(sorted((k, hash(v)) for k, v in srcs.items()))
    if key not in _VIEWS:
        if len(_VIEWS) > 16:
            _VIEWS.clear()
        macros = macros_in(srcs.values())
        numbered = {name: default_lines(text, macros) for name, text in srcs.items()}
        _VIEWS[key] = ({name: _joined(lines) for name, lines in numbered.items()}, numbered)
    return _VIEWS[key]


def default_view(texts):
    """Every source of ``texts`` as a build without test macros reads it (logical lines, no comment left); remembered per content, the rules ask for it again and again."""
    return _views(texts)[0]


def default_numbered(texts):
    """The same view as ``{file: [(first physical line in the file, logical line)]}``."""
    return _views(texts)[1]


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
    for m in re.finditer(r"^" + HASH + r"define[ \t\v\f]+(\w+)\(([^)]*)\)\s*(.*)$", view_text, re.M):
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
BRANCH_PREFIX = tokens("if constexpr (!DECODE) { return p.plan[n]; } else {")

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
    code = list(default_view(texts).values())
    found = {}
    for name in ("BT", "POS_COUNT_SHIFT", "POS_MASK"):
        for text in code:
            m = re.search(r"constexpr\s+int\s+%s\s*=\s*([^;]+);" % name, text)
            if m:
                found[name] = evaluate(m.group(1).strip(), found)
                break
        else:
            raise KeyError("the constant %s is not defined" % name)
    return found


def decode_branch(texts):
    """The tokens of the ``DECODE = true`` branch of ``rows_entry`` as the default build reads them (hooks expanded); ``None`` if the function or the branch is not there."""
    toks = expand_default_hooks(tokens(default_view(texts)["cfbi_rows.cuh"]))
    body = function_body(toks, ENTRY_HEAD)
    if body is None or body[:len(BRANCH_PREFIX)] != BRANCH_PREFIX or body[-1] != "}":
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
    c = constants(texts)
    m = re.search(r"H\s*<\s*(\d+)\s*,\s*\"problem too large", default_view(texts)["cfbi_binding.cpp"])
    if not m:
        return [("R10", "the binding has no size check of the form H < <limit> that the counter capacity can be tied to")]
    limit, shift = int(m.group(1)), c["POS_COUNT_SHIFT"]
    if (limit - 1) * (1 << shift) + c["POS_MASK"] >= 2 ** 31 - (1 << shift):
        return [("R10", "H up to %d heads can overflow the arrival counter: %d x 2^%d" % (limit - 1, limit - 1, shift))]
    return []


# ---- macros: what a #define can do to tokens that the pins read -----------------------------------------------------------------------------------------------
SEQ_WINDOW_STATEMENT = norm("if (w.slot < 0) w.len = 0;")
CONSTANT_NAMES = ("BT", "POS_COUNT_SHIFT", "POS_MASK")
HEAD_NAMES = ("seq_window", "SeqWindow", "rows_entry", "rows_kernel", "RowsParams", "static_assert", "NTHREADS", "CFBI_ROWS_MINB", "__launch_bounds__", "__global__")


def identifiers(snippet):
    return set(t for t in snippet.split() if re.match(r"[A-Za-z_]\w*$", t))


# the names that the pins read as code: the identifiers of every pinned snippet (so also the keywords and the generic names `n`, `old`, `p`: a macro of such a name would rewrite them), the
# constants, the heads of the entry and of the kernel, and the hooks that the reader expands away
PINNED = frozenset().union(*(identifiers(s) for s in (GOLDEN, SKIP, BLOCK_OPEN, ARRIVAL, FILL, ELECTION, STORE, BARRIER, STORE_HELPER, FUSED_TAIL, KERNEL_MAP, ENTRY_CALL, SEQ_WINDOW_STATEMENT,
                                                    *LAUNCH)), CONSTANT_NAMES, HEAD_NAMES, HOOK_BODIES)
ALLOWED_DEFINES = frozenset(HOOK_BODIES) | {"CFBI_ROWS_MINB"}     # the modeled hooks (their bodies are R11) and the tunable: each defined exactly once, never undefined
DIRECTIVE = re.compile(r"^" + HASH + r"(define|undef)[ \t\v\f]+([A-Za-z_]\w*)")
FLAG = re.compile(r"-[DU]\s*([A-Za-z_]\w*)")
INCLUDE = re.compile(r"^" + HASH + r"(?:include_next|include|import)\b[ \t\v\f]*(.*)$")
KEYWORD = re.compile(r"^" + HASH + r"([A-Za-z_]\w*)?")
FORCE_INCLUDE = re.compile(r"(?<![\w-])(?:--pre-include|-include|-imacros)(?![\w-])")
TRIGRAPH = re.compile(r"\?\?[=/'()!<>-]")
# what a directive may be called; any other keyword is a form that this reader does not read
DIRECTIVES = frozenset(("define", "undef", "include", "include_next", "import", "if", "ifdef", "ifndef", "elif", "elifdef", "elifndef", "else", "endif", "pragma", "error", "warning", "line"))


def stripped_lines(numbered):
    """``(file, first physical line, logical line)`` of the default views (``default_numbered``; the lines have no comment and a directive is one line, see ``logical_lines``)."""
    for name in sorted(numbered):
        for number, line in numbered[name]:
            yield name, number, line


def macro_directives(numbered):
    """``(file, line number in the file, kind, name)`` of every ``#define`` and ``#undef`` of the default views; a directive in a test-macro branch that the default build does not take is not in the view."""
    out = []
    for name, number, line in stripped_lines(numbered):
        m = DIRECTIVE.match(line)
        if m:
            out.append((name, number, m.group(1), m.group(2)))
    return out


def macro_problems(texts, numbered):
    """R12: a macro whose name is read as code by the pins would change what the compiler sees while the tokens stay (``#define st_relaxed_gpu(ptr, v) (*(ptr) = (v))`` after the helper
    turns the elected store into a plain assignment and every pin still reads a call of the relaxed helper). Every directive is judged on its own (not the last definition of a name): a
    ``#undef`` of a pinned name, a ``#define`` of one that is not allowed, a second ``#define`` of an allowed one and an allowed one that is not defined once are problems; so is a
    ``-D`` or ``-U`` flag of a pinned name."""
    out, seen = [], {}
    for f, ln, kind, name in macro_directives(numbered):
        if name not in PINNED:
            continue
        if kind == "undef" or name not in ALLOWED_DEFINES:
            out.append(("R12", "%s:%d `#%s %s`: %s is read as code by the protocol pins, so a macro of that name would rewrite it for the compiler while the tokens stay (only the hooks %s and the tunable CFBI_ROWS_MINB are defined, each once)" % (
                f, ln, kind, name, name, ", ".join(sorted(HOOK_BODIES)))))
        else:
            seen.setdefault(name, []).append((f, ln))
    for name in sorted(ALLOWED_DEFINES):
        places = seen.get(name, [])
        if len(places) != 1:
            out.append(("R12", "the macro %s is defined %d times in the default view (%s), expected once" % (name, len(places), ", ".join("%s:%d" % x for x in places) or "nowhere")))
    for key in sorted(k for k in texts if k.startswith("flags:")):
        for m in FLAG.finditer(texts[key]):
            if m.group(1) in PINNED and m.group(1) != "CFBI_ROWS_MINB":
                out.append(("R12", "%s: the build flag `%s` names %s, which the protocol pins read as code" % (key, m.group(0), m.group(1))))
        for m in FORCE_INCLUDE.finditer(texts[key]):
            out.append(("R12", "%s: the build flag `%s` forces the include of a file that the pins do not read, and it could hold any macro" % (key, m.group(0))))
    return out


def include_problems(numbered):
    """R12: every ``#include`` of the default views names a file that the reader has loaded (``cfbi/x`` is the file ``x`` of the source directory, a quoted name may also be relative to the
    including file); any other file could define a macro that no pin sees, and so could an include whose target is a macro (it cannot be read). An angle-bracket include of a file outside
    ``cfbi/`` is the toolchain's and the libraries' own."""
    out = []
    for name, number, line in stripped_lines(numbered):
        m = INCLUDE.match(line)
        if not m:
            continue
        rest = m.group(1).strip()
        quoted, angled = re.match(r'"([^"]*)"', rest), re.match(r"<([^>]*)>", rest)
        if not quoted and not angled:
            out.append(("R12", "%s:%d `%s` is an include whose target is not a file name (a macro could stand for any file), which this reader does not read" % (name, number, line.strip())))
            continue
        target = (quoted or angled).group(1)
        if angled and not target.startswith("cfbi/"):
            continue
        in_source_dir = target[len("cfbi/"):] if target.startswith("cfbi/") else target
        next_to_includer = posixpath.normpath(posixpath.join(posixpath.dirname(name), target)) if quoted else None
        if in_source_dir not in numbered and next_to_includer not in numbered:
            out.append(("R12", "%s:%d `#include %s` names a file that the reader does not load, and it could hold a macro that no pin sees" % (name, number, rest)))
    return out


def directive_problems(texts, numbered):
    """R12: a directive that is not one of ``DIRECTIVES`` (the reader would not know what it does) and a trigraph (``??/`` is a backslash, ``??=`` a ``#``: a spelling of a line splice or a
    directive that this reader does not read; the build does not translate trigraphs in C++17 mode, but nothing here may depend on that)."""
    out = []
    for name, number, line in stripped_lines(numbered):
        m = KEYWORD.match(line) if DIRECTIVE_START.match(line) else None
        if m and m.group(1) is not None and m.group(1) not in DIRECTIVES:
            out.append(("R12", "%s:%d `%s` is a directive that this reader does not read" % (name, number, line.strip()[:60])))
    for name in sorted(source_texts(texts)):
        if TRIGRAPH.search(texts[name]):
            out.append(("R12", "%s holds a trigraph, which could spell a line splice or a directive that this reader does not read" % name))
    return out


# the conditionals that remain in the default view of a file (those on build tunables, which no variant build changes); any other one would hide text from the compiler that the rules still read
# (every file that a pin reads: the constants come from cfbi_primitives.cuh, the counter limit from cfbi_binding.cpp)
ALLOWED_CONDITIONALS = {"cfbi_rows.cuh": {"ifndef CFBI_ROWS_PF_L1", "if CFBI_ROWS_PF_L1", "ifndef CFBI_ROWS_MINB"}, "cfbi_common.cuh": set(), "cfbi_launch.cu": set(), "cfbi_primitives.cuh": set(),
                       "cfbi_binding.cpp": set()}
STRAY = re.compile(r"^" + HASH + r"(if|ifdef|ifndef|elifdef|elifndef|elif)\b[ \t\v\f]*(.*)$")


def stray_conditionals(lines, allowed):
    """``[(first physical line, line)]`` of the conditionals of ``lines`` (``[(number, logical line)]`` of a default view) that are not in ``allowed``."""
    out = []
    for number, line in lines:
        m = STRAY.match(line)
        if m and ("%s %s" % m.groups()).strip() not in allowed:
            out.append((number, line.strip()))
    return out


def problems(texts):
    """What is wrong with the arrival-count protocol as the sources of ``texts`` (file name -> text) define it: ``[(rule, message)]``, empty when every rule holds."""
    out = []
    try:
        view = default_view(texts)
    except (ValueError, AssertionError) as e:
        return [("R12", "the sources cannot be read as the default build reads them: %s" % e)]
    rows = view["cfbi_rows.cuh"]
    numbered = default_numbered(texts)
    out.extend(macro_problems(texts, numbered))
    out.extend(include_problems(numbered))
    out.extend(directive_problems(texts, numbered))
    for name, allowed in ALLOWED_CONDITIONALS.items():
        for ln, line in stray_conditionals(numbered[name], allowed):
            out.append(("R12", "%s:%d holds the conditional `%s`, which the rules cannot resolve (an `#if 0` would hide a protocol line from the compiler and leave it in the text)" % (name, ln, line)))
    # R11: the default hooks are what the expansion assumes
    defs = macro_definitions(rows)
    for name, params in HOOK_DEFAULTS.items():
        if name not in defs or defs[name][0] != params or norm(defs[name][1]) != norm(HOOK_BODIES[name]):
            out.append(("R11", "the default definition of %s is not the one that the rules expand (%s(%s) %s)" % (name, name, ", ".join(params), HOOK_BODIES[name])))
    # R5: constants
    try:
        c = constants(texts)
    except KeyError as e:
        out.append(("R5", str(e)))
    else:
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
    allrows = expand_default_hooks(tokens(rows))
    rt = " ".join(allrows)
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
    sw = function_body(common, "SeqWindow seq_window (")
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
    if branch.count("pos_smem") != 2 or not (0 <= s.find(FILL) < s.find(BARRIER) < s.rfind("pos_smem")):
        out.append(("R9", "pos_smem is not written inside the thread-0 block before the barrier and read once after it"))
    if BARRIER in s and s.find(BARRIER) < s.find(ARRIVAL):
        out.append(("R9", "a barrier comes before the arrival atomic"))
    # R7: no other access to the pos word in the whole row engine; the kernel hands p to rows_entry only
    member = [i for i, t in enumerate(allrows) if t == "pos" and i > 0 and allrows[i - 1] in (".", "->")]
    if len(member) != 3:
        out.append(("R7", "cfbi_rows.cuh reaches a member `pos` %d times, expected 3 (the arrival atomic, the elected store and the fused store of the two-launch path)" % len(member)))
    if FUSED_TAIL not in rt:
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
