"""OCR_ENGINE=hybrid: Unlimited-OCR's text placed on RapidOCR's geometry.

Both recognizers read the same page image. Unlimited-OCR's grounded blocks
decide what the text layer says; RapidOCR's line quads and character boxes
decide where it goes. The RapidOCR lines whose centers fall inside a block are
aligned to that block's text by edit distance, and every aligned character
keeps Unlimited's reading at RapidOCR's position. Block text with no RapidOCR
counterpart keeps the block placement, so recognized text is never dropped;
text only RapidOCR read is never added.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
from difflib import SequenceMatcher
import math
import re
import struct
import unicodedata

from engine import _MODEL_READY_TIMEOUT_SECONDS, Block, OcrError, UnlimitedOCR

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # Optional: difflib yields the same opcode format.
    Levenshtein = None


# A run of substitutions longer than this on either side is more likely
# different text (for example table cells read in another order) than a
# misread of the same glyphs; misreads are short runs between matches.
_MAX_REPLACE = 4
# Alignment rounds per block. Each later round aligns text still without a
# place against the RapidOCR characters no earlier round has taken.
_ROUNDS = 4
# Free space a missed character needs, as a fraction of the line height.
_MIN_ADVANCE = 0.4


def _key(char):
    # Width and compatibility variants of one glyph (（ and (, ０ and 0) align.
    folded = unicodedata.normalize("NFKC", char)
    return folded if len(folded) == 1 else char


def _opcodes(a: str, b: str):
    if Levenshtein is None:
        return SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return [(op.tag, op.src_start, op.src_end, op.dest_start, op.dest_end)
            for op in Levenshtein.opcodes(a, b)]


def _wide(char):
    return bool(char) and unicodedata.east_asian_width(char) in {"W", "F"}


def _line_text(text):
    # One RapidOCR line: a line break inside Unlimited's text between two
    # CJK characters is not a word gap; any other whitespace is one space.
    def gap(match):
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        return "" if "\n" in match.group() and _wide(before) and _wide(after) else " "
    return re.sub(r"\s+", gap, text).strip()


class _Line:
    """A RapidOCR line in page-image pixels, measured along its reading axis."""

    def __init__(self, block, width, height):
        self.block, self.width, self.height = block, width, height
        points = [(x * width, y * height) for x, y in block.quad]
        (ax, ay), (bx, by), _, (dx, dy) = points
        self.origin = (ax, ay)
        self.length = math.hypot(bx - ax, by - ay)
        self.thickness = math.hypot(dx - ax, dy - ay)
        self.keys = [char for char in block.text if not char.isspace()]
        if self.length <= 0 or self.thickness <= 0:
            self.keys = []  # Degenerate geometry: the line is not used.
            return
        self.u = ((bx - ax) / self.length, (by - ay) / self.length)
        self.across = (dx - ax, dy - ay)
        self.center = (sum(p[0] for p in points) / 4, sum(p[1] for p in points) / 4)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        spans = [self._project(box) for _, box in block.chars]
        # Same acceptance rule as the text layer: character boxes are used
        # only when every one has extent and starts never move backwards.
        self.trusted = bool(spans) and all(e > s for s, e in spans) and \
            all(b[0] >= a[0] for a, b in zip(spans, spans[1:]))
        if self.trusted:
            self.spans, self.boxes = spans, [box for _, box in block.chars]
        else:
            step = self.length / len(self.keys)
            self.spans = [(k * step, (k + 1) * step) for k in range(len(self.keys))]
            self.boxes = [None] * len(self.keys)
        # Characters already given to block text; rest() views share them.
        self.root, self.index, self.taken = self, list(range(len(self.keys))), set()

    def rest(self):
        """This line without the characters already taken, or None."""
        keep = [k for k, r in enumerate(self.index) if r not in self.root.taken]
        if not keep:
            return None
        view = copy.copy(self)
        for name in ("keys", "spans", "boxes", "index"):
            setattr(view, name, [getattr(self, name)[k] for k in keep])
        return view

    def take(self, k):
        self.root.taken.add(self.index[k])

    def _project(self, box):
        x0, y0, x1, y1 = box
        along = [(x * self.width - self.origin[0]) * self.u[0] +
                 (y * self.height - self.origin[1]) * self.u[1]
                 for x in (x0, x1) for y in (y0, y1)]
        return max(0.0, min(along)), min(self.length, max(along))

    def quad(self, start, end):
        """The part of the line between two distances along it, normalized."""
        (ax, ay), (ux, uy), (nx, ny) = self.origin, self.u, self.across
        corners = ((start, 0), (end, 0), (end, 1), (start, 1))
        return tuple((min(max((ax + ux * t + nx * k) / self.width, 0.0), 1.0),
                      min(max((ay + uy * t + ny * k) / self.height, 0.0), 1.0))
                     for t, k in corners)

    def box(self, start, end):
        quad = self.quad(start, end)
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        return (min(xs), min(ys), max(xs), max(ys))


def _assign(blocks, lines, width, height):
    """Each RapidOCR line belongs to the smallest block containing its center,
    with a tolerance of half the line height for coarse block boxes."""
    owned = [[] for _ in blocks]
    for index, line in enumerate(lines):
        margin = line.thickness / 2
        best = None
        for owner, block in enumerate(blocks):
            x0, y0, x1, y1 = (block.box[0] * width, block.box[1] * height,
                              block.box[2] * width, block.box[3] * height)
            if x0 - margin <= line.center[0] <= x1 + margin and \
               y0 - margin <= line.center[1] <= y1 + margin:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best[0]:
                    best = (area, owner)
        if best is not None:
            owned[best[1]].append(index)
    return owned


def _align(chars, lines):
    """Place each block character on a line: (line, start, end, RapidOCR box
    or None), or None where RapidOCR has no counterpart."""
    targets = [(n, k) for n, line in enumerate(lines) for k in range(len(line.keys))]
    a = "".join(_key(c) for c in chars)
    b = "".join(_key(lines[n].keys[k]) for n, k in targets)
    placed = [None] * len(chars)
    for tag, i1, i2, j1, j2 in _opcodes(a, b):
        count, available = i2 - i1, j2 - j1
        if tag not in {"equal", "replace"} or \
           (tag == "replace" and max(count, available) > _MAX_REPLACE):
            continue
        if count == available:
            # A match or a one-for-one misread (虚 for 虛, 一 for —).
            for k in range(count):
                n, r = targets[j1 + k]
                lines[n].take(r)
                placed[i1 + k] = (n, *lines[n].spans[r], lines[n].boxes[r])
            continue
        # An unequal misread shares the RapidOCR span in proportion, on the
        # line holding most of that span.
        owners = [targets[j][0] for j in range(j1, j2)]
        n = max(owners, key=owners.count)
        mine = [targets[j][1] for j in range(j1, j2) if targets[j][0] == n]
        for r in mine:
            lines[n].take(r)
        mine = [lines[n].spans[r] for r in mine]
        for k, span in enumerate(_spread(mine[0][0], mine[-1][1], count)):
            placed[i1 + k] = (n, *span)
    _prefer_line_breaks(a, placed)
    return placed


def _prefer_line_breaks(keys, placed):
    """Edit-distance alignments are not unique: characters RapidOCR lacks may
    be reported anywhere among equally good positions, for example inside the
    next line (第一行|第[二行漏檢…第]三行) instead of between the lines
    (第一行|[第二行漏檢…]|第三行). Slide each such run, keeping every
    character on a RapidOCR character with the same key, to the nearest
    position between two lines."""
    def line_at(i):
        return placed[i][0] if 0 <= i < len(placed) and placed[i] else None

    k = 0
    while k < len(placed):
        if placed[k] is not None:
            k += 1
            continue
        a = k
        while k < len(placed) and placed[k] is None:
            k += 1
        b = k
        if a == 0 or b == len(placed) or line_at(a - 1) != line_at(b):
            continue
        options = []
        s = 1
        while a - s >= 0 and placed[a - s] and keys[a - s] == keys[b - s]:
            if line_at(a - s - 1) != line_at(a - s):
                options.append(-s)
                break
            s += 1
        s = 1
        while b + s - 1 < len(placed) and placed[b + s - 1] and keys[a + s - 1] == keys[b + s - 1]:
            if line_at(b + s - 1) != line_at(b + s):
                options.append(s)
                break
            s += 1
        if not options:
            continue
        shift = min(options, key=abs)
        for t in range(1, -shift + 1):
            placed[b - t], placed[a - t] = placed[a - t], None
        for t in range(shift):
            placed[a + t], placed[b + t] = placed[b + t], None
        k = b + shift


def _spread(start, end, count):
    step = (end - start) / count
    return [(start + k * step, start + (k + 1) * step, None) for k in range(count)]


def _fill(spans, line):
    """Place characters RapidOCR missed inside one line: in free space next
    to their neighbours when there is room, otherwise by sharing the
    neighbouring character's span. Items are (start, end, box) or None."""
    k = 0
    while k < len(spans):
        if spans[k] is not None:
            k += 1
            continue
        first = k
        while k < len(spans) and spans[k] is None:
            k += 1
        count = k - first
        before = spans[first - 1] if first else None
        after = spans[k] if k < len(spans) else None
        low = before[1] if before else 0.0
        high = after[0] if after else line.length
        if high - low >= _MIN_ADVANCE * line.thickness * count:
            spans[first:k] = _spread(low, high, count)
        elif after:
            spans[first:k + 1] = _spread(min(low, after[0]), after[1], count + 1)
        else:
            spans[first - 1:k] = _spread(before[0], max(before[1], line.length), count + 1)
    return spans


def _merge_block(block, lines, rounds=_ROUNDS):
    text = block.text
    chars = [(index, char) for index, char in enumerate(text) if not char.isspace()]
    placed = _align([c for _, c in chars], lines)

    # Contiguous runs of block characters on one line, or unplaced.
    runs = []
    for position, target in enumerate(placed):
        owner = target[0] if target else None
        if runs and runs[-1][0] == owner:
            runs[-1][1].append(position)
        else:
            runs.append([owner, [position]])
    for index, (owner, members) in enumerate(runs):
        if owner is not None:
            continue
        before = runs[index - 1][0] if index else None
        after = runs[index + 1][0] if index + 1 < len(runs) else None
        if before is not None and before == after:
            runs[index][0] = before
            continue
        # Between lines: join a line that has room for the run at that end
        # (RapidOCR detected the ink but did not read it). A single character
        # may share its neighbour's span; longer runs keep the block placement.
        need = len(members) * _MIN_ADVANCE
        if before is not None and lines[before].length - placed[runs[index - 1][1][-1]][2] >= \
                need * lines[before].thickness:
            runs[index][0] = before
        elif after is not None and placed[runs[index + 1][1][0]][1] >= need * lines[after].thickness:
            runs[index][0] = after
        elif len(members) == 1 and (before is not None or after is not None):
            runs[index][0] = before if before is not None else after
    merged = []
    for owner, members in runs:
        if merged and owner is not None and merged[-1][0] == owner:
            merged[-1][1].extend(members)
        else:
            merged.append([owner, list(members)])

    pieces = []
    for index, (owner, members) in enumerate(merged):
        first, last = chars[members[0]][0], chars[members[-1]][0]
        if owner is None:
            piece = Block(text[first:last + 1].strip(), _gap_box(block, merged, index, lines))
            rest = [view for view in (line.rest() for line in lines) if view]
            # Reading order may differ: Unlimited reads a table cell by cell,
            # RapidOCR row by row across cells. Align the remainder again
            # against the RapidOCR characters no other text has taken.
            pieces.extend(_merge_block(piece, rest, rounds - 1) if rounds > 1 and rest else [piece])
            continue
        line = lines[owner]
        spans = _fill([placed[m][1:] if placed[m] else None for m in members], line)
        if line.trusted:
            boxes = tuple((chars[m][1], box if box else line.box(start, end))
                          for m, (start, end, box) in zip(members, spans))
            pieces.append(Block(_line_text(text[first:last + 1]), line.block.box,
                                line.block.quad, boxes))
        else:
            # Without character boxes only the covered part of the line is
            # known; the text layer stretches the text across it.
            start, end = spans[0][0], spans[-1][1]
            pieces.append(Block(_line_text(text[first:last + 1]), line.box(start, end),
                                line.quad(start, end)))
    # Defensive: whatever happened above, the block's text survives intact.
    if "".join("".join(p.text.split()) for p in pieces) != "".join(text.split()):
        return [block]
    return pieces


def _gap_box(block, merged, index, lines):
    """Block placement for text RapidOCR did not detect, narrowed to the
    band between the neighbouring placed lines when that band is open."""
    width, height = (lines[0].width, lines[0].height) if lines else (1.0, 1.0)
    x0, y0, x1, y1 = block.box
    placed = [lines[o] for o, _ in merged if o is not None]
    if not placed:
        return block.box
    above = [lines[o].bounds[3] / height for o, _ in merged[:index] if o is not None]
    below = [lines[o].bounds[1] / height for o, _ in merged[index + 1:] if o is not None]
    top = max(above) if above else y0
    bottom = min(below) if below else y1
    if y0 <= top < bottom <= y1 and \
       bottom - top >= min(line.thickness for line in placed) / height / 2:
        return (x0, top, x1, bottom)
    return block.box


def merge(blocks: list[Block], rapid: list[Block], width=1.0, height=1.0) -> list[Block]:
    """Unlimited-OCR blocks with RapidOCR line geometry, in Unlimited's order.

    width and height give the page image aspect so that distances along and
    across lines are measured in pixels.
    """
    lines = [line for line in (_Line(block, width, height) for block in rapid
                               if block.quad is not None) if line.keys]
    result = []
    for block, owned in zip(blocks, _assign(blocks, lines, width, height)):
        result.extend(_merge_block(block, [lines[n] for n in owned]) if owned else [block])
    return result


def _png_size(png: bytes):
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise OcrError("invalid_model_output")
    width, height = struct.unpack(">II", png[16:24])
    if not width or not height:
        raise OcrError("invalid_model_output")
    return width, height


class HybridEngine:
    name = "Unlimited-OCR + RapidOCR PP-OCRv6"

    def __init__(self, unlimited=None, rapid=None):
        if rapid is None:
            from rapidocr_engine import RapidOCREngine
            rapid = RapidOCREngine()
        self.unlimited = unlimited or UnlimitedOCR()
        self.rapid = rapid
        self.cpu_threads = rapid.cpu_threads
        self.worker_address_space = max(self.unlimited.worker_address_space,
                                        rapid.worker_address_space)

    def configured(self) -> bool:
        return self.unlimited.configured() and self.rapid.configured()

    def ready(self, timeout=_MODEL_READY_TIMEOUT_SECONDS) -> bool:
        return self.rapid.ready() and self.unlimited.ready(timeout=timeout)

    def recognize(self, png: bytes) -> list[Block]:
        width, height = _png_size(png)
        # The model request waits on the network; run the local CPU engine
        # meanwhile. A failure of either one fails the page with its code.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.unlimited.recognize, png)
            try:
                lines = self.rapid.recognize(png)
            finally:
                blocks = pending.result()
        return merge(blocks, lines, width, height)
