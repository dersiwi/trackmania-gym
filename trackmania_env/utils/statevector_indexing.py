from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, Iterable, List, ClassVar

LengthSpec = Union[int, str, Callable[[Any], int]]  # 4, "road_features_len", or lambda self: int

@dataclass(frozen=True)
class FieldSpec:
    name: str
    length: LengthSpec
    label: Optional[str] = None  # used by label_block; defaults to name

class _LayoutMeta(type):
    def __new__(mcls, name, bases, ns):
        # 1) inherit fields
        inherited: List[FieldSpec] = []
        for b in bases:
            inherited.extend(getattr(b, "__fields__", []))

        # 2) normalize local FIELDS
        local: List[FieldSpec] = []
        for item in ns.get("FIELDS", []):
            local.append(item if isinstance(item, FieldSpec) else FieldSpec(*item))

        fields = inherited + local
        ns["__fields__"] = tuple(fields)

        # 3) helpers that work with dynamic lengths
        def _resolve_len(self, f: FieldSpec) -> int:
            L = f.length
            if isinstance(L, int):
                return L
            if isinstance(L, str):
                return int(getattr(self, L))
            # callable(self) -> int
            return int(L(self))

        def _offset(self, field_name: str) -> int:
            off = self.base
            for f in self.__fields__:
                if f.name == field_name:
                    break
                off += _resolve_len(self, f)
            return off

        def _slice(self, field_name: str) -> slice:
            start = _offset(self, field_name)
            L = _resolve_len(self, next(f for f in self.__fields__ if f.name == field_name))
            return slice(start, start + L)

        ns["_resolve_len"] = _resolve_len
        ns["_offset"] = _offset
        ns["_slice"] = _slice

        # 4) generate properties:
        #    - If declared length == 1 (constant int), expose `<name>_idx`
        #    - Otherwise expose `<name>` returning a slice (works for dynamic lengths)
        for f in fields:
            if isinstance(f.length, int) and f.length == 1:
                ns[f"{f.name}_idx"] = property(lambda self, n=f.name: _offset(self, n))
            else:
                ns[f.name] = property(lambda self, n=f.name: _slice(self, n))

        # 5) convenience
        ns["stop"] = property(lambda self: self.base + sum(_resolve_len(self, f) for f in self.__fields__))
        ns["length"] = property(lambda self: sum(_resolve_len(self, f) for f in self.__fields__))

        # 6) labeler that adapts to dynamic lengths
        def _label_block(osv, idx: int) -> str:
            b = osv.base
            for f in osv.__fields__:
                L = _resolve_len(osv, f)
                start, end = b + osv._offset(f.name) - osv.base, b + osv._offset(f.name) - osv.base + L
                # compute absolute start once to avoid double work:
                start_abs = osv._offset(f.name)
                end_abs = start_abs + L
                if L == 1:
                    if idx == start_abs:
                        return f.label or f.name
                else:
                    if start_abs <= idx < end_abs:
                        return f.label or f.name
            return "unknown"
        ns["label_block"] = staticmethod(_label_block)

        return super().__new__(mcls, name, bases, ns)

class IOSVBase(metaclass=_LayoutMeta):
    """
    Declarative, extensible layout with dynamic lengths.

    Scalars: length=1 -> exposes `<name>_idx`
    Ranges:  length!=1 (including dynamic) -> exposes `<name>` slice
    """
    label_block: ClassVar[Callable[["IOSVBase", int], str]]
    FIELDS = [
        ("sliding",            4),
        ("ground_contact",     4),
        ("damper_absorb",      4),
        ("gearbox_state",      1),
        ("gear",               1),
        ("actual_rpm",         1),
        ("is_freewheeling",    1),
        ("surface_categories", 4, "surface_category"),
        ("speed",              1),
        # Subclasses can append dynamic ranges like ("road_features", "road_features_len")
    ]

    def __init__(self, base: int = 0, **kwargs):
        self.base = base
        # Accept arbitrary parameters (e.g., road_features_len) that dynamic fields may reference.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def shifted(self, delta: int):
        return type(self)(self.base + delta, **{
            k: getattr(self, k) for k in self.__dict__ if k not in ("base",)
        })

    def append(self, other: "IOSVBase"):
        return type(other)(self.stop + (other.base - 0), **{
            k: getattr(other, k) for k in other.__dict__ if k not in ("base",)
        })
    
    def dimension(self) -> int:
        """Total size of this observation vector (number of elements)."""
        # Equivalent to `self.length`, but explicit as a method.
        return sum(self._resolve_len(f) for f in self.__fields__)

    @property
    def shape(self) -> tuple[int]:
        """NumPy-style shape for a 1D state vector."""
        return (self.dimension(),)
    
    def iter_blocks(self):
        """
        Yield dicts describing each block in order:
        {'name','label','start','stop','length','is_scalar','slice'}.
        Works with dynamic lengths.
        """
        off = self.base
        for f in self.__fields__:
            L = self._resolve_len(f)
            start, stop = off, off + L
            yield {
                "name": f.name,
                "label": (f.label or f.name),
                "start": start,
                "stop": stop,
                "length": L,
                "is_scalar": (L == 1),
                "slice": slice(start, stop),
            }
            off = stop
    
    def layout_str(self, sep: str = " | ") -> str:
        """
        Compact one-line timeline of the vector layout.
        Example: '  0- 3 sliding[4] |  4- 7 ground_contact[4] |  8 gearbox_state | ...'
        """
        parts = []
        for b in self.iter_blocks():
            if b["is_scalar"]:
                parts.append(f"{b['start']:>3} {b['label']}")
            else:
                parts.append(f"{b['start']:>3}-{b['stop']-1:>3} {b['label']}[{b['length']}]")
        return sep.join(parts)
    
    def layout_table(self) -> list[dict]:
        """
        Structured table you can pretty-print or turn into a DataFrame.
        Each row: {'field','label','start','stop','length','kind'}
        """
        rows = []
        for b in self.iter_blocks():
            rows.append({
                "field": b["name"],
                "label": b["label"],
                "start": b["start"],
                "stop": b["stop"],          # one-past-last
                "length": b["length"],
                "kind": "scalar" if b["is_scalar"] else "range",
            })
        return rows

    # ----- print values with ranges -----
    def _fmt_val(self, x: float, fmt) -> str:
        if callable(fmt):
            return fmt(x)
        return format(x, fmt) if fmt else str(x)
        
    def format_vector(self, obs, fmt = ".3g") -> str:
        """
        Return a multi-line string with values grouped by ranges.
        `fmt` is a format spec ('.3g', '.4f', etc.) or a callable(float)->str.
        """
        if len(obs) < self.stop:
            raise ValueError(f"obs length {len(obs)} < required {self.stop}")
        lines = []
        for b in self.iter_blocks():
            if b["is_scalar"]:
                v = self._fmt_val(obs[b["start"]], fmt)
                lines.append(f"{b['start']:>3} {b['label']}: {v}")
            else:
                vals = " ".join(self._fmt_val(x, fmt) for x in obs[b["slice"]])
                lines.append(f"{b['start']:>3}-{b['stop']-1:>3} {b['label']}[{b['length']}]: [{vals}]")
        return "\n".join(lines)


class ExtendedIOSV(IOSVBase):
    FIELDS = IOSVBase.FIELDS + [
        ("road_features", "road_features_len", "road_feature"),  # dynamic range : (road_features is the attribute name, "road_features_len" name of the attribute provided at runtime, "road_feature" label-block override)
        ("lateral_dist", 1),
    ]


eiosv = ExtendedIOSV(road_features_len = 0)
print(eiosv.lateral_dist_idx)
print(eiosv.layout_str())
print(eiosv.format_vector())