"""Resolve module names stored before the OHAGI/PURIN package rename."""


def canonical_module_name(name: str) -> str:
    """Translate historical factory namespaces without changing attributes."""
    for previous, current in (("hax", "ohagi"), ("projx", "purin")):
        if name == previous or name.startswith(previous + "."):
            return current + name[len(previous) :]
    return name
