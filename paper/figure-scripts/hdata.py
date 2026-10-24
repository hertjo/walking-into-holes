"""Data access for the figure scripts (standard library only): REPO and a CSV reader with numeric conversion."""
import csv, json, os

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return {"True": True, "False": False}.get(v, v)
    return f


def rows(path):
    with open(os.path.join(REPO, path), newline="") as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def js(path):
    with open(os.path.join(REPO, path)) as fh:
        return json.load(fh)


def where(rs, **kw):
    return [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
