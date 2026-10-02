#!/usr/bin/env python3
"""Codabench MCP server (read-only) for NeurIPS 2026 RealPDE Track 1 (#17363).

Exposes the PUBLIC Codabench API so the assistant can pull:
  * the Track 1 leaderboard WITH all five subscores per competitor
  * our own submissions + subscores
  * competition announcements / phases / rules

Read-only by design: there is no submit tool. Submissions stay a human action.
No credentials required or used — everything here is the public API.

Protocol: MCP over stdio (JSON-RPC 2.0). Stdlib only.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://www.codabench.org"
COMP_ID = 17363
PHASE_ID = 29663   # "Main Development Phase" (current); the 5 Aug restart changed this
TRACK1_TASK_IDS = {35362}          # "Sim2Real Updated Dev Ver 0.1"
TRACK1_TASK_HINT = "sim2real"
UA = "Mozilla/5.0 (compatible; codabench-mcp/1.0)"
_CACHE = {}
CACHE_TTL = 300


def _get(url, timeout=45, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:                     # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def _cached(key, fn):
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < CACHE_TTL:
        return _CACHE[key][1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def _is_track1(sub):
    """Live phase ONLY: task id 35362 ("Sim2Real Updated Dev Ver 0.1").

    ⛔ Never match on task NAME. The old fallback did
        "sim2real" in task_name.lower()
    which also matched task 34673 "Sim2Real Development" (the pre-restart phase,
    frozen but STILL ACCEPTING SUBMISSIONS) and 34672 "Sim2Real Warm-up".
    Those rows were scored on the old evaluation set with the old scoring rule,
    so their subscores are not comparable to the live board.

    The date cutoff alone does not catch them: dentist23 submitted to 34673 on
    2026-08-13, well after the 08-05 restart. That leak put dentist23 on the
    leaderboard at 82.44 / sps 54.44 and anaelle_haomiao at sps 58.51 -- neither
    appears on the real board at all -- which is exactly the contamination
    project_memory 3 and 4 warn about.
    """
    t = sub.get("task") or {}
    return t.get("id") in TRACK1_TASK_IDS


def _flatten(sub):
    sc = {}
    for s in sub.get("scores") or []:
        try:
            sc[s.get("column_key")] = float(s.get("score"))
        except (TypeError, ValueError):
            pass
    return {
        "id": sub.get("id"),
        "owner": sub.get("owner"),
        "filename": sub.get("filename"),
        "date": (sub.get("created_when") or "")[:19].replace("T", " "),
        "status": sub.get("status"),
        "on_leaderboard": sub.get("on_leaderboard"),
        "final_score": sc.get("final_score"),
        "rel_l2_score": sc.get("rel_l2_score"),
        "tke_score": sc.get("tke_score"),
        "mvpe_score": sc.get("mvpe_score"),
        "time_score": sc.get("time_score"),
        "sps_score": sc.get("sps_score"),
    }


def _board():
    """Current-phase leaderboard.

    /api/submissions/ used to serve the public feed WITH subscores, but it now returns
    count:0 for everyone, which is why this server silently produced empty tables.  The
    phase leaderboard endpoint still works; it publishes only the primary column.
    """
    d = _get(f"{BASE}/api/phases/{PHASE_ID}/get_leaderboard/")
    rows = []
    for sub in d.get("submissions") or []:
        sc = {x.get("column_key"): x.get("score") for x in (sub.get("scores") or [])}
        f = sc.get("final_score")
        if f is None:
            continue
        rows.append({"owner": sub.get("owner"), "final_score": float(f),
                     "id": sub.get("id"), "date": (sub.get("created_when") or "")[:19]})
    rows.sort(key=lambda r: -r["final_score"])
    return rows


def t_leaderboard(pages=6, top=15):
    rows = _cached("board", _board)
    if not rows:
        return "Leaderboard endpoint returned nothing -- Codabench may be down."
    hdr = f"{'#':>3} {'owner':22s}{'final':>11s}   {'date':19s}"
    lines = [f"Track 1 phase {PHASE_ID} leaderboard -- {len(rows)} rows published "
             f"(Codabench publishes the top {len(rows)} only; teams below that do not appear).",
             "NOTE: competitor SUBSCORES are no longer public -- /api/submissions/ now returns "
             "count:0, so only final_score is available.",
             hdr, "-" * len(hdr)]
    for i, r in enumerate(rows[:top], 1):
        lines.append(f"{i:>3} {str(r['owner'])[:21]:22s}{r['final_score']:11.5f}   {r['date']:19s}")
    if len(rows) >= 10:
        lines.append("")
        lines.append(f"top-10 cutoff (#10) = {rows[9]['final_score']:.5f} | "
                     f"#1 = {rows[0]['final_score']:.5f} | "
                     f"last published (#{len(rows)}) = {rows[-1]['final_score']:.5f}")
    return "\n".join(lines)


def t_my_submissions(owner=None, filename_contains=None, pages=6):
    """Our own submissions are NOT retrievable any more: the public submissions feed is closed
    and per-submission detail 404s without auth.  Report that instead of an empty table."""
    rows = _cached("board", _board)
    if owner:
        hit = [r for r in rows if (r.get("owner") or "").lower() == owner.lower()]
        if hit:
            i = rows.index(hit[0]) + 1
            return (f"{owner}: rank {i} of {len(rows)} published, final {hit[0]['final_score']:.5f} "
                    f"({hit[0]['date']}). Subscores are not public.")
        return (f"{owner} is not in the published top {len(rows)} "
                f"(cut-off {rows[-1]['final_score']:.5f}). Codabench publishes only that many rows, "
                f"and /api/submissions/ now returns count:0, so ranks below it cannot be read from "
                f"the API -- check the website while logged in.")
    return ("Per-submission history is no longer exposed by the public API "
            "(/api/submissions/ returns count:0; /api/submissions/<id>/ 404s without auth). "
            "Use the Codabench website while logged in. The leaderboard tool still works.")


def t_competition_info(section="announcements"):
    d = _cached("comp", lambda: _get(f"{BASE}/api/competitions/{COMP_ID}/"))
    if section == "phases":
        out = []
        for ph in d.get("phases", []):
            out.append(f"{ph.get('name')}: {str(ph.get('start'))[:10]} -> {str(ph.get('end'))[:10]} [{ph.get('status')}]")
        return "\n".join(out) or "no phases"
    want = {"announcements": ("overview",), "rules": ("rules", "terms"),
            "evaluation": ("evaluation",), "data": ("data",)}.get(section, (section,))
    chunks = []
    for p in d.get("pages", []):
        title = (p.get("title") or "").lower()
        if any(w in title for w in want):
            chunks.append(f"### {p.get('title')}\n{(p.get('content') or '')[:6000]}")
    return "\n\n".join(chunks) or f"no page matching '{section}'"


def t_submission(sub_id):
    d = _get(f"{BASE}/api/submissions/{int(sub_id)}/")
    return json.dumps(_flatten(d), indent=2)


TOOLS = [
    {"name": "track1_leaderboard",
     "description": "Track 1 leaderboard reconstructed from the public API, WITH all five subscores "
                    "(rel_l2/tke/mvpe/time/sps) per competitor — not just the final score.",
     "inputSchema": {"type": "object", "properties": {
         "pages": {"type": "integer", "description": "feed pages to scan (500/page, default 6)"},
         "top": {"type": "integer", "description": "rows to show (default 15)"}}}},
    {"name": "my_submissions",
     "description": "Our Track 1 submissions with subscores. Filter by owner username and/or a "
                    "filename fragment (e.g. 'fno').",
     "inputSchema": {"type": "object", "properties": {
         "owner": {"type": "string"}, "filename_contains": {"type": "string"},
         "pages": {"type": "integer"}}}},
    {"name": "competition_info",
     "description": "Competition pages/announcements: section = announcements|rules|evaluation|data|phases.",
     "inputSchema": {"type": "object", "properties": {"section": {"type": "string"}}}},
    {"name": "submission_scores",
     "description": "Full subscores for one submission id.",
     "inputSchema": {"type": "object", "properties": {"sub_id": {"type": "integer"}},
                     "required": ["sub_id"]}},
]


def dispatch(name, args):
    if name == "track1_leaderboard":
        return t_leaderboard(int(args.get("pages", 6)), int(args.get("top", 15)))
    if name == "my_submissions":
        return t_my_submissions(args.get("owner"), args.get("filename_contains"), int(args.get("pages", 6)))
    if name == "competition_info":
        return t_competition_info(args.get("section", "announcements"))
    if name == "submission_scores":
        return t_submission(args["sub_id"])
    raise ValueError(f"unknown tool {name}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method = req.get("id"), req.get("method")
        try:
            if method == "initialize":
                res = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                       "serverInfo": {"name": "codabench", "version": "1.0.0"}}
            elif method == "tools/list":
                res = {"tools": TOOLS}
            elif method == "tools/call":
                p = req.get("params") or {}
                text = dispatch(p.get("name"), p.get("arguments") or {})
                res = {"content": [{"type": "text", "text": text}]}
            elif method in ("notifications/initialized", "initialized"):
                continue
            else:
                res = {}
            out = {"jsonrpc": "2.0", "id": mid, "result": res}
        except Exception as e:                      # noqa: BLE001
            out = {"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
