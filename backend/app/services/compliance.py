import json
from dataclasses import dataclass

from .. import settings

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


@dataclass
class Room:
    id: str
    label: str
    labels: list
    status: str
    is_corridor: bool
    area_m2: float
    width_mm: float
    mrr_dims_mm: list
    adjacent: list
    confidence: float


def load_rules():
    return json.loads(settings.RULES_PATH.read_text())


def parse_rooms(manifest):
    return [Room(id=r["id"], label=r.get("raw_label") or "UNLABELLED", labels=r.get("labels") or [],
                 status=r.get("status", "room"), is_corridor=r.get("is_corridor", False),
                 area_m2=r.get("area_m2", 0.0),
                 width_mm=r.get("inscribed_width_mm") or r.get("clear_width_mm") or 0,
                 mrr_dims_mm=r.get("mrr_dims_mm") or [], adjacent=r.get("adjacent") or [],
                 confidence=r.get("confidence", 1.0)) for r in manifest["rooms"]]


def is_a(room, keywords):
    text = " ".join(room.labels + [room.label]).upper()
    return any(k.upper() in text for k in keywords)


def result(rule_id, meta, room, passed, measured, required, detail, value=None, limit=None, neighbours=()):
    margin = round(100 * (value - limit) / limit, 1) if value is not None and limit else None
    return {"key": f"{rule_id}:{room.id}", "rule_id": rule_id, "rule": meta["title"], "room_id": room.id,
            "room": room.label, "status": "PASS" if passed else "FAIL", "measured": measured, "required": required,
            "detail": detail, "severity": meta["severity"], "why": meta["why"], "remedy": meta["remedy"],
            "clause": meta["clause"], "margin_pct": margin, "measured_value": value, "required_value": limit,
            "confidence": room.confidence, "neighbours": list(neighbours)}


def min_area(rules, room, limit):
    return result("min_area", rules["checks"]["min_area"], room, room.area_m2 >= limit, f"{room.area_m2:.2f} m2",
                  f">= {limit} m2", f"{room.label} ({room.id}): area {room.area_m2:.2f} m2, minimum {limit} m2",
                  room.area_m2, limit)


def min_width(rules, rule_id, room, width, limit):
    return result(rule_id, rules["checks"][rule_id], room, width >= limit, f"{width} mm", f">= {limit} mm",
                  f"{room.label} ({room.id}): width {width} mm, minimum {limit} mm", width, limit)


def adjacency(rule, room, index):
    near = rule["near"]
    found = [f"{index[a].label} ({a})" for a in room.adjacent if a in index and is_a(index[a], near)]
    detail = (f"{room.label} ({room.id}) is adjacent to {', '.join(found)}" if found
              else f"{room.label} ({room.id}) has no adjacent {'/'.join(near)}")
    return result(rule["id"], rule, room, bool(found), ", ".join(found) or "none found",
                  f"adjacent to one of: {', '.join(near)}", detail, neighbours=[a for a in room.adjacent if a in index])


def room_checks(room, rules):
    out = []
    for keyword, limits in rules["room_rules"].items():
        if is_a(room, [keyword]):
            if "min_area_m2" in limits:
                out.append(min_area(rules, room, limits["min_area_m2"]))
            if "min_width_mm" in limits:
                out.append(min_width(rules, "min_width", room, room.width_mm, limits["min_width_mm"]))
            break
    if room.is_corridor:
        width = room.mrr_dims_mm[0] if room.mrr_dims_mm else room.width_mm
        out.append(min_width(rules, "corridor_width", room, width, rules["min_corridor_width_mm"]))
    if is_a(room, ["STAIRCASE"]):
        out.append(min_width(rules, "staircase_width", room, room.width_mm, rules["min_staircase_width_mm"]))
    return out


def not_checked_reason(room):
    if room.status == "unlabelled":
        return "no room label"
    if room.confidence < settings.MIN_CONFIDENCE:
        return f"low confidence ({room.confidence:.2f})"
    return None


def check_manifest(manifest, rules=None):
    rules = rules or load_rules()
    rooms = parse_rooms(manifest)
    index = {r.id: r for r in rooms}
    eligible = [r for r in rooms if not not_checked_reason(r)]
    checks = [c for r in eligible for c in room_checks(r, rules)]
    for rule in rules["adjacency_rules"]:
        checks += [adjacency(rule, r, index) for r in eligible if is_a(r, rule["source"])]
    checks.sort(key=lambda c: (c["status"] != "FAIL", SEVERITY_ORDER[c["severity"]], c["margin_pct"] if c["margin_pct"] is not None else 0))
    violations = [c for c in checks if c["status"] == "FAIL"]
    by_severity = {s: sum(v["severity"] == s for v in violations) for s in SEVERITY_ORDER}
    return {"total": len(checks), "passed": len(checks) - len(violations), "failed": len(violations),
            "by_severity": by_severity,
            "not_checked": [{"room_id": r.id, "room": r.label, "reason": not_checked_reason(r)} for r in rooms if not_checked_reason(r)],
            "violations": violations, "checks": checks}
