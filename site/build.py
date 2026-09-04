"""Build the PACT site: inject benchmark results into template.html -> index.html.

Reads results/benchmark/*.csv (the aggregate stage outputs), site/models_meta.json
(hand-entered from paper Table 3), and site/examples.json (curated transcripts),
and replaces the /*__PACT_DATA__*/ marker in site/template.html with a JSON blob.

Rerun after every -Aggregate to refresh the site:  python site/build.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmark"
SITE = ROOT / "site"

AXES = [
    ("default_compliance", "Default Compliance"),
    ("pressure_resistance", "Pressure Resistance"),
    ("pushback_resistance", "Pushback Resistance"),
    ("steerability", "Steerability"),
    ("transparency", "Transparency"),
    ("rule_scope_discernment", "Rule-Scope Discernment"),
]

DOMAINS = {  # keep in sync with src/benchmark/figstyle.py DOMAIN_LABEL
    "advertising": "Advertising", "aml_kyc": "AML", "customer_service": "Customer Service",
    "export_controls": "Export Controls", "finance": "Finance", "gov_services": "Gov. Services",
    "healthcare_admin": "Healthcare", "hr_hiring": "HR/Hiring", "moderation": "Moderation",
    "pharma_medinfo": "Pharma", "privacy": "Privacy", "procurement": "Procurement",
}
PRESSURES = {  # keep in sync with figstyle.PRESSURE_LABEL
    "urgency": "Urgency", "mgr_verbal": "Manager", "peer_escaped": "Peer Escaped",
    "risk_minimize": "Risk Minimized", "cost_pressure": "Cost Pressure",
    "false_clearance": "False Clearance", "fait_accompli": "Fait Accompli",
    "sympathetic_beneficiary": "Sympathetic Beneficiary", "responsibility_shift": "Responsibility Shift",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if any(v.strip() for v in row.values())]


def f3(x: str | float) -> float:
    return round(float(x), 4)


def main() -> None:
    meta = json.loads((SITE / "models_meta.json").read_text(encoding="utf-8"))
    meta.pop("__comment", None)

    metrics = {r["model"]: r for r in read_csv(BENCH / "metrics_v2.csv")}
    dom_rows = {r["model"]: r for r in read_csv(BENCH / "dist_model_domain.csv")}
    prs_rows = {r["model"]: r for r in read_csv(BENCH / "dist_model_pressure.csv")}

    missing = set(metrics) ^ set(meta)
    if missing:
        raise SystemExit(f"models_meta.json out of sync with metrics_v2.csv: {sorted(missing)}")

    models = []
    for mid, m in metrics.items():
        info = meta[mid]
        models.append({
            "id": mid,
            "name": info["display"],
            "provider": info["provider"],
            "open": info["open"],
            "params": info["params_total"],
            "active": info["params_active"],
            "released": info["released"],
            "pact": f3(m["pact_score"]),
            "pact_lo": f3(m["pact_score_lo"]),
            "pact_hi": f3(m["pact_score_hi"]),
            "axes": [f3(m[k]) for k, _ in AXES],
            "domains": [f3(dom_rows[mid][d]) for d in DOMAINS],
            "pressures": [f3(prs_rows[mid][p]) for p in PRESSURES],
        })
    models.sort(key=lambda r: -r["pact"])

    pressure_panel = [
        {"key": r["pressure"], "t1": f3(r["t1_comply"]), "t2": f3(r["t2_hold"])}
        for r in read_csv(BENCH / "dist_pressure.csv")
    ]

    domain_panel = [
        {"label": DOMAINS[r["domain"]], "base": f3(r["default_compliance"]),
         "pressured": f3(r["pressure_resistance"])}
        for r in read_csv(BENCH / "dist_domain_axis.csv")
    ]

    data = {
        "axes": [label for _, label in AXES],
        "domains": list(DOMAINS.values()),
        "pressures": list(PRESSURES.values()),
        "models": models,
        "pressure_panel": pressure_panel,
        "domain_panel": domain_panel,
        "examples": json.loads((SITE / "examples.json").read_text(encoding="utf-8")),
    }

    template = (SITE / "template.html").read_text(encoding="utf-8")
    marker = "/*__PACT_DATA__*/"
    if marker not in template:
        raise SystemExit("template.html is missing the /*__PACT_DATA__*/ marker")
    html = template.replace(marker, "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    (SITE / "index.html").write_text(html, encoding="utf-8")

    # GitHub Pages serves from / or /docs on a branch; keep a copy there.
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(html, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site/index.html + docs/index.html written: {len(models)} models, "
          f"{len(data['examples'])} examples, {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
