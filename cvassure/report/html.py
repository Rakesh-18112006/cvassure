"""One self-contained HTML file.

Every rule here exists because of the air-gap requirement: CSS is inline,
images are base64-embedded, the sort script is a dozen lines of vanilla
JavaScript, and there are no web fonts. Open it on a laptop with no network
and it looks exactly the same.

The writing rule is just as strict. The first screen must make sense to
someone with no machine-learning background, because that is who reads it.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import mimetypes
from pathlib import Path
from typing import Any, Sequence

from cvassure.core.schemas import Finding

CSS = """
:root{
  --bg:#f7f7f5; --panel:#ffffff; --ink:#1a1a1a; --muted:#5c5c5c; --line:#e0ddd6;
  --green:#1a7f4b; --amber:#b06f00; --red:#c0392b; --blue:#0072B2;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
header.top{margin-bottom:28px}
h1{font-size:30px;margin:0 0 6px}
.sub{color:var(--muted);font-size:15px}
.verdict{border-radius:10px;padding:26px 28px;margin:26px 0;color:#fff}
.verdict.green{background:var(--green)} .verdict.amber{background:var(--amber)}
.verdict.red{background:var(--red)}
.verdict h2{margin:0 0 10px;font-size:34px;letter-spacing:.3px}
.verdict p{margin:4px 0;font-size:18px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:22px 24px;margin:20px 0}
section > h3{margin:0 0 6px;font-size:21px}
section > .lead{color:var(--muted);margin:0 0 16px;font-size:15px}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);
  vertical-align:top}
th{background:#f0efe9;font-weight:600;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#e7e5dd}
th::after{content:"  \\2195";color:#aaa;font-size:11px}
tbody tr:hover{background:#faf9f5}
.scroll{overflow-x:auto}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12.5px;
  font-weight:600;white-space:nowrap}
.pill.accept{background:#e3f3ea;color:var(--green)}
.pill.review{background:#fdf0d9;color:var(--amber)}
.pill.quarantine{background:#fbe4e1;color:var(--red)}
.pill.supported{background:#e3f3ea;color:var(--green)}
.pill.partial{background:#fdf0d9;color:var(--amber)}
.pill.unsupported{background:#fbe4e1;color:var(--red)}
.pill.unavailable,.pill\\:not-measured{background:#ececec;color:#666}
figure{margin:14px 0}
figure img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--muted);font-size:13.5px;margin-top:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.gallery figure{margin:0}
.reason{max-width:640px}
ul.plain{margin:8px 0 0;padding-left:20px}
ul.plain li{margin:5px 0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 18px;font-size:14.5px}
.kv dt{color:var(--muted)}
.kv dd{margin:0}
.note{background:#f4f2ec;border-left:3px solid var(--blue);padding:12px 16px;
  border-radius:0 6px 6px 0;margin:14px 0;font-size:14.5px}
footer{color:var(--muted);font-size:13px;margin-top:34px;text-align:center}
code{background:#f0efe9;padding:1px 5px;border-radius:4px;font-size:13px}
@media print{body{background:#fff} section{break-inside:avoid}}
"""

SORT_JS = """
document.querySelectorAll('table').forEach(function(table){
  var head = table.tHead; if(!head) return;
  Array.prototype.forEach.call(head.rows[0].cells, function(th, idx){
    th.addEventListener('click', function(){
      var body = table.tBodies[0];
      var rows = Array.prototype.slice.call(body.rows);
      var asc = th.dataset.asc !== 'true';
      Array.prototype.forEach.call(head.rows[0].cells, function(c){ c.dataset.asc = ''; });
      th.dataset.asc = asc;
      rows.sort(function(a, b){
        var x = a.cells[idx].innerText.trim(), y = b.cells[idx].innerText.trim();
        var nx = parseFloat(x.replace(/[^0-9.\\-]/g, '')), ny = parseFloat(y.replace(/[^0-9.\\-]/g, ''));
        var both = !isNaN(nx) && !isNaN(ny) && x.match(/[0-9]/) && y.match(/[0-9]/);
        var cmp = both ? nx - ny : x.localeCompare(y);
        return asc ? cmp : -cmp;
      });
      rows.forEach(function(r){ body.appendChild(r); });
    });
  });
});
"""


def esc(text: Any) -> str:
    return _html.escape(str(text), quote=True)


def embed_image(path: str | Path) -> str | None:
    """base64 the image straight into the page — no external files, ever."""
    p = Path(path)
    if not p.exists():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def _pill(value: str) -> str:
    cls = str(value).lower().replace(" ", "-")
    return f'<span class="pill {esc(cls)}">{esc(value)}</span>'


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]], pill_cols=()) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            cells.append(
                f"<td>{_pill(value) if i in pill_cols else esc(value)}</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _section(title: str, lead: str, body: str) -> str:
    lead_html = f'<p class="lead">{esc(lead)}</p>' if lead else ""
    return f"<section><h3>{esc(title)}</h3>{lead_html}{body}</section>"


# --------------------------------------------------------------------------


def render(
    *,
    findings: Sequence[Finding],
    verdict,
    coverage,
    inputs: dict[str, Any],
    limitations: Sequence[str] = (),
    tables: Sequence[Any] = (),
    plots: dict[str, Any] | None = None,
    calibration_plot: str | Path | None = None,
    audit_log_result=None,
    max_gallery: int = 12,
    max_findings: int = 400,
) -> str:
    plots = plots or {}
    parts: list[str] = []

    # -- verdict -------------------------------------------------------
    parts.append(
        f'<div class="verdict {esc(verdict.colour)}"><h2>{esc(verdict.headline)}</h2>'
        + "".join(f"<p>{esc(line)}</p>" for line in verdict.lines)
        + "</div>"
    )

    # -- what was audited ----------------------------------------------
    kv = "".join(
        f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in inputs.items() if v is not None
    )
    parts.append(
        _section(
            "What we looked at",
            "Everything below was produced from these inputs, with no network access "
            "at any point.",
            f'<dl class="kv">{kv}</dl>',
        )
    )

    # -- contributors ---------------------------------------------------
    contributors = [f for f in findings if f.asset_type == "contributor"]
    if contributors:
        unavailable = [f for f in contributors if f.is_unavailable]
        if unavailable:
            body = f'<div class="note">{esc(unavailable[0].reason)}</div>'
        else:
            rows = []
            for f in sorted(contributors, key=lambda f: -f.raw_score):
                e = f.evidence
                ci = e.get("credible_interval_95", [0, 0])
                rows.append(
                    [
                        f.asset_ref,
                        e.get("n_samples", ""),
                        e.get("n_flagged", ""),
                        f"{100 * float(e.get('flagged_rate', 0)):.1f}%",
                        f"{100 * ci[0]:.1f}% – {100 * ci[1]:.1f}%",
                        f.disposition,
                        f.reason,
                    ]
                )
            body = _table(
                ["contributor", "images sent", "images flagged", "flagged rate",
                 "we are 95% sure the true rate is in", "recommendation", "why"],
                rows,
                pill_cols={5},
            )
            body += (
                '<div class="note">The range is there because a contributor who sent '
                "three images and had one flagged is not the same as one who sent "
                "twelve hundred and had two hundred flagged, even though both are "
                "33%. The recommendation reads the range, not the single number.</div>"
            )
        parts.append(
            _section(
                "Who supplied the problem data",
                "Contributor-level assessment — the part you can actually act on.",
                body,
            )
        )

    if "fig4_contributor_risk" in plots:
        src = embed_image(plots["fig4_contributor_risk"])
        if src:
            parts.append(
                _section(
                    "Contributor risk at a glance",
                    "",
                    f'<figure><img src="{src}" alt="contributor risk"></figure>',
                )
            )

    # -- model ----------------------------------------------------------
    models = [f for f in findings if f.asset_type == "model"]
    if models:
        rows = [
            [
                f.detector_id,
                "could not run" if f.is_unavailable else f.disposition,
                f.reason,
            ]
            for f in models
        ]
        body = _table(["check", "result", "what we found"], rows, pill_cols={1})
        caveats = [f.reason for f in models if f.is_unavailable]
        if caveats:
            body += (
                '<div class="note"><strong>Access-tier caveat.</strong> '
                + " ".join(esc(c) for c in caveats)
                + "</div>"
            )
        parts.append(
            _section("Is this the model we were given?",
                     "Model integrity, at the level of access this audit was granted.",
                     body)
        )

    # -- provenance -----------------------------------------------------
    receipts = [f for f in findings if f.asset_type == "receipt"]
    if receipts:
        rows = [[f.asset_ref, f.disposition, f.reason] for f in receipts]
        parts.append(
            _section(
                "Can we trust the record of what the model answered?",
                "Every inference is signed and linked to the one before it, so an "
                "edit anywhere breaks the chain from that point on.",
                _table(["record", "result", "what we found"], rows, pill_cols={1}),
            )
        )

    # -- distribution shift ---------------------------------------------
    shift = [f for f in findings if f.detector_id == "shift"]
    if shift:
        f = shift[0]
        if f.is_unavailable:
            body = f'<div class="note">{esc(f.reason)}</div>'
        else:
            e = f.evidence
            attribution = e.get("attribution", {})
            rows = [
                [row["factor"], f"{100 * row['share']:.0f}%", row["verdict"]]
                for row in attribution.get("factors", [])
            ]
            rows.append(
                ["UNEXPLAINED",
                 f"{100 * attribution.get('unexplained_share', 0):.0f}%",
                 "flagged" if attribution.get("unexplained_share", 0) > 0.25 else "normal"]
            )
            body = f"<p>{esc(f.reason)}</p>"
            body += _table(["what changed", "share of the change", "verdict"], rows)
            body += (
                '<div class="note"><strong>How we tell drift from tampering.</strong> '
                + esc(e.get("classification_rule", ""))
                + "</div>"
            )
        parts.append(
            _section("Has the incoming data drifted?",
                     "Data always changes. The question is whether ordinary causes "
                     "explain the change.",
                     body)
        )

    # -- findings table --------------------------------------------------
    samples = [
        f for f in findings
        if f.asset_type == "sample" and not f.is_unavailable and f.disposition != "accept"
    ]
    samples.sort(key=lambda f: -f.score)
    if samples:
        rows = [
            [f.asset_ref, f.detector_id, f"{f.score:.2f}", f.severity, f.disposition, f.reason]
            for f in samples[:max_findings]
        ]
        body = _table(
            ["image", "check", "score", "severity", "recommendation", "why"],
            rows,
            pill_cols={4},
        )
        if len(samples) > max_findings:
            body += (
                f'<p class="lead">Showing the {max_findings} most suspicious of '
                f"{len(samples)}. The full list is in <code>findings.jsonl</code>.</p>"
            )
        parts.append(
            _section(
                f"Individual images we would not accept ({len(samples)})",
                "Click any column heading to sort.",
                body,
            )
        )

    # -- gallery ---------------------------------------------------------
    gallery = []
    for f in samples:
        for art in f.artefacts:
            src = embed_image(art)
            if src:
                gallery.append(
                    f'<figure><img src="{src}" alt="{esc(f.asset_ref)}">'
                    f"<figcaption>{esc(f.asset_ref)} — {esc(f.detector_id)}</figcaption>"
                    f"</figure>"
                )
        if len(gallery) >= max_gallery:
            break
    for f in findings:
        if f.asset_type == "model" and f.artefacts and len(gallery) < max_gallery + 2:
            src = embed_image(f.artefacts[0])
            if src:
                gallery.append(
                    f'<figure><img src="{src}" alt="reconstructed trigger">'
                    f"<figcaption>Reconstructed trigger — {esc(f.detector_id)}"
                    f"</figcaption></figure>"
                )
    if gallery:
        parts.append(
            _section(
                "What the system was looking at",
                "The bright areas are where the check found something out of place.",
                f'<div class="gallery">{"".join(gallery)}</div>',
            )
        )

    # -- calibration ------------------------------------------------------
    cal = calibration_plot or plots.get("fig2_reliability")
    if cal:
        src = embed_image(cal)
        if src:
            parts.append(
                _section(
                    "Do our confidence numbers mean anything?",
                    "If we say we are 90% sure a hundred times, about ninety of those "
                    "should really be poisoned. This chart is how you check that.",
                    f'<figure><img src="{src}" alt="calibration"></figure>',
                )
            )

    # -- coverage ---------------------------------------------------------
    cov_table = coverage.table()
    rows = [[r[c] for c in cov_table.columns] for r in cov_table.rows]
    status_col = cov_table.columns.index("status")
    body = _table(cov_table.columns, rows, pill_cols={status_col})
    body += "<ul class='plain'>" + "".join(
        f"<li>{esc(line)}</li>" for line in coverage.statement()
    ) + "</ul>"
    parts.append(
        _section(
            "What this system can and cannot detect",
            "Including the things it cannot. A clean result on an unsupported row is "
            "not evidence of anything.",
            body,
        )
    )

    # -- results tables ---------------------------------------------------
    for t in tables:
        rows = [[r[c] for c in t.columns] for r in t.rows]
        if not rows:
            continue
        parts.append(
            _section(
                t.title,
                " ".join(t.notes),
                _table(t.columns, rows),
            )
        )

    # -- limitations ------------------------------------------------------
    notes = list(limitations)
    if audit_log_result is not None:
        notes.insert(0, audit_log_result.render().replace("\n", " "))
    notes.append(
        "The scores come from a fixed set of checks. An attacker who knows exactly "
        "which checks these are can design something that avoids all of them; a clean "
        "report means we found nothing, not that there is nothing."
    )
    parts.append(
        _section(
            "Limitations and assumptions",
            "",
            "<ul class='plain'>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>",
        )
    )

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%d %B %Y at %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data and Model Assurance Report</title>
<style>{CSS}</style></head><body><div class="wrap">
<header class="top">
  <h1>Data and Model Assurance Report</h1>
  <p class="sub">Generated {esc(generated)} — produced entirely offline by cvassure.</p>
</header>
{"".join(parts)}
<footer>cvassure — integrity assurance for computer-vision pipelines.
No part of this audit contacted the network.</footer>
</div><script>{SORT_JS}</script></body></html>"""


def write(path: str | Path, **kwargs: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(**kwargs), encoding="utf-8")
    return p
