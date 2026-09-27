// cvassure web console — vanilla JS, no build step.
// Talks to the Flask API in cvassure/webapp.py, which calls the exact same
// detector code the CLI does. Nothing here computes a finding on its own.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// --------------------------------------------------------------------------
// module / tab map
// --------------------------------------------------------------------------

const MODULES = [
  { id: "overview", label: "Overview", icon: "◎", eyebrow: "Start here",
    tabs: [{ id: "home", label: "Home" }] },
  { id: "data", label: "Data Integrity", icon: "▤", eyebrow: "PS clause 2.2.1",
    tabs: [
      { id: "upload", label: "Upload & run" },
      { id: "findings", label: "Findings" },
      { id: "contributors", label: "Contributors" },
      { id: "coverage", label: "Coverage" },
    ] },
  { id: "model", label: "Model Integrity", icon: "⌘", eyebrow: "PS clause 2.2.2",
    tabs: [
      { id: "upload", label: "Upload & run" },
      { id: "findings", label: "Findings" },
      { id: "enroll", label: "Enroll a model" },
    ] },
  { id: "receipts", label: "Inference & Receipts", icon: "✓", eyebrow: "PS clause 2.2.3",
    tabs: [
      { id: "verify", label: "Upload & verify" },
      { id: "chain", label: "Receipt chain" },
    ] },
  { id: "reports", label: "Reports", icon: "☷", eyebrow: "Every run, in one place",
    tabs: [{ id: "runs", label: "All runs" }] },
];

const moduleById = Object.fromEntries(MODULES.map((m) => [m.id, m]));

const ATTACK_LABELS = {
  badnets_patch: "A marker pasted onto the image",
  blended_trigger: "A faint pattern laid over the whole image",
  label_flip: "The picture is fine, the label is wrong",
  systematic_mislabel: "One contributor labels everything the same wrong way",
  near_duplicate_flood: "The same photograph submitted many times",
  ood_insertion: "Images from a completely different source",
  model_substitute: "The model file was swapped",
  model_perturb: "The model's weights were edited",
  model_backdoor: "A hidden trigger was trained into the model",
  receipt_alter: "An inference record was edited afterwards",
  receipt_replay: "An old inference record was submitted again",
  receipt_delete: "An inference record was removed",
  receipt_reorder: "The inference records were shuffled",
  distribution_shift: "The incoming data has drifted from normal",
  clean: "Nothing wrong",
};

// --------------------------------------------------------------------------
// state
// --------------------------------------------------------------------------

const state = {
  module: "overview",
  tab: "home",
  current: { data: null, model: null, receipts: null },
  history: { data: [], model: [], receipts: [], reports: [] },
  busy: false,
};

// --------------------------------------------------------------------------
// tiny fetch helpers
// --------------------------------------------------------------------------

async function apiGet(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

async function apiPostForm(path, formData) {
  const res = await fetch(path, { method: "POST", body: formData });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

function toast(message, kind = "good") {
  const t = $("#toast");
  t.textContent = message;
  t.className = `toast show ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 4200);
}

// --------------------------------------------------------------------------
// formatting helpers
// --------------------------------------------------------------------------

const pct = (x) => `${Math.round(Number(x) * 100)}%`;
const dispBadge = (d) => `<span class="badge badge-${esc(d)}">${esc(d)}</span>`;
const sevBadge = (s) => `<span class="sev sev-${esc(s)}">${esc(s)}</span>`;
function verdictBanner(run, module) {
  const v = run && run.verdict;
  if (!v) return "";
  const meta = [];
  if (run.assessment_id) meta.push(`Assessment <b>${esc(run.assessment_id)}</b>`);
  if (run.risk) meta.push(`Overall risk <b>${esc(run.risk)}</b>`);
  if (run.confidence != null) meta.push(`Overall confidence <b>${pct(run.confidence)}</b>`);
  const metaLine = meta.length ? `<div class="verdict-meta">${meta.join(" &nbsp;·&nbsp; ")}</div>` : "";
  const download = module
    ? `<a class="text-button small verdict-download" href="/api/reports/${esc(module)}/${esc(run.id)}/assurance_record.json" target="_blank" rel="noopener">Download assurance record (JSON) ⤓</a>`
    : "";
  return `<div class="verdict-banner verdict-${esc(v.colour)}">
        <strong>${esc(v.headline)}</strong>
        ${(v.lines || []).map((l) => `<p>${esc(l)}</p>`).join("")}
        ${metaLine}
        ${download}
      </div>`;
}

function attackLabel(a) {
  return ATTACK_LABELS[a] || a;
}

// --------------------------------------------------------------------------
// router
// --------------------------------------------------------------------------

function parseHash() {
  const raw = (location.hash || "#/overview/home").slice(2);
  const [m, t] = raw.split("/");
  const mod = moduleById[m] ? m : "overview";
  const tabs = moduleById[mod].tabs.map((x) => x.id);
  const tab = tabs.includes(t) ? t : tabs[0];
  return { mod, tab };
}

function navigate(mod, tab) {
  location.hash = `#/${mod}/${tab}`;
}

window.addEventListener("hashchange", render);

// --------------------------------------------------------------------------
// sidebar + topbar chrome
// --------------------------------------------------------------------------

function renderSidebar() {
  $("#sidebarNav").innerHTML = MODULES.map(
    (m) => `<button class="sidebar-link ${m.id === state.module ? "active" : ""}" data-mod="${m.id}">
      <span class="sidebar-icon">${m.icon}</span><span class="sidebar-label">${esc(m.label)}</span>
    </button>`
  ).join("");
  $$(".sidebar-link", $("#sidebarNav")).forEach((btn) =>
    btn.addEventListener("click", () => navigate(btn.dataset.mod, moduleById[btn.dataset.mod].tabs[0].id))
  );
}

function renderTopbar() {
  const mod = moduleById[state.module];
  $("#moduleEyebrow").textContent = mod.eyebrow;
  $("#moduleTitle").textContent = mod.label;
  $("#moduleTabs").innerHTML = mod.tabs
    .map(
      (t) =>
        `<button class="module-tab ${t.id === state.tab ? "active" : ""}" data-tab="${t.id}">${esc(t.label)}</button>`
    )
    .join("");
  $$(".module-tab", $("#moduleTabs")).forEach((btn) =>
    btn.addEventListener("click", () => navigate(state.module, btn.dataset.tab))
  );
}

function initSidebarCollapse() {
  const btn = $("#sidebarToggle");
  let collapsed = false;
  try {
    collapsed = localStorage.getItem("cvassure.sidebar.collapsed") === "1";
  } catch (_) {}
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  btn.textContent = collapsed ? "»" : "«";
  btn.addEventListener("click", () => {
    collapsed = !document.body.classList.contains("sidebar-collapsed");
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    btn.textContent = collapsed ? "»" : "«";
    try {
      localStorage.setItem("cvassure.sidebar.collapsed", collapsed ? "1" : "0");
    } catch (_) {}
  });
}

// --------------------------------------------------------------------------
// upload widgets
// --------------------------------------------------------------------------

function dropzoneHtml({ name, label, hint, accept, required }) {
  return `<label class="dropzone" data-field="${name}">
    <input type="file" name="${name}" accept="${accept || ""}" ${required ? "required" : ""} />
    <span class="dropzone-icon">⤓</span>
    <span class="dropzone-label">${esc(label)}</span>
    <span class="dropzone-hint" data-hint>${esc(hint || "")}</span>
  </label>`;
}

function wireDropzones(root) {
  $$(".dropzone", root).forEach((zone) => {
    const input = $("input", zone);
    const hint = $("[data-hint]", zone);
    const base = hint.textContent;
    const show = () => {
      if (input.files && input.files.length) {
        zone.classList.add("has-file");
        hint.textContent = input.files[0].name;
      } else {
        zone.classList.remove("has-file");
        hint.textContent = base;
      }
    };
    input.addEventListener("change", show);
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        show();
      }
    });
  });
}

function setBusy(form, busy, label) {
  state.busy = busy;
  const btn = $("button[type=submit]", form);
  if (!btn) return;
  btn.disabled = busy;
  btn.textContent = busy ? label || "Running…" : btn.dataset.idle;
}

// --------------------------------------------------------------------------
// generic findings table
// --------------------------------------------------------------------------

function findingsTable(findings, { assetTypes } = {}) {
  const rows = assetTypes ? findings.filter((f) => assetTypes.includes(f.asset_type)) : findings;
  if (!rows.length) {
    return `<p class="empty-note">No findings of this kind in this run.</p>`;
  }
  const detectors = Array.from(new Set(rows.map((f) => f.detector_id))).sort();
  const html = `
    <div class="findings-toolbar">
      <label>Detector
        <select data-filter="detector"><option value="">all</option>${detectors
          .map((d) => `<option value="${esc(d)}">${esc(d)}</option>`)
          .join("")}</select>
      </label>
      <label>Disposition
        <select data-filter="disposition">
          <option value="">all</option>
          <option value="quarantine">quarantine</option>
          <option value="review">review</option>
          <option value="accept">accept</option>
        </select>
      </label>
      <label>Search <input type="search" data-filter="search" placeholder="asset id…" /></label>
    </div>
    <div class="findings-scroll">
      <table class="findings-table">
        <thead><tr>
          <th>Asset</th><th>Attack class</th><th>Check</th><th>Severity</th>
          <th>Disposition</th><th>Score</th><th>Reason</th>
        </tr></thead>
        <tbody>
          ${rows
            .map(
              (f) => `<tr data-row data-detector="${esc(f.detector_id)}" data-disposition="${esc(f.disposition)}" data-asset="${esc(f.asset_ref)}">
              <td class="mono">${esc(f.asset_ref)}</td>
              <td>${esc(attackLabel(f.attack_class))}</td>
              <td class="mono">${esc(f.detector_id)}</td>
              <td>${f.is_unavailable ? '<span class="sev sev-unavailable">n/a</span>' : sevBadge(f.severity)}</td>
              <td>${f.is_unavailable ? "—" : dispBadge(f.disposition)}</td>
              <td class="mono">${f.is_unavailable ? "—" : (f.score ?? f.raw_score).toFixed(2)}</td>
              <td class="reason">${esc(f.is_unavailable ? f.reason : f.reason)}</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>
    </div>`;
  return html;
}

function wireFindingsFilters(root) {
  const apply = () => {
    const detector = $('[data-filter="detector"]', root)?.value || "";
    const disposition = $('[data-filter="disposition"]', root)?.value || "";
    const search = ($('[data-filter="search"]', root)?.value || "").toLowerCase();
    $$("tr[data-row]", root).forEach((row) => {
      const okD = !detector || row.dataset.detector === detector;
      const okP = !disposition || row.dataset.disposition === disposition;
      const okS = !search || row.dataset.asset.toLowerCase().includes(search);
      row.style.display = okD && okP && okS ? "" : "none";
    });
  };
  $$("[data-filter]", root).forEach((el) => el.addEventListener("input", apply));
}

// --------------------------------------------------------------------------
// history side panel (recent runs for a module)
// --------------------------------------------------------------------------

function historyPanel(module, runs) {
  if (!runs.length) return `<p class="empty-note">No runs yet on this machine.</p>`;
  return `<div class="run-history">
    ${runs
      .slice(0, 8)
      .map(
        (r) => `<button class="run-row" data-run="${esc(r.id)}">
          <span class="mono">${esc(r.id)}</span>
          ${verdictChip(r.verdict)}
        </button>`
      )
      .join("")}
  </div>`;
}

function verdictChip(v) {
  if (!v) return "";
  return `<span class="chip chip-${esc(v.colour)}">${esc(v.headline)}</span>`;
}

async function wireHistory(root, module, onLoad) {
  $$(".run-row", root).forEach((btn) =>
    btn.addEventListener("click", async () => {
      try {
        const detail = await apiGet(`/api/${module}/runs/${btn.dataset.run}`);
        state.current[module] = detail;
        onLoad(detail);
      } catch (err) {
        toast(err.message, "bad");
      }
    })
  );
}

// --------------------------------------------------------------------------
// OVERVIEW
// --------------------------------------------------------------------------

async function renderOverview(root) {
  root.innerHTML = `
    <section class="overview-hero">
      <p class="eyebrow">Computer-vision integrity assurance</p>
      <h2>Know what made it in.</h2>
      <p class="lede">Upload a contributor's dataset or a vendor's model and get the same checks
        <code>cvassure audit</code> runs from the command line — nothing here leaves this machine.</p>
      <div class="overview-cards">
        <button class="overview-card" data-go="data/upload">
          <span class="overview-card-icon">▤</span><strong>Data Integrity</strong>
          <span>Add a contributor's dataset and screen it for wrong labels, flooded duplicates,
            foreign images and pasted-in triggers.</span>
        </button>
        <button class="overview-card" data-go="model/upload">
          <span class="overview-card-icon">⌘</span><strong>Model Integrity</strong>
          <span>Add a vendor's model (.onnx / .pt) and check it for substitution, edited weights
            and hidden backdoors.</span>
        </button>
        <button class="overview-card" data-go="receipts/verify">
          <span class="overview-card-icon">✓</span><strong>Inference &amp; Receipts</strong>
          <span>Verify a signed inference log's hash chain end to end.</span>
        </button>
      </div>
    </section>
    <section class="overview-recent">
      <h3>Recent activity</h3>
      <div id="overviewRecent"><p class="empty-note">Loading…</p></div>
    </section>`;
  $$(".overview-card", root).forEach((btn) =>
    btn.addEventListener("click", () => {
      const [m, t] = btn.dataset.go.split("/");
      navigate(m, t);
    })
  );
  try {
    const runs = await apiGet("/api/reports");
    $("#overviewRecent").innerHTML = runs.length
      ? runsTable(runs.slice(0, 6))
      : `<p class="empty-note">Nothing has been run on this machine yet.</p>`;
    wireRunsTable($("#overviewRecent"));
  } catch (err) {
    $("#overviewRecent").innerHTML = `<p class="empty-note">${esc(err.message)}</p>`;
  }
}

// --------------------------------------------------------------------------
// DATA INTEGRITY
// --------------------------------------------------------------------------

async function renderDataUpload(root) {
  root.innerHTML = `
    <div class="workspace">
      <form class="panel upload-form" id="dataForm">
        <h3>Upload a dataset</h3>
        <p class="panel-copy">A .zip of an ImageFolder layout (one subfolder per class), a COCO
          instances JSON with its images, or a YOLO <code>data.yaml</code> + <code>labels/</code>.
          A <code>contributors.json</code> at the root is picked up automatically.</p>
        ${dropzoneHtml({ name: "dataset", label: "Dataset .zip", hint: "drop a .zip, or click to browse", accept: ".zip", required: true })}
        ${dropzoneHtml({ name: "contributors", label: "contributors.json (optional)", hint: "maps sample id → contributor", accept: ".json" })}
        <label class="text-field">Contributor-from-path regex (optional)
          <input type="text" name="contributor_from_path" placeholder="e.g. contributor_(\\w+)" />
        </label>
        <button type="submit" class="primary-button" data-idle="Run data integrity audit">Run data integrity audit</button>
      </form>
      <aside class="panel">
        <h3>Recent runs</h3>
        <div id="dataHistory">${historyPanel("data", state.history.data)}</div>
      </aside>
    </div>`;
  wireDropzones(root);
  wireHistory($("#dataHistory"), "data", (detail) => {
    state.current.data = detail;
    navigate("data", "findings");
  });

  $("#dataForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    if (!fd.get("dataset") || !fd.get("dataset").name) {
      toast("Attach a dataset .zip first.", "bad");
      return;
    }
    setBusy(form, true, "Running the data checks…");
    try {
      const result = await apiPostForm("/api/data/runs", fd);
      state.current.data = result;
      toast(`Audit complete — ${result.verdict.headline}`, result.verdict.colour === "red" ? "bad" : "good");
      await refreshHistory("data");
      navigate("data", "findings");
    } catch (err) {
      toast(err.message, "bad");
    } finally {
      setBusy(form, false);
    }
  });
}

function renderDataFindings(root) {
  const run = state.current.data;
  if (!run) {
    root.innerHTML = emptyState("data", "upload", "Upload a dataset to see findings here.");
    return;
  }
  const s = run.summary;
  root.innerHTML = `
    ${verdictBanner(run, "data")}
    <div class="stat-row">
      <div><span>IMAGES</span><strong>${s.n_samples}</strong></div>
      <div><span>CLASSES</span><strong>${s.classes.length}</strong></div>
      <div><span>CONTRIBUTORS</span><strong>${s.n_contributors}</strong></div>
      <div><span>FLAGGED</span><strong>${run.n_flagged}</strong></div>
    </div>
    ${s.limitations.length ? `<ul class="limitations">${s.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : ""}
    ${categoryRiskTable(run.category_risk)}
    <div class="panel">${findingsTable(run.findings, { assetTypes: ["sample"] })}</div>`;
  wireFindingsFilters(root);
}

function renderDataContributors(root) {
  const run = state.current.data;
  if (!run) {
    root.innerHTML = emptyState("data", "upload", "Upload a dataset to see per-contributor risk here.");
    return;
  }
  const cards = run.findings.filter(
    (f) => f.asset_type === "contributor" && !f.is_unavailable && !String(f.asset_ref).startsWith("batch:")
  );
  const unavailable = run.findings.filter((f) => f.asset_type === "contributor" && f.is_unavailable);
  if (!cards.length) {
    root.innerHTML = `<div class="panel">${
      unavailable.length
        ? unavailable.map((f) => `<p class="empty-note">${esc(f.reason)}</p>`).join("")
        : `<p class="empty-note">No contributor metadata in this dataset.</p>`
    }</div>`;
    return;
  }
  root.innerHTML = `<div class="contributor-grid">
    ${cards
      .sort((a, b) => b.raw_score - a.raw_score)
      .map(
        (f) => `<article class="contributor-card contributor-${esc(f.disposition)}">
        <div class="contributor-card-head"><strong class="mono">${esc(f.asset_ref)}</strong>${dispBadge(f.disposition)}</div>
        <div class="contributor-stat"><b>${f.evidence.n_flagged}</b> / ${f.evidence.n_samples} flagged
          <span>(${pct(f.evidence.flagged_rate)})</span></div>
        <div class="inspector-meter"><i style="width:${Math.min(100, f.evidence.flagged_rate * 300)}%"></i></div>
        <p>${esc(f.reason)}</p>
        <small>credible interval [${pct(f.evidence.credible_interval_95[0])}, ${pct(f.evidence.credible_interval_95[1])}]</small>
      </article>`
      )
      .join("")}
  </div>`;
}

function renderDataCoverage(root) {
  const run = state.current.data;
  if (!run) {
    root.innerHTML = emptyState("data", "upload", "Upload a dataset to see coverage here.");
    return;
  }
  root.innerHTML = coverageTable(run.coverage);
}

// --------------------------------------------------------------------------
// MODEL INTEGRITY
// --------------------------------------------------------------------------

async function renderModelUpload(root) {
  root.innerHTML = `
    <div class="workspace">
      <form class="panel upload-form" id="modelForm">
        <h3>Upload a model</h3>
        <p class="panel-copy">ONNX (<code>.onnx</code>) or TorchScript (<code>.pt</code> / <code>.pth</code>).
          A dataset is optional but unlocks two more checks (spectral signature, trigger reconstruction).</p>
        ${dropzoneHtml({ name: "model", label: "Model file", hint: "drop a .onnx / .pt / .pth, or click to browse", accept: ".onnx,.pt,.pth,.ts,.torchscript", required: true })}
        <fieldset class="tier-field">
          <legend>Access tier declared for this model</legend>
          <label><input type="radio" name="access_tier" value="0" checked /> 0 — answers only</label>
          <label><input type="radio" name="access_tier" value="1" /> 1 — weights readable</label>
          <label><input type="radio" name="access_tier" value="2" /> 2 — internals readable</label>
        </fieldset>
        ${dropzoneHtml({ name: "dataset", label: "Dataset .zip (optional)", hint: "unlocks activation-based checks", accept: ".zip" })}
        ${dropzoneHtml({ name: "enrolled_fingerprint", label: "Enrolled fingerprint (optional)", hint: "from the Enroll tab — lets us prove nothing changed", accept: ".json" })}
        <button type="submit" class="primary-button" data-idle="Run model integrity audit">Run model integrity audit</button>
      </form>
      <aside class="panel">
        <h3>Recent runs</h3>
        <div id="modelHistory">${historyPanel("model", state.history.model)}</div>
      </aside>
    </div>`;
  wireDropzones(root);
  wireHistory($("#modelHistory"), "model", (detail) => {
    state.current.model = detail;
    navigate("model", "findings");
  });

  $("#modelForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    if (!fd.get("model") || !fd.get("model").name) {
      toast("Attach a model file first.", "bad");
      return;
    }
    setBusy(form, true, "Running the model checks…");
    try {
      const result = await apiPostForm("/api/model/runs", fd);
      state.current.model = result;
      toast(`Audit complete — ${result.verdict.headline}`, result.verdict.colour === "red" ? "bad" : "good");
      await refreshHistory("model");
      navigate("model", "findings");
    } catch (err) {
      toast(err.message, "bad");
    } finally {
      setBusy(form, false);
    }
  });
}

function renderModelFindings(root) {
  const run = state.current.model;
  if (!run) {
    root.innerHTML = emptyState("model", "upload", "Upload a model to see findings here.");
    return;
  }
  root.innerHTML = `
    ${verdictBanner(run, "model")}
    <div class="stat-row">
      <div><span>MODEL KIND</span><strong>${esc(run.model.kind)}</strong></div>
      <div><span>ACCESS TIER</span><strong>${run.access_tier}</strong></div>
      <div><span>FINGERPRINT</span><strong>${run.enrolled ? "compared" : "not enrolled"}</strong></div>
      <div><span>FINDINGS</span><strong>${run.n_findings}</strong></div>
    </div>
    <div class="panel">${findingsTable(run.findings)}</div>`;
  wireFindingsFilters(root);
}

function renderModelEnroll(root) {
  root.innerHTML = `
    <div class="workspace">
      <form class="panel upload-form" id="enrollForm">
        <h3>Enroll a model</h3>
        <p class="panel-copy">Records this model's fixed-probe fingerprint (and, at tier 1+, a
          per-layer weight digest) as a JSON file. Keep it, and hand it back in on the
          <em>Upload &amp; run</em> tab next time this model is audited — that is what proves the
          file has not been swapped or edited.</p>
        ${dropzoneHtml({ name: "model", label: "Model file", hint: "drop a .onnx / .pt / .pth, or click to browse", accept: ".onnx,.pt,.pth,.ts,.torchscript", required: true })}
        <fieldset class="tier-field">
          <legend>Access tier available for enrollment</legend>
          <label><input type="radio" name="access_tier" value="2" checked /> 2 — internals readable</label>
          <label><input type="radio" name="access_tier" value="1" /> 1 — weights readable</label>
          <label><input type="radio" name="access_tier" value="0" /> 0 — answers only</label>
        </fieldset>
        <button type="submit" class="primary-button" data-idle="Enroll &amp; download fingerprint">Enroll &amp; download fingerprint</button>
      </form>
      <aside class="panel">
        <h3>Why this matters</h3>
        <p class="panel-copy">Without an enrolled fingerprint, the identity checks can only say
          "we have nothing to compare against." With one, a future audit of the same model can
          prove — not guess — whether it is still the model that was accepted.</p>
      </aside>
    </div>`;
  wireDropzones(root);
  $("#enrollForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    if (!fd.get("model") || !fd.get("model").name) {
      toast("Attach a model file first.", "bad");
      return;
    }
    setBusy(form, true, "Enrolling…");
    try {
      const res = await fetch("/api/model/enroll", { method: "POST", body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `request failed (${res.status})`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "enrolled_fingerprint.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast("Fingerprint downloaded.", "good");
    } catch (err) {
      toast(err.message, "bad");
    } finally {
      setBusy(form, false);
    }
  });
}

// --------------------------------------------------------------------------
// RECEIPTS
// --------------------------------------------------------------------------

async function renderReceiptsVerify(root) {
  root.innerHTML = `
    <div class="workspace">
      <form class="panel upload-form" id="receiptsForm">
        <h3>Verify an inference log</h3>
        <p class="panel-copy">A signed, hash-chained receipts <code>.jsonl</code> (from
          <code>cvassure provenance sign</code>) and the public key it was signed with. If you
          omit the key, this machine's own <code>keys/pub.pem</code> is used when present.</p>
        ${dropzoneHtml({ name: "receipts", label: "Receipts .jsonl", hint: "drop the signed log, or click to browse", accept: ".jsonl,.json", required: true })}
        ${dropzoneHtml({ name: "pubkey", label: "Public key (optional)", hint: "falls back to this machine's keys/pub.pem", accept: ".pem" })}
        <button type="submit" class="primary-button" data-idle="Verify chain">Verify chain</button>
      </form>
      <aside class="panel">
        <h3>Recent runs</h3>
        <div id="receiptsHistory">${historyPanel("receipts", state.history.receipts)}</div>
      </aside>
    </div>`;
  wireDropzones(root);
  wireHistory($("#receiptsHistory"), "receipts", (detail) => {
    state.current.receipts = detail;
    navigate("receipts", "chain");
  });

  $("#receiptsForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    if (!fd.get("receipts") || !fd.get("receipts").name) {
      toast("Attach a receipts .jsonl file first.", "bad");
      return;
    }
    setBusy(form, true, "Verifying…");
    try {
      const result = await apiPostForm("/api/receipts/runs", fd);
      state.current.receipts = result;
      toast(`Verification complete — ${result.verdict.headline}`, result.verdict.colour === "red" ? "bad" : "good");
      await refreshHistory("receipts");
      navigate("receipts", "chain");
    } catch (err) {
      toast(err.message, "bad");
    } finally {
      setBusy(form, false);
    }
  });
}

function renderReceiptsChain(root) {
  const run = state.current.receipts;
  if (!run) {
    root.innerHTML = emptyState("receipts", "verify", "Verify a receipts log to see its chain here.");
    return;
  }
  const r = run.result;
  const badSeqs = new Set(r.failures.map((f) => f.seq).filter((s) => s != null));
  const chips = [];
  for (let i = 1; i <= Math.min(r.total, 60); i++) {
    chips.push(`<span class="chain-link ${badSeqs.has(i) ? "chain-bad" : "chain-good"}" title="record ${i}">${i}</span>`);
  }
  root.innerHTML = `
    ${verdictBanner(run, "receipts")}
    <div class="stat-row">
      <div><span>RECORDS</span><strong>${r.total}</strong></div>
      <div><span>VERIFIED</span><strong>${r.n_verified}</strong></div>
      <div><span>FAILURES</span><strong>${r.failures.length}</strong></div>
    </div>
    <div class="panel">
      <h3>Chain (${Math.min(r.total, 60)} of ${r.total} shown)</h3>
      <div class="chain-strip">${chips.join("")}</div>
      ${
        r.failures.length
          ? `<table class="findings-table"><thead><tr><th>Record</th><th>Problem</th><th>Detail</th></tr></thead><tbody>
             ${r.failures
               .map(
                 (f) => `<tr><td class="mono">${f.seq ?? f.index}</td><td class="mono">${esc(f.code)}</td>
                   <td>${esc(f.explanation)} <span class="reason-detail">${esc(f.detail)}</span></td></tr>`
               )
               .join("")}
             </tbody></table>`
          : `<p class="empty-note">Every record verified. The chain is unbroken.</p>`
      }
    </div>`;
}

// --------------------------------------------------------------------------
// REPORTS
// --------------------------------------------------------------------------

function runsTable(runs) {
  return `<table class="findings-table runs-table"><thead><tr>
    <th>Module</th><th>Assessment</th><th>Created</th><th>Verdict</th><th>Risk</th><th></th>
  </tr></thead><tbody>
    ${runs
      .map(
        (r) => `<tr>
        <td><span class="module-chip module-chip-${esc(r.module)}">${esc(moduleById[r.module]?.label || r.module)}</span></td>
        <td class="mono">${esc(r.assessment_id || r.id)}</td>
        <td class="mono">${esc(r.created)}</td>
        <td>${verdictChip(r.verdict)}</td>
        <td>${r.risk ? `<span class="chip chip-${riskColour(r.risk)}">${esc(r.risk)}</span>` : "—"}</td>
        <td class="run-actions">
          <button class="text-button small" data-load="${esc(r.module)}:${esc(r.id)}">Load</button>
          <a class="text-button small" href="/api/reports/${esc(r.module)}/${esc(r.id)}/report.html" target="_blank" rel="noopener">Open report ↗</a>
          <a class="text-button small" href="/api/reports/${esc(r.module)}/${esc(r.id)}/assurance_record.json" target="_blank" rel="noopener">JSON ↗</a>
        </td>
      </tr>`
      )
      .join("")}
  </tbody></table>`;
}

function riskColour(risk) {
  return risk === "HIGH" ? "red" : risk === "MEDIUM" ? "amber" : "green";
}

function wireRunsTable(root) {
  $$("[data-load]", root).forEach((btn) =>
    btn.addEventListener("click", async () => {
      const [module, id] = btn.dataset.load.split(":");
      try {
        const detail = await apiGet(`/api/${module}/runs/${id}`);
        state.current[module] = detail;
        const firstTab = module === "receipts" ? "chain" : module === "data" ? "findings" : "findings";
        navigate(module, firstTab);
      } catch (err) {
        toast(err.message, "bad");
      }
    })
  );
}

async function renderReportsRuns(root) {
  root.innerHTML = `<div class="panel"><h3>Every run on this machine</h3><div id="reportsList"><p class="empty-note">Loading…</p></div></div>`;
  try {
    const runs = await apiGet("/api/reports");
    state.history.reports = runs;
    $("#reportsList").innerHTML = runs.length
      ? runsTable(runs)
      : `<p class="empty-note">Nothing has been run yet. Try Data Integrity or Model Integrity.</p>`;
    wireRunsTable($("#reportsList"));
  } catch (err) {
    $("#reportsList").innerHTML = `<p class="empty-note">${esc(err.message)}</p>`;
  }
}

// --------------------------------------------------------------------------
// shared bits
// --------------------------------------------------------------------------

function emptyState(mod, tab, message) {
  return `<div class="panel empty-panel">
    <p class="empty-note">${esc(message)}</p>
    <button class="primary-button" data-go="${mod}/${tab}">Go to upload</button>
  </div>`;
}

function coverageTable(rows) {
  return `<div class="panel"><table class="findings-table"><thead><tr>
    <th>What an attacker did</th><th>Access tier</th><th>Status</th><th>Check</th><th>Measured</th><th>Note</th>
  </tr></thead><tbody>
    ${rows
      .map(
        (r) => `<tr>
        <td>${esc(attackLabel(r.attack_class))}</td>
        <td class="mono">${r.access_tier}</td>
        <td><span class="chip chip-${r.status === "supported" ? "green" : r.status === "partial" ? "amber" : "red"}">${esc(r.status)}</span></td>
        <td class="mono">${esc(r.detector || "—")}</td>
        <td>${esc(r.measured || "—")}</td>
        <td>${esc(r.note)}</td>
      </tr>`
      )
      .join("")}
  </tbody></table></div>`;
}

function categoryRiskTable(rows) {
  if (!rows || !rows.length) return "";
  return `<div class="panel category-risk">
    <h3>Training-data risk by category</h3>
    <table class="findings-table"><thead><tr>
      <th>Category</th><th>Images checked</th><th>Images flagged</th><th>Level</th><th>Status</th>
    </tr></thead><tbody>
      ${rows
        .map(
          (r) => `<tr>
          <td>${esc(r.category)}</td>
          <td class="mono">${r.n_checked}</td>
          <td class="mono">${r.n_flagged}</td>
          <td>${sevBadge(r.level)}</td>
          <td>${dispBadge(r.status)}</td>
        </tr>`
        )
        .join("")}
    </tbody></table>
  </div>`;
}

async function refreshHistory(module) {
  try {
    state.history[module] = await apiGet(`/api/${module}/runs`);
  } catch (_) {
    state.history[module] = [];
  }
}

// --------------------------------------------------------------------------
// main render
// --------------------------------------------------------------------------

const RENDERERS = {
  "overview/home": renderOverview,
  "data/upload": renderDataUpload,
  "data/findings": renderDataFindings,
  "data/contributors": renderDataContributors,
  "data/coverage": renderDataCoverage,
  "model/upload": renderModelUpload,
  "model/findings": renderModelFindings,
  "model/enroll": renderModelEnroll,
  "receipts/verify": renderReceiptsVerify,
  "receipts/chain": renderReceiptsChain,
  "reports/runs": renderReportsRuns,
};

async function render() {
  const { mod, tab } = parseHash();
  state.module = mod;
  state.tab = tab;
  renderSidebar();
  renderTopbar();
  const content = $("#moduleContent");
  const fn = RENDERERS[`${mod}/${tab}`];
  if (fn) {
    await fn(content);
  } else {
    content.innerHTML = `<p class="empty-note">Nothing here yet.</p>`;
  }
  $$("[data-go]", content).forEach((btn) =>
    btn.addEventListener("click", () => {
      const [m, t] = btn.dataset.go.split("/");
      navigate(m, t);
    })
  );
}

async function boot() {
  initSidebarCollapse();
  await Promise.all([refreshHistory("data"), refreshHistory("model"), refreshHistory("receipts")]);
  if (!location.hash) location.hash = "#/overview/home";
  render();
}

boot();
