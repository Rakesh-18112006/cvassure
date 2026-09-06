const scenarios = {
  clean: {
    pill: "STEP 01 / CLEAN INTAKE",
    title: "Nothing wrong with it.",
    copy: "The baseline is clean. This matters: cvassure does not turn uncertainty into a finding just to look busy.",
    finding:
      "All 5 contributors accepted. Model fingerprint matches enrollment.",
    check: "A clean reference dataset and enrolled model.",
    why: "A clean baseline keeps later findings honest.",
    action: "Accept the intake and continue.",
    verdict: "BASELINE ACCEPTED",
    image: "../data/synth10/aircraft/aircraft_00003.png",
    caption: "clean reference",
    good: true,
  },
  poison: {
    pill: "STEP 02 / CONTRIBUTOR RISK",
    title: "Quarantine contributor C4.",
    copy: "Twelve images from one source share the same suspicious signature. The finding is aggregated into an action, not left as a pile of image IDs.",
    finding: "C4 flagged 12 / 72 samples. P(quarantine threshold) = 1.000.",
    check:
      "Duplicates, out-of-distribution images, and label disagreement by source.",
    why: "An image list does not tell an operator who to stop.",
    action: "Quarantine C4; review the remaining sources.",
    verdict: "QUARANTINE RECOMMENDED",
    image: "../data/synth10/aircraft/aircraft_00011.png",
    caption: "flagged sample / C4",
    good: false,
  },
  model: {
    pill: "STEP 03 / MODEL IDENTITY",
    title: "The file was swapped.",
    copy: "The filename stayed the same. At access tier 0, the model answers alone are enough to show that this is not the enrolled model.",
    finding: "Fingerprint distance 0.060. Expected same-model distance: 1e-15.",
    check: "200 fixed questions against the fingerprint recorded at intake.",
    why: "A vendor can change the file while keeping its name unchanged.",
    action: "Stop deployment and request the enrolled artifact.",
    verdict: "MODEL SUBSTITUTED",
    image: "../data/synth10/aircraft/aircraft_00018.png",
    caption: "black-box probe",
    good: false,
  },
  receipt: {
    pill: "STEP 04 / RECEIPT CHAIN",
    title: "One answer was edited.",
    copy: "The cryptographic chain breaks at record 347. No statistical judgement is needed: the signature does not match what was signed.",
    finding: "Record #347 / SIGNATURE_INVALID / chain stops here.",
    check:
      "Signature, sequence, nonce, timestamp, and link to the prior receipt.",
    why: "The record can be proven different from what was signed.",
    action: "Reject the log and investigate record 347.",
    verdict: "RECEIPT TAMPERED",
    image: "../data/synth10/aircraft/aircraft_00024.png",
    caption: "receipt #347",
    good: false,
  },
};
const $ = (id) => document.getElementById(id),
  cards = document.querySelectorAll(".step-card");
const contributors = [
  {
    id: "C0",
    samples: 61,
    flagged: 2,
    rate: 3,
    interval: "[1.0, 11.2]%",
    chance: "39.5%",
    disposition: "accept",
  },
  {
    id: "C1",
    samples: 61,
    flagged: 1,
    rate: 2,
    interval: "[0.4, 8.7]%",
    chance: "17.7%",
    disposition: "accept",
  },
  {
    id: "C2",
    samples: 61,
    flagged: 2,
    rate: 3,
    interval: "[1.0, 11.2]%",
    chance: "39.5%",
    disposition: "accept",
  },
  {
    id: "C3",
    samples: 60,
    flagged: 3,
    rate: 5,
    interval: "[1.8, 13.7]%",
    chance: "63.6%",
    disposition: "review",
  },
  {
    id: "C4",
    samples: 72,
    flagged: 12,
    rate: 17,
    interval: "[9.8, 27.0]%",
    chance: "100%",
    disposition: "quarantine",
  },
];
const phaseFor = {
    clean: "data",
    poison: "data",
    model: "model",
    receipt: "receipts",
  },
  phaseCopy = {
    data: "Start with the data that entered the pipeline.",
    model: "Prove the model is the one you enrolled.",
    receipts: "Trace every answer back to a signed record.",
  };
const phaseData = {
  data: {
    title: "DATA / source signal map",
    copy: "See how five contributors separate into accepted, review, and quarantine decisions.",
    kpis: ["SOURCES", "05", "FLAGGED", "20", "ACTION", "C4"],
    caption:
      "Each bar is one contributor. Hover-ready evidence is rendered locally.",
    cursor: "C4 / highest risk",
    events: [
      ["09:41:52", "C4", "12 samples crossed the quarantine threshold", "bad"],
      ["09:41:54", "C3", "3 samples moved to analyst review", "warn"],
      ["09:41:58", "C0–C2", "No source-level action required", "good"],
    ],
  },
  model: {
    title: "MODEL / identity fingerprint",
    copy: "Compare the enrolled answers with the file currently in service. A filename is not an identity.",
    kpis: ["PROBES", "200", "DISTANCE", "0.060", "STATUS", "SWAP"],
    caption:
      "The coral trace is the live model; the lime trace is the enrolled baseline.",
    cursor: "distance 0.060 / mismatch",
    events: [
      ["09:42:04", "PROBE 001", "Baseline and live model diverged", "bad"],
      ["09:42:06", "PROBE 087", "Confidence drift confirmed", "warn"],
      ["09:42:08", "VERDICT", "Model substitution recommended", "bad"],
    ],
  },
  receipts: {
    title: "RECEIPTS / chain integrity",
    copy: "Follow the signed sequence from one answer to the next. One broken link is enough to stop the log.",
    kpis: ["CHECKED", "500", "VALID", "499", "BREAK", "347"],
    caption:
      "The chain is continuous until record 347, where the signature no longer matches.",
    cursor: "record 347 / broken link",
    events: [
      ["09:42:12", "RECORD 346", "Signature and nonce verified", "good"],
      ["09:42:13", "RECORD 347", "SIGNATURE_INVALID detected", "bad"],
      ["09:42:13", "CHAIN", "Verification stopped safely", "warn"],
    ],
  },
};
let activePhase = "data",
  activeLabMode = "overview";
function selectPhase(name) {
  activePhase = name;
  document
    .querySelectorAll(".phase-tab")
    .forEach((tab) =>
      tab.classList.toggle("active", tab.dataset.phase === name),
    );
  document.body.dataset.phase = name;
  $("phaseDescription").textContent = phaseCopy[name];
  renderPhaseLab();
}
function selectScenario(name) {
  const data = scenarios[name];
  cards.forEach((card) => {
    card.classList.toggle("active", card.dataset.scenario === name);
    card.setAttribute(
      "aria-current",
      card.dataset.scenario === name ? "step" : "false",
    );
  });
  selectPhase(phaseFor[name]);
  $("detailPill").textContent = data.pill;
  $("detailTitle").textContent = data.title;
  $("detailCopy").textContent = data.copy;
  $("findingText").textContent = data.finding;
  $("checkText").textContent = data.check;
  $("whyText").textContent = data.why;
  $("actionText").textContent = data.action;
  $("verdict").textContent = data.verdict;
  $("sampleImage").src = data.image;
  $("imageCaption").textContent = data.caption;
  document.querySelector(".finding-dot").className =
    `finding-dot ${data.good ? "good-dot" : "bad-dot"}`;
}
cards.forEach((card) =>
  card.addEventListener("click", () => selectScenario(card.dataset.scenario)),
);
document
  .querySelectorAll(".phase-tab")
  .forEach((tab) =>
    tab.addEventListener("click", () =>
      selectScenario(
        tab.dataset.phase === "data"
          ? "poison"
          : tab.dataset.phase === "model"
            ? "model"
            : "receipt",
      ),
    ),
  );
function renderPhaseLab() {
  const data = phaseData[activePhase];
  $("labTitle").textContent = data.title;
  $("labCopy").textContent = data.copy;
  $("chartCaption").textContent = data.caption;
  $("chartCursor").textContent = data.cursor;
  [
    ["kpiOneLabel", "kpiOne"],
    ["kpiTwoLabel", "kpiTwo"],
    ["kpiThreeLabel", "kpiThree"],
  ].forEach((pair, index) => {
    $(pair[0]).textContent = data.kpis[index * 2];
    $(pair[1]).textContent = data.kpis[index * 2 + 1];
  });
  $("feedCount").textContent =
    `${data.events.length.toString().padStart(2, "0")} events`;
  $("phaseEvents").innerHTML = data.events
    .map(
      (event) =>
        `<div class="feed-event"><span class="feed-time">${event[0]}</span><span class="feed-code ${event[3]}">${event[1]}</span><span>${event[2]}</span></div>`,
    )
    .join("");
  drawPhaseChart();
}
function drawPhaseChart() {
  const canvas = $("phaseChart"),
    ctx = canvas.getContext("2d"),
    w = 820,
    h = 360,
    dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#d8d4c9";
  ctx.fillStyle = "#787b72";
  ctx.font = "11px SFMono-Regular, monospace";
  for (let i = 0; i < 5; i++) {
    const y = 65 + i * 58;
    ctx.beginPath();
    ctx.moveTo(54, y);
    ctx.lineTo(785, y);
    ctx.stroke();
    ctx.fillText(`${20 - i * 5}%`, 8, y + 4);
  }
  if (activePhase === "data") {
    contributors.forEach((item, index) => {
      const x = 90 + index * 135,
        value = item.rate,
        bar = (value / 20) * 232;
      ctx.fillStyle =
        item.disposition === "quarantine"
          ? "#ff7d65"
          : item.disposition === "review"
            ? "#f1bd67"
            : "#8cd9d2";
      if (activeLabMode === "overview") ctx.fillRect(x, 297 - bar, 58, bar);
      else {
        ctx.beginPath();
        ctx.arc(x + 29, 297 - bar, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#191b1a";
        ctx.stroke();
      }
      ctx.fillStyle = "#191b1a";
      ctx.fillText(item.id, x + 20, 325);
      ctx.fillStyle = "#787b72";
      ctx.fillText(`${item.rate}%`, x + 14, 285 - bar);
    });
  } else if (activePhase === "model") {
    ctx.strokeStyle = "#c6ee77";
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i < 12; i++) {
      const x = 60 + i * 65,
        y = 180 + Math.sin(i * 0.8) * 32 - (i % 3) * 7;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.strokeStyle = "#ff7d65";
    ctx.beginPath();
    for (let i = 0; i < 12; i++) {
      const x = 60 + i * 65,
        y =
          180 +
          Math.sin(i * 0.8) * 32 -
          (i % 3) * 7 +
          (i > 6 ? (i - 5) * 8 : 0);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#191b1a";
    ctx.font = "11px SFMono-Regular, monospace";
    ctx.fillText("enrolled", 60, 270);
    ctx.fillStyle = "#ff7d65";
    ctx.fillText("live model", 145, 270);
    ctx.fillStyle = "#787b72";
    ctx.fillText("probe sequence", 650, 325);
  } else {
    for (let i = 0; i < 13; i++) {
      const x = 60 + i * 60;
      ctx.strokeStyle = i === 7 ? "#ff7d65" : "#8cd9d2";
      ctx.lineWidth = i === 7 ? 6 : 3;
      ctx.beginPath();
      ctx.moveTo(x, 180);
      ctx.lineTo(x + 45, 180);
      ctx.stroke();
      ctx.fillStyle = i === 7 ? "#ff7d65" : "#191b1a";
      ctx.beginPath();
      ctx.arc(x + 22, 180, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(i === 7 ? "347" : `${345 + i}`, x + 8, 220);
    }
    ctx.fillStyle = "#ff7d65";
    ctx.font = "bold 12px SFMono-Regular, monospace";
    ctx.fillText("CHAIN BREAK", 470, 125);
    ctx.strokeStyle = "#ff7d65";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(500, 130);
    ctx.lineTo(500, 175);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
document.querySelectorAll(".lab-mode").forEach((button) =>
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".lab-mode")
      .forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activeLabMode = button.dataset.labMode;
    drawPhaseChart();
  }),
);
window.addEventListener("resize", drawPhaseChart);
function renderContributors() {
  const threshold = Number($("threshold").value);
  $("thresholdValue").value = `${threshold}%`;
  $("contributorList").innerHTML = contributors
    .map(
      (item) =>
        `<button class="contributor-row" data-contributor="${item.id}"><span class="contributor-id">${item.id}</span><span><strong>${item.flagged} / ${item.samples} flagged</strong><small>${item.disposition} / P &gt; 5%: ${item.chance}</small></span><b class="risk-rate ${item.rate > threshold ? "risk-high" : ""}">${item.rate}%</b></button>`,
    )
    .join("");
  document
    .querySelectorAll(".contributor-row")
    .forEach((row) =>
      row.addEventListener("click", () =>
        selectContributor(row.dataset.contributor),
      ),
    );
  drawChart();
}
function selectContributor(id) {
  document.querySelector(".analytics-section").classList.add("is-focused");
  document
    .querySelectorAll(".contributor-row")
    .forEach((row) =>
      row.classList.toggle("selected", row.dataset.contributor === id),
    );
  const item = contributors.find((entry) => entry.id === id);
  $("selectedContributor").textContent =
    `${item.id} selected / ${item.disposition} / credible interval ${item.interval}`;
  $("inspectorTitle").textContent = `${item.id} / ${item.disposition}`;
  $("inspectorSummary").textContent =
    `A complete view of ${item.id}'s contribution to the assurance decision.`;
  $("inspectorStatus").textContent = `${item.rate}% FLAGGED`;
  $("inspectorDataStat").textContent = `${item.flagged} / ${item.samples}`;
  $("inspectorDataMeter").style.width = `${Math.min(item.rate * 5, 100)}%`;
  $("inspectorDataNote").textContent =
    `Credible interval ${item.interval} · P(rate > 5%) ${item.chance}`;
  const modelDistance =
    item.id === "C4" ? "0.060" : item.id === "C3" ? "0.018" : "0.004";
  $("inspectorModelStat").textContent = modelDistance;
  $("inspectorModelMeter").style.width =
    `${Math.min(Number(modelDistance) * 1000, 100)}%`;
  $("inspectorModelNote").textContent =
    item.id === "C4"
      ? "Highest source risk is reflected in the live probe comparison."
      : "No contributor-specific model mismatch detected.";
  const receiptCount = item.id === "C4" ? "499 / 500" : "500 / 500";
  $("inspectorReceiptStat").textContent = receiptCount;
  $("inspectorReceiptMeter").style.width = item.id === "C4" ? "99.8%" : "100%";
  $("inspectorReceiptNote").textContent =
    item.id === "C4"
      ? "Review the source alongside receipt #347 before release."
      : "All signed records connected to this source verify cleanly.";
  $("contributorInspector").classList.add("is-open");
  $("contributorInspector").scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}
function showAllContributors() {
  document.querySelector(".analytics-section").classList.remove("is-focused");
  $("contributorInspector").classList.remove("is-open");
  $("selectedContributor").textContent = "Select a source to inspect";
}
function drawChart() {
  const canvas = $("riskChart"),
    ctx = canvas.getContext("2d"),
    threshold = Number($("threshold").value),
    style = document.querySelector(".viz-button.active").dataset.chart,
    dpr = window.devicePixelRatio || 1;
  canvas.width = 700 * dpr;
  canvas.height = 300 * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, 700, 300);
  ctx.font = "11px SFMono-Regular, monospace";
  ctx.fillStyle = "#787b72";
  ctx.strokeStyle = "#d8d4c9";
  for (let tick = 0; tick <= 20; tick += 5) {
    const y = 245 - (tick / 20) * 190;
    ctx.fillText(`${tick}%`, 8, y + 4);
    ctx.beginPath();
    ctx.moveTo(45, y);
    ctx.lineTo(680, y);
    ctx.stroke();
  }
  const xStep = 635 / contributors.length;
  contributors.forEach((item, index) => {
    const x = 70 + index * xStep,
      barHeight = (item.rate / 20) * 190;
    ctx.fillStyle =
      item.disposition === "quarantine"
        ? "#ff7d65"
        : item.disposition === "review"
          ? "#f1bd67"
          : "#8cd9d2";
    if (style === "bars") {
      ctx.fillRect(x - 24, 245 - barHeight, 48, barHeight);
    } else {
      ctx.beginPath();
      ctx.arc(x, 245 - barHeight, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#191b1a";
      ctx.beginPath();
      ctx.moveTo(
        x,
        245 - ((item.interval.includes("27.0") ? 27 : 12) / 20) * 190,
      );
      ctx.lineTo(
        x,
        245 - ((item.interval.includes("27.0") ? 9.8 : 1) / 20) * 190,
      );
      ctx.stroke();
    }
    ctx.fillStyle = "#191b1a";
    ctx.fillText(item.id, x - 8, 268);
  });
  ctx.strokeStyle = "#c55745";
  ctx.setLineDash([5, 5]);
  const thresholdY = 245 - (threshold / 20) * 190;
  ctx.beginPath();
  ctx.moveTo(45, thresholdY);
  ctx.lineTo(680, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);
  $("chartTitle").textContent =
    style === "bars"
      ? "Flagged rate by contributor"
      : "Credible interval by contributor";
}
document.querySelectorAll(".viz-button").forEach((button) =>
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".viz-button")
      .forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    drawChart();
  }),
);
$("threshold").addEventListener("input", renderContributors);
window.addEventListener("resize", drawChart);
renderContributors();
selectPhase("data");
$("backToContributors").addEventListener("click", showAllContributors);
document.querySelectorAll(".tier").forEach((tier) =>
  tier.addEventListener("click", (event) => {
    document
      .querySelectorAll(".tier")
      .forEach((item) => item.classList.remove("active"));
    event.currentTarget.classList.add("active");
    const labels = {
      0: "Answers only",
      1: "Weights available",
      2: "Internals available",
    };
    $("tierHelp").textContent = labels[event.currentTarget.dataset.tier];
    if (event.currentTarget.dataset.tier === "2")
      $("findingText").textContent =
        "Tier 2 enabled. Trigger reconstruction is now available to inspect.";
  }),
);
$("runButton").addEventListener("click", () => {
  const button = $("runButton");
  button.disabled = true;
  button.innerHTML = '<span class="button-icon">…</span> Running audit';
  let step = 0;
  const sequence = ["clean", "poison", "model", "receipt"];
  const timer = setInterval(() => {
    selectScenario(sequence[step]);
    step += 1;
    if (step === sequence.length) {
      clearInterval(timer);
      button.disabled = false;
      button.innerHTML = '<span class="button-icon">✓</span> Run again';
      $("toast").classList.add("show");
      setTimeout(() => $("toast").classList.remove("show"), 3200);
    }
  }, 650);
});
$("evidenceButton").addEventListener("click", () => {
  $("toast").textContent =
    "Evidence detail is available in the generated report.";
  $("toast").classList.add("show");
  setTimeout(() => $("toast").classList.remove("show"), 3200);
});
$("flowButton").addEventListener("click", () =>
  document.querySelector("#flow").scrollIntoView({ behavior: "smooth" }),
);
$("themeButton").addEventListener("click", () =>
  document.body.classList.toggle("high-contrast"),
);
