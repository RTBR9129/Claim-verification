const get = id => document.getElementById(id);
const form = get("claim-form");
const claimBox = get("claim");
const checkButton = get("check-button");
const randomButton = get("random-button");
const moreButton = get("show-more");
const extraEvidence = get("extra-evidence");

function message(text, error = false) {
  get("input-message").textContent = text;
  get("input-message").classList.toggle("error", error);
}

function busy(on) {
  checkButton.disabled = on;
  randomButton.disabled = on;
  claimBox.readOnly = on;
  form.setAttribute("aria-busy", String(on));
}

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

randomButton.addEventListener("click", async () => {
  busy(true);
  message("Picking a claim…");
  try {
    const data = await request("/api/random");
    claimBox.value = data.claim;
    message("Random claim loaded. You can edit it before checking.");
  } catch (error) {
    message(error.message, true);
  } finally {
    busy(false);
    claimBox.focus();
  }
});

claimBox.addEventListener("input", () => message(""));
form.addEventListener("submit", async event => {
  event.preventDefault();
  const claim = claimBox.value.trim();
  if (!claim) {
    message("Enter a claim first.", true);
    claimBox.focus();
    return;
  }
  busy(true);
  message("Checking the claim. The first check may take a little longer.");
  try {
    const result = await request("/api/verify", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({claim}),
    });
    showResult(result);
  } catch (error) {
    message(error.message, true);
  } finally {
    busy(false);
  }
});

function textElement(tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

function showScores(scores) {
  const container = get("scores");
  container.replaceChildren();
  get("score-note").textContent = scores
    ? "Scores from the evidence model."
    : "No evidence scores were available for this claim.";
  if (!scores) return;
  for (const [key, label, style] of [
    ["supports", "Support", "support"],
    ["refutes", "Refute", "refute"],
    ["neutral", "Neutral", "neutral"],
  ]) {
    const value = scores[key];
    if (!Number.isFinite(value)) continue;
    const card = document.createElement("div");
    card.className = "score " + style;
    const meter = document.createElement("meter");
    meter.min = 0;
    meter.max = 1;
    meter.value = value;
    meter.setAttribute("aria-label", label + " score");
    card.append(textElement("strong", (value * 100).toFixed(1) + "%"), textElement("span", label), meter);
    container.append(card);
  }
}

function showReview(result) {
  get("review").hidden = !result.review;
  const parts = result.parts;
  get("parts-unavailable").hidden = parts.length !== 0;
  get("parts").hidden = parts.length === 0;
  const resolved = get("resolved-list");
  const unresolved = get("unresolved-list");
  resolved.replaceChildren();
  unresolved.replaceChildren();
  for (const part of parts) {
    const list = part.label === "NOT ENOUGH INFO" ? unresolved : resolved;
    list.append(textElement("li", part.claim + (list === resolved ? " (" + part.label + ")" : "")));
  }
  if (!resolved.children.length) resolved.append(textElement("li", "None"));
  if (!unresolved.children.length) unresolved.append(textElement("li", "None"));
}

function passageContent(container, passage) {
  container.append(textElement("h3", passage.title), textElement("blockquote", passage.text));
}

function showResult(result) {
  get("checked-claim").textContent = result.claim;
  const verdict = get("verdict");
  verdict.textContent = result.label;
  const color = result.label === "SUPPORTS" ? "support" : result.label === "REFUTES" ? "refute" : "";
  verdict.className = "badge " + color;
  showScores(result.scores);
  showReview(result);

  const [first, ...others] = result.passages;
  const evidence = get("evidence");
  evidence.replaceChildren(textElement("h2", "Evidence excerpt"));
  if (first) passageContent(evidence, first);
  else evidence.append(textElement("p", "No passages were retrieved."));
  extraEvidence.replaceChildren();
  for (const passage of others) {
    const card = document.createElement("article");
    card.className = "card evidence";
    passageContent(card, passage);
    extraEvidence.append(card);
  }
  moreButton.hidden = others.length === 0;
  extraEvidence.hidden = true;
  moreButton.textContent = "Show more evidence";
  moreButton.setAttribute("aria-expanded", "false");

  get("input-screen").hidden = true;
  get("result-screen").hidden = false;
  get("step-input").removeAttribute("aria-current");
  get("step-result").setAttribute("aria-current", "step");
  get("result-title").focus();
}

moreButton.addEventListener("click", () => {
  const show = extraEvidence.hidden;
  extraEvidence.hidden = !show;
  moreButton.textContent = show ? "Show less evidence" : "Show more evidence";
  moreButton.setAttribute("aria-expanded", String(show));
});

get("back-button").addEventListener("click", () => {
  get("result-screen").hidden = true;
  get("input-screen").hidden = false;
  get("step-result").removeAttribute("aria-current");
  get("step-input").setAttribute("aria-current", "step");
  message("");
  claimBox.focus();
});
