// --- tokenizer ---
const vocab = ["<pad>", "<bos>", "<eos>", "hello", "how", "are", "you", "i", "am", "fine"];
const wordToId = {};
vocab.forEach((w, i) => wordToId[w] = i);

function encode(text) {
  return text.split(/\s+/).map(w => wordToId[w] ?? 0);
}
function decode(ids) {
  return ids.map(id => vocab[id] ?? "<unk>").join(" ");
}

// --- model ---
const model = new TinyTransformer(vocab.length, 32, 64, 16);

// --- dataset ---
let dataset = JSON.parse(localStorage.getItem("dataset") || "[]");

function saveDataset() {
  localStorage.setItem("dataset", JSON.stringify(dataset));
}

function renderDataset() {
  const div = document.getElementById("datasetList");
  if (dataset.length === 0) {
    div.textContent = "No examples yet.";
    return;
  }
  div.innerHTML = dataset
    .map((ex, i) => `<div><b>${i+1}.</b> ${ex.input} → ${ex.output}</div>`)
    .join("");
}

renderDataset();

// --- Add example ---
document.getElementById("addExampleBtn").onclick = () => {
  const inp = document.getElementById("trainIn").value.trim();
  const out = document.getElementById("trainOut").value.trim();
  if (!inp || !out) return;

  dataset.push({ input: inp, output: out });
  saveDataset();
  renderDataset();
};

// --- Clear dataset ---
document.getElementById("clearDatasetBtn").onclick = () => {
  dataset = [];
  saveDataset();
  renderDataset();
};

// --- Training step (dataset + trainable memory) ---
function trainExample(inputText, outputText) {
  const inputIds = encode("<bos> " + inputText);
  const targetIds = encode(outputText);

  const logits = model.forward(inputIds);
  const lastLogits = logits[logits.length - 1];

  const maxVal = Math.max(...lastLogits);
  const exps = lastLogits.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a,b)=>a+b,0);
  const probs = exps.map(v => v/sum);

  const target = targetIds[0];
  const lr = 0.01;

  for (let i = 0; i < model.vocabSize; i++) {
    const grad = probs[i] - (i === target ? 1 : 0);
    model.bo[i] -= lr * grad;

    for (let j = 0; j < model.dModel; j++) {
      const h = logits[logits.length - 2][j];
      model.Wo[j][i] -= lr * grad * h;
    }
  }
}

// --- Train on dataset ---
document.getElementById("trainAllBtn").onclick = () => {
  dataset.forEach(ex => trainExample(ex.input, ex.output));
  addBotMessage("Training complete on " + dataset.length + " examples.");
};

// --- Chat + memory ---

// short‑term conversation memory
let chatHistory = [];

// long‑term memory (persists across sessions)
let longTermMemory = JSON.parse(localStorage.getItem("longTermMemory") || "[]");

function saveLongTermMemory() {
  localStorage.setItem("longTermMemory", JSON.stringify(longTermMemory));
}

const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "chatMsg userMsg";
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addBotMessage(text) {
  const div = document.createElement("div");
  div.className = "chatMsg botMsg";
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// detect "remember" pattern and store + train
function maybeHandleMemoryCommand(userText) {
  const lower = userText.toLowerCase();
  if (!lower.startsWith("remember")) return false;

  const fact = userText.replace(/^remember( that)?/i, "").trim();
  if (!fact) return false;

  longTermMemory.push(fact);
  saveLongTermMemory();

  trainExample("what did i tell you to remember?", fact);

  addBotMessage("Okay, I'll remember: " + fact);
  return true;
}

function buildContext(userText) {
  const recent = chatHistory.slice(-4).join(" ");

  const memoryPrefix = longTermMemory.length
    ? "memory: " + longTermMemory.join(" ; ") + " "
    : "";

  return memoryPrefix + recent + " user: " + userText;
}

function generateReply(userText) {
  chatHistory.push("user: " + userText);

  const ctx = buildContext(userText);
  const ids = encode("<bos> " + ctx);
  const out = model.generate(ids, 12);
  const reply = decode(out);

  chatHistory.push("bot: " + reply);
  return reply;
}

// unified send function
function sendMessage() {
  const msg = chatInput.value.trim();
  if (!msg) return;

  addUserMessage(msg);
  chatInput.value = "";

  if (maybeHandleMemoryCommand(msg)) return;

  const reply = generateReply(msg);
  addBotMessage(reply);
}

// Enter key
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter") {
    sendMessage();
  }
});

// Send button
sendBtn.addEventListener("click", () => {
  sendMessage();
});



