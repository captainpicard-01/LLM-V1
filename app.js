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

// --- Training step (same as before) ---
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

// --- Train on entire dataset ---
document.getElementById("trainAllBtn").onclick = () => {
  dataset.forEach(ex => trainExample(ex.input, ex.output));
  document.getElementById("output").textContent =
    "Training complete on " + dataset.length + " examples.";
};

// --- Generate ---
document.getElementById("genBtn").onclick = () => {
  const text = document.getElementById("genInput").value.trim();
  const ids = encode("<bos> " + text);
  const out = model.generate(ids, 12);
  document.getElementById("output").textContent = decode(out);
};

