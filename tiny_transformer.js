// tiny_transformer.js
// Minimal transformer-style language model from scratch in JS (forward + generation only)

//////////////////////
// Utility functions
//////////////////////

function randn(scale = 0.02) {
  // Box-Muller
  const u = Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return z * scale;
}

function zeros(rows, cols) {
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) {
    m[i] = new Array(cols).fill(0);
  }
  return m;
}

function matMul(A, B) {
  const n = A.length;
  const d = A[0].length;
  const m = B[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < d; k++) {
      const aik = A[i][k];
      for (let j = 0; j < m; j++) {
        out[i][j] += aik * B[k][j];
      }
    }
  }
  return out;
}

function addBias(X, b) {
  const out = X.map(row => row.slice());
  for (let i = 0; i < out.length; i++) {
    for (let j = 0; j < out[0].length; j++) {
      out[i][j] += b[j];
    }
  }
  return out;
}

function softmaxRow(row) {
  const maxVal = Math.max(...row);
  const exps = row.map(v => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function softmax2D(X) {
  return X.map(softmaxRow);
}

function relu2D(X) {
  return X.map(row => row.map(v => Math.max(0, v)));
}

function transpose(X) {
  const rows = X.length;
  const cols = X[0].length;
  const out = zeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j][i] = X[i][j];
    }
  }
  return out;
}

function scale2D(X, s) {
  return X.map(row => row.map(v => v * s));
}

function add2D(A, B) {
  const out = zeros(A.length, A[0].length);
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[0].length; j++) {
      out[i][j] = A[i][j] + B[i][j];
    }
  }
  return out;
}

///////////////////////////
// Tiny transformer block
///////////////////////////

class TinyTransformer {
  constructor(vocabSize, dModel = 32, dFF = 64, maxLen = 16) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.dFF = dFF;
    this.maxLen = maxLen;

    // token embeddings: vocabSize x dModel
    this.E = zeros(vocabSize, dModel);
    for (let i = 0; i < vocabSize; i++) {
      for (let j = 0; j < dModel; j++) {
        this.E[i][j] = randn();
      }
    }

    // positional embeddings: maxLen x dModel
    this.P = zeros(maxLen, dModel);
    for (let i = 0; i < maxLen; i++) {
      for (let j = 0; j < dModel; j++) {
        this.P[i][j] = randn();
      }
    }

    // self-attention weights (single head)
    this.Wq = zeros(dModel, dModel);
    this.Wk = zeros(dModel, dModel);
    this.Wv = zeros(dModel, dModel);
    this.bq = new Array(dModel).fill(0);
    this.bk = new Array(dModel).fill(0);
    this.bv = new Array(dModel).fill(0);

    // feed-forward
    this.W1 = zeros(dModel, dFF);
    this.b1 = new Array(dFF).fill(0);
    this.W2 = zeros(dFF, dModel);
    this.b2 = new Array(dModel).fill(0);

    // output projection to vocab
    this.Wo = zeros(dModel, vocabSize);
    this.bo = new Array(vocabSize).fill(0);

    // init weights
    const mats = [this.Wq, this.Wk, this.Wv, this.W1, this.W2, this.Wo];
    for (const M of mats) {
      for (let i = 0; i < M.length; i++) {
        for (let j = 0; j < M[0].length; j++) {
          M[i][j] = randn();
        }
      }
    }
  }

  embed(tokens) {
    const L = tokens.length;
    const X = zeros(L, this.dModel);
    for (let t = 0; t < L; t++) {
      const idx = tokens[t];
      for (let j = 0; j < this.dModel; j++) {
        X[t][j] = this.E[idx][j] + this.P[t][j];
      }
    }
    return X;
  }

  selfAttention(X) {
    const L = X.length;
    const Q = addBias(matMul(X, this.Wq), this.bq);
    const K = addBias(matMul(X, this.Wk), this.bk);
    const V = addBias(matMul(X, this.Wv), this.bv);

    const KT = transpose(K);
    let scores = matMul(Q, KT);
    const scale = 1 / Math.sqrt(this.dModel);
    scores = scale2D(scores, scale);

    // causal mask: no looking ahead
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        if (j > i) scores[i][j] = -1e9;
      }
    }

    const attn = softmax2D(scores);
    const out = matMul(attn, V);
    return out;
  }

  feedForward(X) {
    const h1 = relu2D(addBias(matMul(X, this.W1), this.b1));
    const h2 = addBias(matMul(h1, this.W2), this.b2);
    return h2;
  }

  forward(tokens) {
    let X = this.embed(tokens);
    const attnOut = this.selfAttention(X);
    X = add2D(X, attnOut); // residual
    const ffOut = this.feedForward(X);
    X = add2D(X, ffOut);   // residual
    const logits = addBias(matMul(X, this.Wo), this.bo);
    return logits;
  }

  generate(prefixTokens, maxNewTokens = 10) {
    const tokens = prefixTokens.slice();
    for (let step = 0; step < maxNewTokens; step++) {
      const ctx = tokens.slice(-this.maxLen);
      const logits = this.forward(ctx);
      const lastLogits = logits[logits.length - 1];
      const probs = softmaxRow(lastLogits);

      // sample
      let r = Math.random();
      let cum = 0;
      let next = probs.length - 1;
      for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (r <= cum) {
          next = i;
          break;
        }
      }
      tokens.push(next);
    }
    return tokens;
  }
}

//////////////////////
// Toy usage example
//////////////////////

// tiny vocab + tokenizer
const vocab = ["<pad>", "<bos>", "<eos>", "hello", "how", "are", "you", "i", "am", "fine"];
const wordToId = {};
vocab.forEach((w, i) => (wordToId[w] = i));

function encode(words) {
  return words.map(w => wordToId[w] ?? 0);
}
function decode(ids) {
  return ids.map(id => vocab[id] ?? "<unk>");
}

// create model
const model = new TinyTransformer(vocab.length, 32, 64, 16);

// prefix: pretend user says "hello how are you"
const prefixWords = ["<bos>", "hello", "how", "are", "you"];
const prefixIds = encode(prefixWords);

const genIds = model.generate(prefixIds, 8);
const genWords = decode(genIds);

console.log("Prefix:", prefixWords.join(" "));
console.log("Generated:", genWords.join(" "));
