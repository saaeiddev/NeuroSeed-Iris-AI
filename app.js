"use strict";
/**
 * NeuroSeed: genuine trained PyTorch weights, executed as plain JavaScript.
 * Storage order: each layer's row-major weight matrix followed by its bias.
 * Then the training-set feature means and standard deviations.
 * 4x12 + 12 + 12x8 + 8 + 8x3 + 3 = 191 learnable parameters.
 * The final 8 floats store normalization constants (total 199).
 * This page runs offline after being downloaded: it never calls an AI API.
 */
const MODEL_BASE64 = "MiO1PgPOpj5mh5Q+O1B2P7Bc4L04os49EEV5vihYlj7AYpo9T32rvvz9aj+swCE/TSR+PiX1Jb1vGRo+vt8kvjaBiD7RHWa9NUyzvktEVzw1OzS/7zDcPinUL7/Dxgy+x3Y4vyIYwj0qOwu/jNE6v/OHFD/79+S+wQ+SPmUnWL9FaLA+y5fvviA7fD6FQEk/qBXLvty5BT/ZzPm+Z9Mzv66+4z30LzO/Z8UUP/XqWz9JHwQ+gwkAv/o4Az7eeKO97cozPIYInL4m2ge/k/Oyvg3b/r4jyZ0+zNMgP1oyCz+X5A8/luXoPONdWz/iV+2+Qg9FvtvCSb4U1uu9tB58vkn+eLzgEfU+L39KPzRlkT1ord++aYmmPptzsb3kXSC9X/0RP1T0c77/T188GHZNvVFRlb2VsyQ+4EEEve/3ij47/Ds+UkbHvoX7Az+J40E9utPsPjFLjL1eSi4/3ytLvc/NAL2oRNw+zqFmPrOTFb8MfLw+GHMTPyZ0QTx4zaq8mRhaPgp+x72OXR2+mkGpvoFIKb6Dr5M9DBIcvQaTVr4PkTa90utJvrGul74YxyK+dxVCPs44PL7BpmY/9+GZPv7MVj76ETy+/Gv2viVCJb9SSs8+q6tvvihYlz77gvm9RhKaPmr9271lXKI+RyB1vKR3ob0QAPk8AF/GvFdN57sgFhA/mcFDvtWTFT8bKoE+43TvPGORTD4zfQC+JkFvPYmIF75K5xO+vhTXPj/x9j5eIBE/LAw0vy2gJT9wmCs+TB+hPmb4dj2akj0/L0jcPiz2xD59hZ690dcTvj8QDL9RN0g+ZzChvVM9Cz8ADyq+clYnPxEa5T4iUum99mNUvtStFL7lgrA+bvwDP3uO3r4m6S0/pjvkvvSHoj5xWIE+ws+Qvn4jFr8+WiG/sRmtvrpcP75NSM8+xTU8v+VjML47nhG/U+SnPi1ZUD9wIF+/+AK+vjPzfj56hUA+FOuYvYVtUz+JSLo+XDsMvgEO4T76gLk+EIG8PtxSDb/6xbpAddpAQDUScEDLUJk/rn9cP+z00z4kkOE/N81BPw==";
const SIZES = [4, 12, 8, 3];
const CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"];
const FEATURE_NAMES = ["Sepal length", "Sepal width", "Petal length", "Petal width"];
const PRESETS = [
  [5.1, 3.5, 1.4, 0.2],
  [6.0, 2.7, 4.5, 1.5],
  [6.7, 3.0, 5.2, 2.3]
];
let selectedNeuron = 0;
let lastForward = null;

function decodeWeights() {
  const binary = atob(MODEL_BASE64);
  if (binary.length !== 199 * 4) throw new Error("Invalid model data length");
  const view = new DataView(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i++) view.setUint8(i, binary.charCodeAt(i));
  let offset = 0;
  const layers = [];
  for (let i = 1; i < SIZES.length; i++) {
    const nIn = SIZES[i - 1], nOut = SIZES[i];
    const count = nIn * nOut;
    const weights = [];
    for (let j = 0; j < count; j++) weights.push(view.getFloat32(4 * offset++, true));
    const biases = [];
    for (let j = 0; j < nOut; j++) biases.push(view.getFloat32(4 * offset++, true));
    layers.push({weights:weights, biases:biases, inputSize:nIn, outputSize:nOut});
  }
  const means = [], stds = [];
  for (let i = 0; i < 4; i++) means.push(view.getFloat32(4 * offset++, true));
  for (let i = 0; i < 4; i++) stds.push(view.getFloat32(4 * offset++, true));
  if (offset !== 199) throw new Error("Invalid model tensor layout");
  return {layers:layers, means:means, stds:stds};
}
const model = decodeWeights();

function dense(input, layer, applyRelu) {
  const preActivation = [], activation = [];
  for (let o = 0; o < layer.outputSize; o++) {
    let z = layer.biases[o];
    for (let i = 0; i < layer.inputSize; i++) {
      z += input[i] * layer.weights[o * layer.inputSize + i];
    }
    preActivation.push(z);
    activation.push(applyRelu ? Math.max(0, z) : z);
  }
  return {z:preActivation, a:activation};
}

function infer(raw) {
  if (raw.length !== 4 || raw.some(function(x){return !Number.isFinite(x);})) {
    throw new Error("Please enter four valid numeric measurements.");
  }
  const standardized = raw.map(function(v, i){return (v - model.means[i]) / model.stds[i];});
  const hidden1 = dense(standardized, model.layers[0], true);
  const hidden2 = dense(hidden1.a, model.layers[1], true);
  const output = dense(hidden2.a, model.layers[2], false);
  const maxLogit = Math.max.apply(null, output.z);
  const exps = output.z.map(function(z){return Math.exp(z - maxLogit);});
  const denom = exps.reduce(function(a,b){return a+b;},0);
  const probs = exps.map(function(v){return v/denom;});
  const predictedIndex = probs.indexOf(Math.max.apply(null, probs));
  return {raw:raw, standardized:standardized, hidden1:hidden1, hidden2:hidden2, output:output, probs:probs, predictedIndex:predictedIndex};
}

function $(id) {return document.getElementById(id);}
function fmt(v, decimals) {return (v >= 0 ? "+" : "") + v.toFixed(decimals === undefined ? 3 : decimals);}
function setPreset(index) {
  PRESETS[index].forEach(function(value, i) {
    $("number-" + i).value = value;
    $("range-" + i).value = value;
  });
  document.querySelectorAll("[data-preset]").forEach(function(btn) {
    btn.classList.toggle("active", Number(btn.dataset.preset) === index);
  });
  run();
}
function renderPrediction(data) {
  const best = data.predictedIndex;
  const confidence = data.probs[best] * 100;
  $("winner").textContent = CLASS_NAMES[best];
  $("winner-text").textContent = "Highest output score across the three flower classes.";
  $("confidence").textContent = confidence.toFixed(1) + "%";
  $("confidence-ring").style.background = "conic-gradient(#7be8f6 " + confidence + "%, #283c55 " + confidence + "%)";
  data.probs.forEach(function(p, i) {
    $("score-" + i).textContent = (p * 100).toFixed(2) + "%";
    $("bar-" + i).style.width = (p * 100) + "%";
  });
  $("compute-status").textContent = "Inference complete · 4 inputs → 3 outputs";
}

function renderInspector(data) {
  const n = selectedNeuron;
  $("selected-neuron").textContent = "H1 · " + String(n + 1).padStart(2,"0");
  const terms = $("math-terms");
  terms.replaceChildren();
  FEATURE_NAMES.forEach(function(name, i){
    const x = data.standardized[i];
    const w = model.layers[0].weights[n * 4 + i];
    const item = document.createElement("div");
    item.className = "math-term";
    const feature = document.createElement("span");
    feature.textContent = "x" + (i+1) + " · w" + (i+1) + "  /  " + name;
    const val = document.createElement("strong");
    val.textContent = fmt(x,3) + " × " + fmt(w,3) + " = " + fmt(x*w,3);
    item.append(feature,val);
    terms.appendChild(item);
  });
  const biasRow = document.createElement("div");
  biasRow.className = "math-term";
  const biasLabel = document.createElement("span");
  biasLabel.textContent = "Learned bias";
  const biasValue = document.createElement("strong");
  biasValue.textContent = fmt(model.layers[0].biases[n], 4);
  biasRow.append(biasLabel,biasValue);
  terms.appendChild(biasRow);
  $("z-value").textContent = fmt(data.hidden1.z[n], 4);
  $("activation-value").textContent = data.hidden1.a[n].toFixed(4);
}
const SVG_NS="http://www.w3.org/2000/svg";
function svgElement(name, attrs) {
  const el = document.createElementNS(SVG_NS,name);
  Object.keys(attrs).forEach(function(k){el.setAttribute(k,attrs[k]);});
  return el;
}
function neuronPositions() {
  const xs = [84, 328, 592, 836];
  const ys = SIZES.map(function(n) {
    const top = n > 9 ? 67 : 112;
    const bottom = n > 9 ? 440 : 408;
    return Array.from({length:n},function(_,i){return n === 1 ? 250 : top+(bottom-top)*i/(n-1);});
  });
  return ys.map(function(layer,index) {return layer.map(function(y){return {x:xs[index], y:y};});});
}
function renderNetwork(data) {
  const svg=$("network-svg");
  svg.replaceChildren();
  const title=svgElement("title",{});
  title.textContent="Real neural network, 4 input, 12 + 8 hidden, 3 output";
  svg.appendChild(title);
  const pos=neuronPositions();
  const activations=[data.standardized,data.hidden1.a,data.hidden2.a,data.probs];
  const colors=["#78e6f7","#a998ff","#a998ff","#ffc992"];
  const labels=["INPUTS","HIDDEN 01","HIDDEN 02","OUTPUTS"];
  labels.forEach(function(label,i) {
    const text=svgElement("text",{x:pos[i][0].x,y:"27","text-anchor":"middle",fill:"#8ea5c6","font-size":"11","font-weight":"700","letter-spacing":"2"});
    text.textContent=label;
    svg.appendChild(text);
  });
  const wires=svgElement("g",{"aria-hidden":"true"});
  for(let layer=0;layer<3;layer++) {
    const weights=model.layers[layer].weights;
    for(let o=0;o<pos[layer+1].length;o++) {
      for(let i=0;i<pos[layer].length;i++) {
        const w=weights[o*pos[layer].length+i];
        const signal=Math.abs(activations[layer][i]);
        const strength=Math.min(1,Math.abs(w)*0.65);
        const line=svgElement("line",{
          x1:pos[layer][i].x+12,y1:pos[layer][i].y,
          x2:pos[layer+1][o].x-12,y2:pos[layer+1][o].y,
          stroke:w>=0?"#72dfee":"#c59efb",
          "stroke-width":Math.max(0.5,Math.min(2.0,0.65+Math.abs(w)*0.42)),
          "stroke-opacity":Math.min(.48, .055+.13*strength+.035*Math.min(signal,3)),
          "stroke-linecap":"round"
        });
        wires.appendChild(line);
      }
    }
  }
  svg.appendChild(wires);
  pos.forEach(function(layerPos,layerIndex){
    const vals=activations[layerIndex];
    const max=Math.max(0.001,...vals.map(function(v){return Math.abs(v);}));
    layerPos.forEach(function(pos1,i) {
      const selected=layerIndex===1&&i===selectedNeuron;
      const radius=layerIndex===1?9:layerIndex===2?11:14;
      const strength=Math.min(1,Math.abs(vals[i])/max);
      if(selected){
        const glow=svgElement("circle",{cx:pos1.x,cy:pos1.y,r:radius+9,fill:colors[layerIndex],"fill-opacity":"0.13",stroke:"#b5f7ff","stroke-width":"1.5","stroke-opacity":"0.85"});
        svg.appendChild(glow);
      }
      const circle=svgElement("circle",{
        cx:pos1.x,cy:pos1.y,r:radius,fill:colors[layerIndex],
        "fill-opacity":.30+strength*.65,stroke:selected?"#fff":colors[layerIndex],
        "stroke-width":selected?2.5:1.2,
        "stroke-opacity":selected?1:.85,
        class:layerIndex===1?"clickable":""
      });
      if(layerIndex===1){
        circle.setAttribute("tabindex","0");
        circle.setAttribute("role","button");
        circle.setAttribute("aria-label","Inspect hidden layer 1 neuron "+(i+1));
        circle.addEventListener("click",function(){selectedNeuron=i;renderInspector(lastForward);renderNetwork(lastForward);});
        circle.addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();selectedNeuron=i;renderInspector(lastForward);renderNetwork(lastForward);}});
      }
      svg.appendChild(circle);
    });
  });
}
function run() {
  try {
    const numbers=Array.from({length:4},function(_,i){return Number($("number-"+i).value);});
    const invalid=numbers.some(function(v,i){
      const field=$("number-"+i);
      return field.value.trim()===""||!Number.isFinite(v)||v<Number(field.min)||v>Number(field.max);
    });
    if(invalid) throw new Error("Enter values within the range shown for each feature.");
    lastForward=infer(numbers);
    renderPrediction(lastForward);
    renderInspector(lastForward);
    renderNetwork(lastForward);
  }catch(error) {
    $("compute-status").textContent=error.message;
  }
}
function setup() {
  for(let i=0;i<4;i++){
    const num=$("number-"+i);
    const range=$("range-"+i);
    range.addEventListener("input",function(){
      num.value=range.value;
      document.querySelectorAll("[data-preset]").forEach(function(btn){btn.classList.remove("active");});
      run();
    });
    num.addEventListener("change",function(){
      if(num.value!=="") range.value=num.value;
      document.querySelectorAll("[data-preset]").forEach(function(btn){btn.classList.remove("active");});
      run();
    });
  }
  document.querySelectorAll("[data-preset]").forEach(function(btn){
    btn.addEventListener("click",function(){setPreset(Number(btn.dataset.preset));});
  });
  $("run").addEventListener("click",run);
  run();
  // Regression checks against the PyTorch model evaluated on the same inputs.
  const expected=[
    [PRESETS[0],0,.998216],
    [PRESETS[1],1,.760366],
    [PRESETS[2],2,.996240]
  ];
  expected.forEach(function(sample){
    const result=infer(sample[0]);
    if(result.predictedIndex!==sample[1] || Math.abs(result.probs[sample[1]]-sample[2])>0.00005){
      console.error("NeuroSeed regression mismatch",sample,result);
    }
  });
}
if(document.readyState==="loading"){document.addEventListener("DOMContentLoaded",setup);}
else{setup();}
