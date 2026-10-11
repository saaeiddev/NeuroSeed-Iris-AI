/* NeuroSeed Signal Studio — interactive single-neuron sandbox.
   Uses the exact normalized inputs and learned first-layer weights from app.js.
   Sandbox edits NEVER mutate the trained classifier or its predictions. */
(function () {
  "use strict";
  const root = document.getElementById("neuron-lab");
  if (!root) return;
  const q = id => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const names = ["SEPAL LENGTH", "SEPAL WIDTH", "PETAL LENGTH", "PETAL WIDTH"];
  const colors = {plus:"#68f6dc", minus:"#ff8a95"};
  let index = 0, weights = [0,0,0,0], bias = 0, mode = "relu", edited = false;

  const node = (name, attrs, text) => {
    const el = document.createElementNS(NS, name);
    Object.entries(attrs || {}).forEach(([key,value]) => el.setAttribute(key,String(value)));
    if (text !== undefined) el.textContent=String(text);
    return el;
  };
  const nice = (n, dp=3) => (n >= 0 ? "+" : "") + n.toFixed(dp);
  const clamp = (v, a, b) => Math.min(b, Math.max(a,v));
  const inputs = () => {
    if (typeof lastForward !== "undefined" && lastForward) return lastForward.standardized;
    const raw = PRESETS[0];
    return raw.map((n,i)=>(n-model.means[i])/model.stds[i]);
  };
  const output = z => mode === "sigmoid" ? 1/(1+Math.exp(-z)) :
                        mode === "tanh" ? Math.tanh(z) : Math.max(0,z);

  function controls() {
    const slot = q("lab-weights");
    for (let i=0; i<4; i++) {
      const row = document.createElement("div"); row.className="lab-weight-row";
      const head = document.createElement("div"); head.className="lab-weight-head";
      const label = document.createElement("label");
      label.htmlFor="lab-weight-"+i; label.textContent="W"+(i+1)+" / "+names[i];
      const value = document.createElement("output"); value.id="lab-weight-value-"+i;
      head.append(label,value);
      const slider = document.createElement("input");
      slider.id="lab-weight-"+i; slider.type="range"; slider.min="-4"; slider.max="4";
      slider.step="0.01"; slider.setAttribute("aria-label","Weight "+(i+1));
      slider.addEventListener("input",()=>{
        weights[i]=Number(slider.value);
        edited=true; render();
      });
      row.append(head,slider); slot.append(row);
    }
    const menu=q("lab-neuron");
    for(let i=0;i<12;i++) {
      const option=document.createElement("option");
      option.value=String(i); option.textContent="H1 / neuron "+String(i+1).padStart(2,"0");
      menu.append(option);
    }
    menu.addEventListener("change",()=>loadNeuron(Number(menu.value),true));
    q("lab-bias").addEventListener("input",()=>{
      bias=Number(q("lab-bias").value); edited=true; render();
    });
    q("lab-activation").addEventListener("change",()=>{
      mode=q("lab-activation").value; edited=true; render();
    });
    q("lab-reset").addEventListener("click",()=>loadNeuron(index,true));
    q("lab-pulse").addEventListener("click",pulse);
    for(let i=0;i<4;i++) {
      ["number-"+i,"range-"+i].forEach(id => {
        q(id)?.addEventListener("input",()=>requestAnimationFrame(render));
        q(id)?.addEventListener("change",()=>requestAnimationFrame(render));
      });
    }
    document.querySelectorAll("[data-preset]").forEach(btn =>
      btn.addEventListener("click",()=>requestAnimationFrame(render)));
    q("run")?.addEventListener("click",()=>requestAnimationFrame(render));
    window.addEventListener("neuroseed:select",event=>{
      const n=event.detail?.neuron;
      if(Number.isInteger(n)&&n>=0&&n<12)loadNeuron(n,false);
    });
    loadNeuron(0,false);
  }

  function loadNeuron(n, setInspector) {
    index=n;
    weights=model.layers[0].weights.slice(n*4,n*4+4);
    bias=model.layers[0].biases[n];
    mode="relu"; edited=false;
    q("lab-neuron").value=String(n);
    q("lab-activation").value="relu";
    if(setInspector && typeof selectedNeuron !== "undefined" && typeof lastForward !== "undefined" && lastForward) {
      selectedNeuron=n;
      renderInspector(lastForward); renderNetwork(lastForward);
    }
    render();
  }

  function render() {
    const x=inputs();
    const pieces=x.map((v,i)=>v*weights[i]);
    const z=pieces.reduce((sum,n)=>sum+n,bias);
    const a=output(z);
    for(let i=0;i<4;i++) {
      q("lab-weight-"+i).value=String(weights[i]);
      q("lab-weight-value-"+i).textContent=nice(weights[i],2);
      q("lab-weight-"+i).style.setProperty("--slider-color",weights[i]>=0?colors.plus:colors.minus);
    }
    q("lab-bias").value=String(clamp(bias,-4,4));
    q("lab-bias-value").textContent=nice(bias,3);
    q("lab-z").textContent=nice(z,4);
    q("lab-a").textContent=a.toFixed(4);
    q("lab-terms").textContent=pieces.map((v,i)=>nice(x[i],2)+" × "+nice(weights[i],2)).join("  +  ")+"  +  "+nice(bias,2);
    q("lab-mode-caption").textContent=mode === "relu"?"ReLU(z) = max(0, z)":mode === "sigmoid"?"σ(z) = 1 / (1 + e⁻ᶻ)":"tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)";
    q("lab-state").textContent=edited?"EXPERIMENT MODE · NOT APPLIED TO CLASSIFIER":"TRAINED WEIGHTS · REAL MODEL";
    q("lab-state").classList.toggle("edited",edited);
    const svg=q("lab-svg"); svg.replaceChildren();
    const defs=node("defs",{});
    const gradient=node("radialGradient",{id:"lab-core-fill",cx:"35%",cy:"26%",r:"78%"});
    [["0%","#b1fff0"],["31%","#41a6b9"],["76%","#132c52"],["100%","#07152a"]].forEach(([offset,color])=>
      gradient.append(node("stop",{offset,"stop-color":color})));
    const halo=node("radialGradient",{id:"lab-halo"});
    [["0%","#5fffea","stop-opacity",.40],["100%","#5fffea","stop-opacity",0]].forEach(([off,col,_,opacity])=>
      halo.append(node("stop",{offset:off,"stop-color":col,"stop-opacity":opacity})));
    const filter=node("filter",{id:"lab-glow",x:"-100%",y:"-100%",width:"300%",height:"300%"});
    filter.append(node("feGaussianBlur",{stdDeviation:"8"}));
    defs.append(gradient,halo,filter);svg.append(defs);
    const guides=node("g",{"aria-hidden":"true"});
    for(let d=0;d<900;d+=45) guides.append(node("line",{x1:d,y1:0,x2:d,y2:440,stroke:"#bfe9ff","stroke-opacity":".055"}));
    for(let d=0;d<440;d+=44) guides.append(node("line",{x1:0,y1:d,x2:900,y2:d,stroke:"#bfe9ff","stroke-opacity":".055"}));
    svg.append(guides);
    svg.append(node("text",{x:44,y:29,class:"lab-svg-caption"},"01 / INPUT SIGNALS"));
    svg.append(node("text",{x:414,y:29,class:"lab-svg-caption"},"02 / SUM + ACTIVATE"));
    svg.append(node("text",{x:735,y:29,class:"lab-svg-caption"},"03 / OUTPUT"));
    const wires=node("g",{});
    const ys=[88,176,264,352];
    pieces.forEach((v,i)=>{
      const color=weights[i]>=0?colors.plus:colors.minus;
      const strength=clamp(Math.abs(v),0,3);
      const width=1.6+1.5*strength;
      wires.append(node("line",{x1:169,y1:ys[i],x2:386,y2:220,stroke:"#76f8e8","stroke-opacity":".12","stroke-width":width+8,filter:"url(#lab-glow)"}));
      wires.append(node("line",{x1:169,y1:ys[i],x2:386,y2:220,stroke:color,"stroke-opacity":".3"+"" , "stroke-width":width}));
      wires.append(node("line",{x1:169,y1:ys[i],x2:386,y2:220,stroke:color,"stroke-width":2.2,
        "stroke-dasharray":"9 23",class:"lab-signal-path"}));
    });
    wires.append(node("line",{x1:589,y1:220,x2:757,y2:220,stroke:colors.plus,"stroke-width":Math.max(2,Math.min(8,1+a*2)),"stroke-opacity":".63"}));
    wires.append(node("line",{x1:589,y1:220,x2:757,y2:220,stroke:"#f7fffb","stroke-width":"2","stroke-dasharray":"10 22",class:"lab-signal-path"}));
    svg.append(wires);
    ys.forEach((y,i)=>{
      const col=weights[i]>=0?colors.plus:colors.minus;
      svg.append(node("path",{d:`M 52 ${y-16} L 158 ${y-16} L 172 ${y} L 158 ${y+16} L 52 ${y+16} L 38 ${y} Z`,
        fill:"#112b42",stroke:col,"stroke-width":1.5}));
      svg.append(node("text",{x:104,y:y-24,"text-anchor":"middle",class:"lab-svg-label"},names[i]));
      svg.append(node("text",{x:105,y:y+6,"text-anchor":"middle",class:"lab-svg-input"},"x"+(i+1)+" = "+nice(x[i],2)));
      svg.append(node("text",{x:245,y:y+(220-y)*.30-11,"text-anchor":"middle",class:"lab-svg-weight"},"w"+(i+1)+" "+nice(weights[i],2)));
    });
    svg.append(node("circle",{cx:487,cy:220,r:144,fill:"url(#lab-halo)"}));
    svg.append(node("circle",{cx:487,cy:220,r:112,fill:"none",stroke:"#75ffe1","stroke-opacity":".25","stroke-width":"1","stroke-dasharray":"5 9",class:"lab-orbit"}));
    svg.append(node("circle",{cx:487,cy:220,r:91,fill:"url(#lab-core-fill)",stroke:"#92fff0","stroke-width":"2"}));
    svg.append(node("circle",{cx:473,cy:200,r:73,fill:"none",stroke:"#b5f9f4","stroke-opacity":".23","stroke-width":"1.2"}));
    svg.append(node("ellipse",{cx:465,cy:167,rx:38,ry:20,fill:"#e6ffff","fill-opacity":".15",transform:"rotate(-24 465 167)"}));
    svg.append(node("text",{x:487,y:209,"text-anchor":"middle",class:"lab-svg-sigma"},"Σ"));
    svg.append(node("text",{x:487,y:242,"text-anchor":"middle",class:"lab-svg-z"},"z = "+nice(z,2)));
    svg.append(node("path",{d:"M 760 193 L 864 193 L 879 220 L 864 247 L 760 247 L 745 220 Z",fill:"#112b42",stroke:"#67f6dc","stroke-width":"2"}));
    svg.append(node("text",{x:813,y:213,"text-anchor":"middle",class:"lab-svg-label"},"ACTIVATION"));
    svg.append(node("text",{x:813,y:233,"text-anchor":"middle",class:"lab-svg-out"},a.toFixed(3)));
    q("lab-visual-status").textContent="4 live inputs   /   "+(edited?"experimental parameters":"learned parameters")+"   /   "+mode.toUpperCase();
  }

  function pulse() {
    const svg=q("lab-svg");
    svg.classList.remove("lab-firing");
    void svg.getBoundingClientRect();
    svg.classList.add("lab-firing");
  }

  if(typeof model==="undefined"||!model.layers||typeof PRESETS==="undefined") {
    q("lab-state").textContent="MODEL NOT AVAILABLE";
  } else if(document.readyState==="loading") {
    document.addEventListener("DOMContentLoaded",controls,{once:true});
  } else controls();
})();
