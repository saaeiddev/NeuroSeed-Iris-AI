# NeuroSeed AI — Interactive Neural Network Laboratory

**[Open the live demo](https://saaeiddev.github.io/NeuroSeed-Iris-AI/)** · **[Original PyTorch source](https://github.com/saaeiddev/projects/tree/main/NeuroSeed-Iris-AI)**

An interactive, fully browser-based neural network experiment using **the actual weights learned by a PyTorch model trained from scratch** on the Iris dataset. No backend, external AI service, fake answers, or Python runtime is needed in the visitor's browser.

**Created by Amir Saeid Dehghan · 2026**

## How it works

The Iris classifier consumes four measurements in centimeters:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

It standardizes inputs using means/standard deviations from the **training set**, then runs:

```
4 inputs → Linear(4,12) → ReLU → Linear(12,8) → ReLU → Linear(8,3) → Softmax
```

The **191 learned parameters** (weight matrices + biases) and eight normalization constants are exported as 32-bit floating point values, packed into a Base64 constant in `app.js`. JavaScript executes the same forward-pass arithmetic as PyTorch.

The **Neuron Inspector** displays the actual contributions of all four standardized inputs, the selected first-hidden-layer neuron's learned bias, its preactivation weighted sum, and its ReLU activation. Click any neuron in the first hidden layer of the interactive network diagram to explore other calculations.

## Results and accuracy

The original training project records **28 / 30 held-out Iris test examples correctly classified (93.3%)** with a stratified 90 train / 30 validation / 30 test split (seed 42). **Note:** This is a small, educational dataset; scores are model softmax outputs, not guaranteed calibrated confidence. Out-of-distribution measurements may not be meaningful. This is not an image classifier.

Three built-in example measurements are regression-checked against the corresponding Python/PyTorch outputs (within 0.00005 probability error):

| Example | PyTorch expected prediction | Approx. score |
| --- | --- | ---: |
| 5.1, 3.5, 1.4, 0.2 | Setosa | 99.82% |
| 6.0, 2.7, 4.5, 1.5 | Versicolor | 76.04% |
| 6.7, 3.0, 5.2, 2.3 | Virginica | 99.62% |

## Local development

Open `index.html` in a browser, or serve the repository with any local static web server:

```bash
python -m http.server 8000
```

Visit `http://localhost:8000/`. There is **no build step** and no third-party browser dependency.

## GitHub Pages deployment

In **Settings → Pages**, choose **Deploy from a branch**, then **main**, then **/(root)**, and save. GitHub publishes `https://saaeiddev.github.io/NeuroSeed-Iris-AI/`. This repository is a standalone static site, intentionally separate from the original Python repository.

## Files

- `index.html` — document layout, inputs, outputs, network explorer and math inspector
- `style.css` — dark responsive glass-panel interface
- `app.js` — real trained model parameters, forward pass, softmax, SVG network rendering, regression checks
- `favicon.svg` — project icon

## Technical notes

- **Training script and reproducibility**: [Original project source and weights](https://github.com/saaeiddev/projects/tree/main/NeuroSeed-Iris-AI)
- Model architecture: **4 → 12 → 8 → 3**
- Dense layers: **3**; hidden layers: **2**, activation: **ReLU**
- Output: raw logits transformed by **softmax**
- Rendering: HTML + CSS + vanilla JavaScript + inline SVG
- Designed to work on mobile and desktop, with accessible controls and reduced-motion preferences
- Source code licensed under **MIT**; see `LICENSE`.

## Deployment troubleshooting

The entry point is `index.html` in the repository root. GitHub Pages must use **Settings → Pages → Build and deployment → Deploy from a branch → main → /(root)**. If a first-time deployment returns 404, wait for the initial Pages build, then check the GitHub Actions tab for any build or deployment failure. A new commit to `main` can retrigger deployment after Pages has been enabled.
