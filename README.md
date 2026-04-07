# 🌆 Urban Heat Surrogate Explorer

A scientific, interactive dashboard for analyzing urban microclimate using Gaussian Process Regression.

This project enables exploration, interpretation, and optimization of how urban design parameters affect **peak canyon temperature**.

---

## 🚀 Features

- 📊 Gaussian Process surrogate modeling
- 🌡️ Temperature prediction (°C) with uncertainty
- 🏙️ Interactive city visualization (urban canyon)
- ⚙️ Parameter sensitivity exploration
- 📉 Surrogate-based optimization
- 🤖 Built-in **App Assistant** (chatbot)
  - Answers based on live model state (no hallucination)
  - Optional Hugging Face explanation layer

---

## 🧠 Methodology

We use a Gaussian Process Regressor:

$begin:math:display$
\\hat\{T\}\(x\) \= \\mu\(x\)\, \\quad \\sigma\(x\)
$end:math:display$

Optimization is formulated as:

$begin:math:display$
x\^\* \= \\arg\\min\_x \\hat\{T\}\(x\)
$end:math:display$

Where:
- $begin:math:text$ x $end:math:text$ = urban parameters (geometry, vegetation, albedo)
- $begin:math:text$ \\hat\{T\}\(x\) $end:math:text$ = predicted peak canyon temperature

---

## 🏗️ Parameters

| Parameter | Symbol | Description |
|----------|--------|------------|
| Wall albedo | α_w | Reflectivity of building walls |
| Canyon height | H_c | Height of buildings |
| Canyon width | W_c | Distance between buildings |
| Roof width | W_r | Width of rooftop |
| Ground vegetation | f_g | Vegetation fraction on ground |
| Roof vegetation | f_r | Vegetation fraction on roofs |
| Tree height | H_t | Height of trees |
| Tree radius | R_t | Tree crown radius |

---

## 🖥️ How to Run

### 1. Clone the repo

```bash
git clone https://github.com/gagankaushikmanyam/CityPlanning.git
cd CityPlanning
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app_wrapper.py
```

---

## 🤖 App Assistant

The assistant:
- Uses **live app state (no hallucination)**
- Can explain:
  - model metrics (R², RMSE)
  - parameter values
  - optimization results
  - temperature predictions

Optional:
- Uses Hugging Face (`flan-t5`) for improved explanations

---

## 📸 Preview

_Add screenshots here later_

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Plotly
- Hugging Face Transformers

---

## 📌 Future Work

- RAG-based paper integration
- Multi-objective optimization
- Real-world city datasets
- Deployment (Streamlit Cloud / Docker)

---

## 👤 Author

**Gagan Kaushik Manyam**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!