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

$$
\hat{T}(x) = \mu(x), \quad \sigma(x)
$$

Optimization is formulated as:

$$
x^* = \arg\min_x \hat{T}(x)
$$

Where:
- $x$ = urban parameters (geometry, vegetation, albedo)
- $\hat{T}(x)$ = predicted peak canyon temperature

---

## 🏗️ Parameters

| Parameter | Symbol | Description |
|----------|--------|------------|
| Wall albedo | $\alpha_w$ | Reflectivity of building walls |
| Canyon height | $H_c$ | Height of buildings |
| Canyon width | $W_c$ | Distance between buildings |
| Roof width | $W_r$ | Width of rooftop |
| Ground vegetation | $f_g$ | Vegetation fraction on ground |
| Roof vegetation | $f_r$ | Vegetation fraction on roofs |
| Tree height | $H_t$ | Height of trees |
| Tree radius | $R_t$ | Tree crown radius |

---

## 🐳 Run with Docker (Recommended)

### 1. Build the image

```bash
docker build -t respace-app .
```

### 2. Run the container

```bash
docker run -p 8501:8501 respace-app
```

### 3. Open the app

```
http://localhost:8501
```

---

## 🧑‍💻 Run with VS Code DevContainer

1. Install **Dev Containers extension** in VS Code  
2. Open the project folder  
3. Press:

```
Cmd/Ctrl + Shift + P → Dev Containers: Reopen in Container
```

The app will build and start automatically.

---

## 🤖 App Assistant

The assistant:
- Uses **live app state (no hallucination)**
- Can explain:
  - model metrics ($R^2$, RMSE)
  - parameter values
  - optimization results
  - temperature predictions

Optional:
- Uses Hugging Face (`flan-t5`) for improved explanations

---

## 📸 Preview

_Add screenshots here_

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Plotly
- Hugging Face Transformers
- Docker

---

## 📌 Future Work

- RAG-based paper integration
- Multi-objective optimization
- Real-world city datasets
- Cloud deployment

---

## 👤 Author

**Gagan Kaushik Manyam**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!