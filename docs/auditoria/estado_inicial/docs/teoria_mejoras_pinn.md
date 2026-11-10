# Base teórica y mejoras posibles — PINN para cables subterráneos

Documento de referencia para la tesis MIA.  
Estado: activo — se actualiza conforme avanza la investigación.

---

## 1. Problema y formulación matemática

El campo de temperatura $T(x,y)$ alrededor de cables XLPE subterráneos
obedece la ecuación de conducción de calor en estado estacionario con
conductividad térmica variable:

$$\nabla \cdot \left( k(x,y)\, \nabla T \right) = 0 \quad \text{en } \Omega$$

donde $\Omega \subset \mathbb{R}^2$ es el dominio rectangular del suelo y
$k(x,y)$ puede variar espacialmente (capas de suelo, zona PAC de relleno
mejorado, etc.).

**Condiciones de borde:**

| Borde | Tipo | Descripción |
|-------|------|-------------|
| Superior ($y = y_\text{max}$) | Dirichlet | $T = T_\text{amb}$ (temperatura de superficie) |
| Inferior ($y = y_\text{min}$) | Dirichlet | $T = T_\text{prof}$ (temperatura profunda) |
| Laterales ($x = x_\text{min,max}$) | Dirichlet PiecewiseLinear | $T(y)$ con perfil lineal a trozos según profundidad |
| Interfaces de cable | Condición de flujo | $k \partial T / \partial n = Q_\text{lin} / (2\pi r)$ — flujo de calor inyectado |

La ecuación diferencial en forma fuerte se expresa en el residuo de colocación utilizado por la PINN:

$$r(x,y) = \frac{\partial k}{\partial x}\frac{\partial T}{\partial x} + \frac{\partial k}{\partial y}\frac{\partial T}{\partial y} + k\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)$$

Implementado en `pinn_cables/pinn/pde.py` → función `laplace_variable_k()` usando autograd de PyTorch con `create_graph=True`.

---

## 2. Arquitectura PINN actual

### 2.1 Modelo residual (ResidualPINNModel)

La red **no** predice $T$ directamente sino la corrección $u = T - T_\text{bg}$:

$$T(x,y) = \underbrace{T_\text{bg}(x,y)}_{\text{Kennelly analítico}} + \underbrace{u_\theta(x,y)}_{\text{red neuronal}}$$

donde $T_\text{bg}$ es la superposición de Kennelly (solución analítica para
cables en suelo homogéneo). Esto reduce drásticamente la magnitud del
residuo que debe aprender la red.

### 2.2 Backbone MLP tanh

Todos los perfiles de producción usan **MLP puro con activación tanh** (sin Fourier features):

| Perfil | Arquitectura | Parámetros |
|--------|-------------|------------|
| quick | 64×4 MLP tanh | ~17 k |
| research | 128×5 MLP tanh | ~83 k |
| dense (ref.) | 256×6 MLP tanh | ~400 k |

- Entradas: $(x, y)$ normalizadas a $[-1, 1]$
- Salida: corrección escalar $u(x,y)$ (temperatura residual en K)

> **Nota sobre Fourier features:** se probaron con $\sigma = 3.0$ y $m = 64$ en el
> perfil research, pero introdujeron una sobreestimación de +6.7 K en la zona PAC
> (`run_research_pac.py`), probablemente por amplificación de altas frecuencias
> espurias en la interfaz sigmoide. Se eliminaron del perfil de producción.
> Referencia teórica: Tancik et al. (2020).

### 2.3 Función de pérdida

$$\mathcal{L} = w_\text{pde}\,\mathcal{L}_\text{pde} + w_\text{bc}\,\mathcal{L}_\text{bc}$$

$$\mathcal{L}_\text{pde} = \frac{1}{N_\text{int}}\sum_{i=1}^{N_\text{int}} r(x_i, y_i)^2, \qquad \mathcal{L}_\text{bc} = \frac{1}{N_\text{bc}}\sum_{j=1}^{N_\text{bc}} (T_\theta(x_j) - T_\text{ref,j})^2$$

Con $w_\text{pde} = 1.0$ y $w_\text{bc} = 50.0$ (fijos).

### 2.4 Ciclo de entrenamiento (train_custom.py)

1. **Pre-entrenamiento** (800 pasos Adam): aprende $u \approx 0$ cerca de cables y fronteras.
2. **Curriculum warmup** (30% primeros Adam steps): $k(x,y) = k_\text{soil}$ constante — estabiliza la red antes de activar la discontinuidad PAC.
3. **Adam** ($5\,000$–$10\,000$ pasos, lr$= 5 \times 10^{-4}$): exploración global.
4. **L-BFGS** (500–1000 pases, muestra fija): refinamiento local cuadrático.
5. **Adam2** (500 pasos finos, lr$= 10^{-5}$): corrección post-LBFGS activada en perfil research.

---

## 3. Diagnóstico de fallos en casos multicapa

### 3.1 Causa principal: gradientes desbalanceados (Wang & Perdikaris 2020)

**Referencia:** S.Wang, X.Yu & P.Perdikaris, *"When and why PINNs fail to train: A neural tangent kernel perspective"*, Journal of Computational Physics (2022). Preprint: `arXiv:2007.14527`

**Referencia original de patologías:** S.Wang & P.Perdikaris, *"Understanding and mitigating gradient pathologies in physics-informed neural networks"*, SIAM J. Sci. Comput. (2021). Preprint: `arXiv:2001.04536` 

Cuando la pérdida tiene múltiples términos ($\mathcal{L}_{PDE}$ y $\mathcal{L}_{BC}$), los gradientes retropropagados tienen magnitudes muy distintas. Si:

$$\|\nabla_\theta \mathcal{L}_\text{bc}\| \gg \|\nabla_\theta \mathcal{L}_\text{pde}\|$$

la red satisface las condiciones de borde pero acumula residuo PDE alto en interfaces de capas. Con $w_\text{bc} = 50$, esto se agrava en redes grandes (128×5) donde los gradientes son intrínsecamente mayores.

**Solución propuesta:** Pesos adaptativos mediante estadísticas de gradiente:

$$\hat{\lambda}_\text{bc}(t) = \frac{\max_\theta |\nabla_\theta \mathcal{L}_\text{pde}(t)|}{\text{mean}_\theta |\nabla_\theta \mathcal{L}_\text{bc}(t)|}$$

Reportan mejoras de 50–100× en precisión.

### 3.2 Sesgo espectral (Spectral bias)

Los MLP con activación $\tanh$ aprenden primero frecuencias espaciales bajas
(componente suave de $T$) y luego —lentamente— las frecuencias altas
asociadas a discontinuidades de $k(y)$.

**Referencia:** Rahaman et al. (2019), *"On the spectral bias of neural networks"*, ICML 2019. `arXiv:1806.08734`

**Solución ya implementada:** Fourier feature mapping con frecuencia alta ($\sigma=10$) mejora la representación de gradients abruptos.

### 3.3 Paisaje de pérdida mal condicionado (Krishnapriyan et al. 2021)

**Referencia:** A.Krishnapriyan, A.Gholami, S.Zhe, R.Kirby & M.Mahoney, *"Characterizing possible failure modes in physics-informed neural networks"*, NeurIPS 2021. arXiv: `arXiv:2109.01050`

Con múltiples discontinuidades de $k$ (3 interfaces de capas de suelo + zona PAC sigmoide), el Hessiano de $\mathcal{L}_\text{pde}$ en las interfaces es muy mal condicionado. Más pasos Adam no ayudan —sobrepasan el mínimo. Proponen:

- **Curriculum regularization**: comenzar con PDE simplificada e ir aumentando complejidad — parcialmente implementado con el warmup curricuular.
- **Sequence-to-sequence decomposition**: subdividir el dominio espacial.

### 3.4 Fracasos de propagación: distribución fija de puntos de colocación

**Referencia:** A.Daw, J.Bu, S.Wang, P.Perdikaris & A.Karpatne, *"Mitigating Propagation Failures in Physics-informed Neural Networks using Retain-Resample-Release (R3) Sampling"*, ICML 2023. `arXiv:2207.02338`

Los puntos de colocación fijos con sobremuestreo en interfaces suelo/PAC no garantizan que el residuo PDE total sea mínimo. En regiones de alto residuo que no coinciden exactamente con las interfaces muestreadas, el optimizador no ve señal de corrección.

**Solución R3 Sampling:** Cada $N$ pasos Adam:
1. **Retain**: conservar $f_r N_\text{int}$ puntos de mayor residuo
2. **Resample**: redistribuir $f_r N_\text{int}$ puntos nuevos muestrando proporcional a $r^2(x,y)$
3. **Release**: liberar los $f_r N_\text{int}$ de menor residuo

### 3.5 Preacondicionamiento del operador (De Ryck et al. 2023)

**Referencia:** T.De Ryck, S.Lanthaler & S.Mishra, *"An operator preconditioning perspective on training in physics-informed machine learning"*, ICLR 2024. `arXiv:2310.05801`

El operador $-\nabla\cdot(k\nabla\cdot)$ tiene un cuadrado hermitiano cuyo condicionamiento crece polinomialmente con la frecuencia. Más capas de suelo y zona PAC → más valores propios extremos → entrenamiento más lento o infactible.

---

## 4. Mejoras concretas implementadas / por implementar

### 4.1 ✅ Implementado: formulación residual con Kennelly

La red aprende $u = T - T_{bg}$ donde $T_{bg}$ es la superposición de Kennelly.
Reduce el problema de aprendizaje a una perturbación pequeña, eliminando mínimos
locales espurios.

### 4.2 ✅ Implementado: lr reducida en perfil research

El archivo `solver_params_research.csv` usa `lr = 1.0e-3` (igual que quick).
Una red 128×5 genera gradientes más grandes que 64×4 — la misma lr es aceptable
gracias a la formulación residual que amortigua la magnitud de las salidas.

### 4.3 ✅ Implementado: Adam2 fine-tuning post-L-BFGS

**Motivación:** L-BFGS puede terminar en un mínimo local con pérdida baja pero
con artefactos locales cerca de interfaces. 500–1000 pasos adicionales de Adam
con lr$= 10^{-5}$ (re-muestreo de puntos de colocación) eliminan artefactos locales.

**Activado en:** `solver_params_research.csv` → `adam2_steps = 500`.

### 4.4 ✅ Implementado: curriculum warmup con k homogéneo

El 30 % inicial de los pasos Adam se entrena con $k(x,y) = k_\text{soil}$ constante.
Estabiliza la red antes de activar la discontinuidad PAC/multicapa.

### 4.5 ✅ Implementado: estudio de convergencia en tamaño de red

El perfil `dense` (256×6, ~400 k parámetros, 22 000 pasos) actúa como
**pseudo-referencia de campo completo** análoga a la malla fina en FEM. Permite
calcular $\text{RMSE}_\text{zona}$ del campo $T(x,y)$ alrededor de los cables
para quick y research, superando la limitación del punto único $T_\max$ del paper.

Se corre con:
```bash
python examples/kim_2024_154kv_bedding/run_multilayer_dense.py
```

### 4.6 Propuesta futura: pesos adaptativos (Wang & Perdikaris 2021)

Implementar en `pinn_cables/pinn/train_custom.py` dentro del bucle Adam:

```python
# Cada `weight_update_every` pasos Adams, recalcular w_bc
if step % weight_update_every == 0:
    # Forward pass con ambos términos para obtener gradientes separados
    loss_pde_only = w_pde * loss_pde
    loss_bc_only = w_bc * loss_bc
    grads_pde = torch.autograd.grad(loss_pde_only, model.parameters(), retain_graph=True)
    grads_bc  = torch.autograd.grad(loss_bc_only,  model.parameters())
    max_grad_pde  = max(g.abs().max() for g in grads_pde if g is not None)
    mean_grad_bc  = sum(g.abs().mean() for g in grads_bc if g is not None) / n_params
    w_bc = float(max_grad_pde / (mean_grad_bc + 1e-8))
```

### 4.7 Propuesta futura: R3 Sampling adaptativo

Implementar en `pinn_cables/pinn/train_custom.py` como alternativa al muestreo fijo:

```python
# Cada resample_every pasos, actualizar puntos de colocación
if resample_every > 0 and step % resample_every == 0:
    with torch.no_grad():
        residuals = pde_residual_steady(model(norm_fn(xy_soil)), xy_soil, k_fn, 0.0)
        weights = residuals.squeeze().abs() + 1e-6
        weights /= weights.sum()
        idx   = torch.multinomial(weights, n_int, replacement=True)
        xy_soil = xy_soil[idx]
```

### 4.8 ✅ Implementado: módulo FNO (Fourier Neural Operator)

Ubicado en `pinn_cables/fno/`. Ver sección 5 para detalles.

---

## 5. Arquitecturas alternativas

### 5.1 Fourier Neural Operator (FNO)

**Referencia principal:** Z.Li, N.Kovachki, K.Azizzadenesheli, B.Liu, K.Bhattacharya, A.Stuart & A.Anandkumar, *"Fourier Neural Operator for Parametric Partial Differential Equations"*, ICLR 2021. `arXiv:2010.08895`

FNO aprende el **operador solución** $\mathcal{G}: (k, Q, \text{BC}) \mapsto T$ directamente, sin resolver la PDE para cada instancia. Ventajas:

- Entrena una vez con muchos escenarios → inferencia en $< 1$ ms
- Manejo nativo de multi-escala (convolución en espacio de Fourier)
- Resolución-invariante (zero-shot super-resolution)

**Arquitectura FNO 2D:**

```
Input  : (batch, C_in, N_x, N_y)   ← [k(x,y), Q_src(x,y), BC_mask(x,y)]
         ↓
P      : pointwise lifting MLP       (C_in → d_model)
         ↓
[FNO layer × L]:
  SpectralConv2d  : multiply K modes in Fourier space (learnable W)
  + Conv1×1       : pointwise skip connection
  + GELU activation
         ↓
Q      : pointwise projection MLP    (d_model → 1)
         ↓
Output : (batch, 1, N_x, N_y)       ← T(x,y)
```

**Capa SpectralConv2d:**

$$v_{l+1}(x) = \sigma\!\left(\mathcal{F}^{-1}\!\left[R_l \cdot \mathcal{F}[v_l]\right](x) + W_l v_l(x)\right)$$

donde $R_l \in \mathbb{C}^{d \times d \times k_1 \times k_2}$ contiene $k_1 k_2$ modos de Fourier aprendibles.

**Implementado en:** `pinn_cables/fno/model.py` → clases `SpectralConv2d` y `CableFNO2d`.

### 5.2 Deep Operator Network (DeepONet)

**Referencia:** L.Lu, P.Jin, G.Pang, Z.Zhang & G.E.Karniadakis, *"Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators"*, Nature Machine Intelligence (2021). Preprint: `arXiv:1910.03193`

Dos sub-redes:
- **Branch net**: codifica la función de entrada $f$ (p.ej. distribución $k$) evaluada en $m$ sensores fijos.
- **Trunk net**: codifica las coordenadas de salida $(x,y)$.

$$\mathcal{G}(f)(y) = \sum_{k=1}^p b_k(f(x_1), \ldots, f(x_m)) \cdot t_k(y)$$

Ventaja: convergencia exponencial para operadores suaves. Ideal si el mapeo $k \mapsto T$ es regular.

### 5.3 Redes inspiradas en visión artificial

#### U-Net para campos de temperatura

**Referencia:** Ronneberger, Fischer & Brox (2015), *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, MICCAI 2015. `arXiv:1505.04597`

La analogía es directa:
- **Dominio** $\Omega$ $\leftrightarrow$ imagen 2D
- **k(x,y)** rasterizado $\leftrightarrow$ canal de entrada
- **T(x,y)** rasterizado $\leftrightarrow$ imagen de salida
- **Skip connections** del encoder al decoder $\leftrightarrow$ propagación de información de BCs hacia el interior

U-Net maneja naturalmente la diferencia de escala entre el dominio grande (91×46 m) y las estructuras finas de los cables (radio ~cm) mediante el pooling/unpooling jerárquico.

#### Vision Transformer (ViT) para PDEs

**Referencia reciente:** Herde et al. (2024), *"Poseidon: Efficient Foundation Models for PDEs"*, NeurIPS 2024. `arXiv:2405.19101`

Los transformers aplican atención global entre parches del dominio — equivalente a integrar condiciones de borde en todo el campo en un solo paso (análogo al método de contorno integral).

### 5.4 Kolmogorov-Arnold Networks (KAN)

**Referencia:** Z.Liu et al., *"KAN: Kolmogorov-Arnold Networks"*, ICLR 2025. `arXiv:2404.19756`

En lugar de $\text{MLP}$ con activaciones fijas en nodos, KAN usa **splines aprendibles en aristas**. Para PDE solving:
- Menor error con las mismas cantidades de parámetros
- Más interpretables (se pueden extraer relaciones físicas de los splines)
- Más lentos de entrenar actualmente (~5-10× vs MLP)

### 5.5 gPINN (Gradient-enhanced PINN)

**Referencia:** J.Yu, L.Lu, X.Meng & G.E.Karniadakis, *"Gradient-enhanced physics-informed neural networks for forward and inverse problems"*, Computer Methods in Applied Mechanics and Engineering (2022). `arXiv:2111.02801`

Añade el gradiente del residuo PDE como término adicional de pérdida:

$$\mathcal{L}_\text{gPINN} = \mathcal{L}_\text{PDE} + w_g \left\| \nabla_{(x,y)} r(x,y) \right\|^2$$

Esto obliga a la red a satisfacer **también** las condiciones de continuidad del flujo en las interfaces ($\partial r / \partial n = 0$), lo que es especialmente útil para el caso multicapa.

---

## 6. Comparación de enfoques

| Método | Ventaja principal | Limitación | Paperbenchmark | Relevancia para cables |
|--------|------------------|-----------|----------------|------------------------|
| **PINN actual** | Sin datos de entrenamiento, física exacta | Convergencia inconsistente en k-variable multicapa | Wang (2020) | Alta — implementado |
| **PINN + pesos adaptativos** | 50-100× mejora en precisión | Coste extra por forward pass adicional | Wang & Perdikaris arXiv:2001.04536 | Muy alta — siguiente paso |
| **PINN + R3 Sampling** | Foco de colocación en zonas de alto residuo | Coste de computar residuos periódicamente | Daw et al. arXiv:2207.02338 | Alta |
| **gPINN** | Mejor en discontinuidades de k | 2–3× más lento por pass de gradientes | Yu et al. arXiv:2111.02801 | Muy alta para multicapa |
| **FNO** | Inferencia en <1 ms, operator learning | Necesita training data (variantes) | Li et al. arXiv:2010.08895 | Alta para multi-escenario |
| **DeepONet** | Convergencia exponencial | Datos de entrenamiento requeridos | Lu et al. arXiv:1910.03193 | Media-alta |
| **U-Net** | Multi-escala, skip connections | Discretización fija de grid | Ronneberger arXiv:1505.04597 | Media — exploratoria |
| **KAN** | Interpretable, escalado neural superior | Entrenamiento lento en 2026 | Liu et al. arXiv:2404.19756 | Baja-media (experimental) |

---

## 7. Roadmap de mejoras para la tesis

```
Prioridad 1 (implementado):
  ✅ Formulación residual T = T_bg + u_theta (Kennelly)
  ✅ Curriculum warmup 30% pasos Adam (k homogéneo → k variable)
  ✅ Adam2 post-L-BFGS (research: 500 pasos, dense: 1000 pasos)
  ✅ Perfil denso 256x6 (~400k params) como referencia de convergencia
  ✅ eval_all.py con RMSE del campo T(x,y) vs solución densa
  ✅ Módulo FNO en pinn_cables/fno/

Prioridad 2 (corto plazo):
  □ Pesos adaptativos de pérdida (Wang & Perdikaris 2021)
  □ R3 Sampling como alternativa al muestreo fijo
  
  ⚠️ run_research_pac.py con perfil research sigue sobreestimando (+6.7 K).
     Causa probable: gradientes desbalanceados en zona PAC sigmoide con red 128x5.
     Pesos adaptativos son la mejora más probable para resolver este caso.

Prioridad 3 (mediano plazo):
  □ gPINN: añadir L_grad = MSE(dr/dx, 0) + MSE(dr/dy, 0)
  □ FNO: generación de dataset paramétrico + entrenamiento completo
  
  Especialmente relevante: gPINN para el caso multicapa, donde las
  interfaces de k generan discontinuidades en el gradiente de T.

Prioridad 4 (contribución diferenciadora):
  □ Comparación PINN vs FNO en benchmarks Aras (2005) + Kim (2024)
  □ Publicable como artículo de journal (IEEE Trans. Power Delivery)
```

---

## 8. Referencias bibliográficas completas

1. **Raissi, Perdikaris & Karniadakis (2019)** — PINNs fundacional:  
   M.Raissi, P.Perdikaris & G.E.Karniadakis,  
   *"Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear PDEs"*,  
   Journal of Computational Physics 378:686–707, 2019.  
   https://doi.org/10.1016/j.jcp.2018.10.045

2. **Aras, Oysu & Yilmaz (2005)** — Benchmark cables 154 kV:  
   F.Aras, C.Oysu & G.Yilmaz,  
   *"An Assessment of the Methods for Calculating Ampacity of Underground Power Cables"*,  
   Electric Power Components and Systems 33(12):1385–1402, 2005.  
   https://doi.org/10.1080/15325000590964969

3. **Kim, Cho & Choi (2024)** — Benchmark PAC bedding 154 kV:  
   J.Kim, S.Cho & S.Choi,  
   *"Thermal Analysis of 154 kV Underground Cable System with PAC Bedding Using COMSOL Multiphysics"*,  
   *(En revisión / preprint, 2024). Datos FEM: I = 865 A, suelo arena 77.6 °C, PAC 70.6 °C.)*

4. **Wang & Perdikaris (2021)** — Gradient pathologies in PINNs:  
   S.Wang & P.Perdikaris, *"Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks"*,  
   SIAM Journal on Scientific Computing 43(5):A3055-A3081, 2021.  
   Preprint: https://arxiv.org/abs/2001.04536

5. **Wang, Yu & Perdikaris (2022)** — NTK perspective:  
   S.Wang, X.Yu & P.Perdikaris, *"When and why PINNs fail to train: A neural tangent kernel perspective"*,  
   Journal of Computational Physics 449:110768, 2022.  
   Preprint: https://arxiv.org/abs/2007.14527

6. **Krishnapriyan et al. (NeurIPS 2021)** — Failure modes in PINNs:  
   A.Krishnapriyan, A.Gholami, S.Zhe, R.Kirby & M.Mahoney,  
   *"Characterizing Possible Failure Modes in Physics-Informed Neural Networks"*,  
   NeurIPS 2021. https://arxiv.org/abs/2109.01050

7. **Daw et al. (ICML 2023)** — R3 Sampling:  
   A.Daw, J.Bu, S.Wang, P.Perdikaris & A.Karpatne,  
   *"Mitigating Propagation Failures in Physics-Informed Neural Networks using Retain-Resample-Release (R3) Sampling"*,  
   ICML 2023. https://arxiv.org/abs/2207.02338

8. **De Ryck et al. (ICLR 2024)** — Operator preconditioning:  
   T.De Ryck, S.Lanthaler & S.Mishra,  
   *"An Operator Preconditioning Perspective on Training in Physics-Informed Machine Learning"*,  
   ICLR 2024. https://arxiv.org/abs/2310.05801

9. **Yu et al. (2022)** — gPINNs:  
   J.Yu, L.Lu, X.Meng & G.E.Karniadakis,  
   *"Gradient-Enhanced Physics-Informed Neural Networks for Forward and Inverse Problems"*,  
   Computer Methods in Applied Mechanics and Engineering 393:114823, 2022.  
   https://arxiv.org/abs/2111.02801

10. **Li et al. (ICLR 2021)** — Fourier Neural Operator (FNO):  
    Z.Li, N.Kovachki, K.Azizzadenesheli, B.Liu, K.Bhattacharya, A.Stuart & A.Anandkumar,  
    *"Fourier Neural Operator for Parametric Partial Differential Equations"*,  
    ICLR 2021. https://arxiv.org/abs/2010.08895

11. **Lu et al. (2021)** — DeepONet:  
    L.Lu, P.Jin, G.Pang, Z.Zhang & G.E.Karniadakis,  
    *"Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators"*,  
    Nature Machine Intelligence 3:218-229, 2021.  
    https://arxiv.org/abs/1910.03193

12. **Wang et al. (2024)** — Causal PINNs:  
    S.Wang, S.Sankaran & P.Perdikaris,  
    *"Respecting Causality for Training Physics-Informed Neural Networks"*,  
    Computer Methods in Applied Mechanics and Engineering 421:116813, 2024.  
    https://arxiv.org/abs/2203.07404

13. **Tancik et al. (NeurIPS 2020)** — Fourier Features:  
    M.Tancik, P.Srinivasan, B.Mildenhall, S.Fridovich-Keil, N.Raghavan, U.Singhal,  
    R.Ramamoorthi, J.Barron & R.Ng,  
    *"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"*,  
    NeurIPS 2020. https://arxiv.org/abs/2006.10739

14. **Liu et al. (ICLR 2025)** — KAN:  
    Z.Liu, Y.Wang, S.Vaidya, F.Ruehle, J.Halverson, M.Soljačić, T.Y.Hou & M.Tegmark,  
    *"KAN: Kolmogorov-Arnold Networks"*,  
    ICLR 2025. https://arxiv.org/abs/2404.19756

15. **Ronneberger et al. (MICCAI 2015)** — U-Net:  
    O.Ronneberger, P.Fischer & T.Brox,  
    *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*,  
    MICCAI 2015. https://arxiv.org/abs/1505.04597

16. **Rahaman et al. (ICML 2019)** — Spectral bias:  
    N.Rahaman, A.Baratin, D.Arpit, F.Draxler, M.Lin, F.Hamprecht, Y.Bengio & A.Courville,  
    *"On the Spectral Bias of Neural Networks"*,  
    ICML 2019. https://arxiv.org/abs/1806.08734

17. **IEC 60287-1-1 (2023)** — Cálculo de capacidad de corriente:  
    International Electrotechnical Commission,  
    *"Electric cables — Calculation of the current rating — Part 1-1"*,  
    IEC 60287-1-1:2023.

18. **Carslaw & Jaeger (1959)** — Conducción de calor:  
    H.S.Carslaw & J.C.Jaeger,  
    *"Conduction of Heat in Solids"* (2nd ed.), Oxford University Press, 1959.

---

*Documento activo de investigación — actualizar conforme avanza la tesis.*
