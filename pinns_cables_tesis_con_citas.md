# Investigación propuesta: PINNs para conducción de calor 2D en cables subterráneos con coeficientes espacialmente variables

> Documento de trabajo (Markdown) para desarrollar en tu entorno.  
> Enfoque: modelación **estacionaria** y **transitoria** del campo térmico en sección 2D de cables subterráneos (capas concéntricas) + zanja/terreno, usando **Physics-Informed Neural Networks (PINNs)**.

---

## 0. Punto de partida y adaptación

En ampacidad de cables subterráneos, es común partir de métodos analíticos “tipo estándar” (p.ej., IEC 60287) y contrastarlos con modelos numéricos (FEM) cuando aparecen **geometrías complejas** y/o **heterogeneidad térmica del medio** (IEC, 2023; Aras et al., 2005). En un enfoque PINN, el objetivo es resolver el mismo problema físico (PDE + BC/IC + acoples), pero con una formulación que pueda:
- absorber con naturalidad **k(x,y)** variable en suelo (estratos/humedad),
- manejar **discontinuidades por capas** (conductor/XLPE/pantalla/cubierta),
- escalar a variantes de instalación (zanja, múltiples cables, interferencias térmicas),
- y cubrir **estado estacionario** y **transitorio** (ciclos de carga / sobrecargas).

Como antecedente técnico, la comparación entre métodos normativos y métodos numéricos para casos subterráneos con geometrías realistas se encuentra bien documentada, incluyendo limitaciones de los modelos simplificados y el uso de FEM como referencia (Aras et al., 2005). Para el marco de PINNs y su formalización (resolución de PDEs por redes con información física), se toma como base el enfoque de *physics-informed neural networks* (Raissi et al., 2019).

---

## 1. Planteamiento del problema

### 1.1. Problema técnico
La **ampacidad** está limitada por la temperatura máxima admisible del cable (y su entorno). En operación real, la temperatura depende de:
- geometría de instalación (zanja, profundidad, separación),
- propiedades térmicas del suelo y su variación espacial/temporal,
- pérdidas internas (Joule, dieléctricas, pantalla),
- condiciones de contorno (superficie/ambiente, frontera “lejana”, simetrías).

Los métodos analíticos se degradan cuando la instalación se aleja de supuestos simplificados; en esos casos se recurre a FEM como referencia numérica (Aras et al., 2005). De modo complementario, formulaciones clásicas históricas de cálculo de carga/temperatura de sistemas de cables (precursoras de enfoques normativos posteriores) se remontan a contribuciones como Neher y McGrath (1957).

### 1.2. Pregunta de investigación (propuesta)
**¿En qué medida un solucionador PINN 2D (estacionario y transitorio) permite estimar el campo de temperatura y la ampacidad de cables subterráneos multicapa, cuando el medio presenta coeficientes térmicos espacialmente variables, manteniendo precisión comparable a FEM y reduciendo esfuerzo de modelado para múltiples escenarios?**

### 1.3. Hipótesis de trabajo (propuesta)
Un PINN que incorpore la física (PDE + BC/IC + continuidad en interfaces) puede aproximar con buena precisión el campo térmico 2D de la sección del cable y su entorno, incluso con **k(x,y)** variable y discontinuidades por capas, y puede generalizar a escenarios paramétricos (profundidad, separaciones, k_suelo, carga) con costo incremental bajo (Raissi et al., 2019).

---

## 2. Objetivos

### 2.1. Objetivo general
Desarrollar y validar un solucionador basado en **PINNs** para conducción de calor 2D en cables subterráneos con capas concéntricas, considerando casos **estacionarios y transitorios**, y **coeficientes espacialmente variables**, aplicado a estimación de temperatura y ampacidad.

### 2.2. Objetivos específicos
1. Formular el modelo físico (PDE) y condiciones de contorno relevantes (zanja/terreno, límites, simetrías, fuentes) para el cálculo térmico asociado a ampacidad (IEC, 2023; Aras et al., 2005).
2. Implementar un PINN en PyTorch para:
   - estacionario (∂T/∂t = 0),
   - transitorio (∂T/∂t ≠ 0),
   - k(x,y) y (ρc)(x,y) variables (por región o continuo en suelo).
3. Implementar un **sampler de geometría** robusto (capas concéntricas + zanja) para collocation points, interfaces y bordes.
4. Validar con:
   - benchmarks analíticos (Carslaw & Jaeger, 1959),
   - **manufactured solutions (MMS)** para k variable,
   - comparación con FEM (como referencia) en casos representativos (Aras et al., 2005).
5. Aplicar a un caso representativo (XLPE) y derivar ampacidad por criterio T_límite (IEC, 2023).

---

## 3. Modelo matemático

### 3.1. Ecuación de conducción (2D, coeficientes variables)
Modelo general (forma divergente) en 2D:

\[
\frac{\partial}{\partial x}\left(k \frac{\partial T}{\partial x}\right)+
\frac{\partial}{\partial y}\left(k \frac{\partial T}{\partial y}\right)+Q
= \rho c \frac{\partial T}{\partial t}
\]

Esta forma es consistente con formulaciones estándar de conducción y con el planteo aplicado al entorno de cables subterráneos (Aras et al., 2005). En particular:
- k puede ser constante por capa (discontinuo entre capas) y **variable en suelo** k(x,y).
- Q representa fuentes volumétricas (pérdidas) o puede reemplazarse por flujos impuestos.

**Casos:**
- **Estacionario**: ∂T/∂t = 0
- **Transitorio**: ∂T/∂t ≠ 0

### 3.2. Condiciones de contorno (plantilla)
Según el dominio:
- Superficie del terreno: Dirichlet (T = T_amb) o Robin (convección) (IEC, 2023).
- Laterales/fondo: Dirichlet en frontera “lejana” (aproximación a medio infinito) o Neumann ~ 0 si se justifica simetría/aislamiento (Aras et al., 2005).
- Simetrías: Neumann (flujo normal 0) en ejes de simetría (Aras et al., 2005).

### 3.3. Interfaces entre capas
En cada interfaz:
- continuidad de temperatura: \(T_i = T_j\)
- continuidad de flujo normal: \(-k_i \nabla T_i \cdot \mathbf{n} = -k_j \nabla T_j \cdot \mathbf{n}\)

En conducción en sólidos, este tratamiento de acople por continuidad es estándar en problemas con materiales por capas (Carslaw & Jaeger, 1959).

---

## 4. Enfoque PINN

### 4.1. PINN clásico
Aproximar \(T_\theta(x,y,t)\) con una red y entrenar minimizando una función de pérdida que combine:
- residuo PDE en puntos internos (collocation),
- error de BC,
- error de IC (transitorio),
- error de interfaces (T y flujo).

La formulación general y el uso de autodiferenciación para construir residuos se apoya en el marco PINN establecido por Raissi et al. (2019).

### 4.2. Discontinuidades: 2 estrategias prácticas
1) **PINN global + pérdida de interfaces** (y muestreo reforzado cerca de interfaces) (Raissi et al., 2019).  
2) **Domain decomposition / XPINNs**: una red por subdominio con condiciones de acople en interfaces, particularmente útil con discontinuidades fuertes (Jagtap & Karniadakis, 2020).

---

## 5. Validación y benchmarks (recomendado)

### 5.1. Benchmarks analíticos “mínimos”
- Laplace/Poisson en rectángulo con Dirichlet: solución por series (clásico en conducción).
- Conducción radial estacionaria en cilindro/anillo (multicapa ideal): perfiles logarítmicos y acople en interfaces, útil para “modo cable” sin zanja (Carslaw & Jaeger, 1959).

### 5.2. MMS (Manufactured Solutions) para k(x,y) variable
Elegir:
- \(T^*(x,y)\) suave (senos/cosenos/polinomios),
- \(k(x,y)\) suave,
- derivar \(Q(x,y)\) para que la PDE se cumpla exactamente.

Esto valida el caso “coeficientes espacialmente variables” sin depender de FEM, y permite detectar sesgos numéricos del entrenamiento (Raissi et al., 2019).

### 5.3. Benchmark aplicado (cables)
Reproducir al menos un caso de referencia de cable XLPE (geometría de capas + propiedades + comparación IEC/FEM/experimento), y si es posible añadir:
- tres cables (mutua interferencia térmica),
- fuente térmica externa o condición de contorno perturbada (Aras et al., 2005).

---

## 6. Arquitectura Python (PyTorch) para solver PINN (estacionario + transitorio)

### 6.1. Estructura sugerida de proyecto
```
pinn_cables/
  configs/
    base.yaml
    steady.yaml
    transient.yaml
  pinn/
    model.py          # MLP / Fourier features
    pde.py            # operador PDE y residuales
    losses.py         # pérdidas: PDE, BC, IC, interfaces
    train.py          # loops Adam + LBFGS
    utils.py          # seeds, normalización, logging
  geom/
    sampler.py        # muestreo robusto (incluido abajo)
  materials/
    props.py          # k(x,y), rho*c, Q por región
  scripts/
    run_steady.py
    run_transient.py
  post/
    eval.py           # métricas y comparación con FEM/MMS
    plots.py
```

### 6.2. Convenciones recomendadas
- Unidades SI (m, s, K, W).
- Normalizar x,y,t a [-1,1] (mejora estabilidad numérica y entrenamiento) (Raissi et al., 2019).
- Entrenamiento: Adam (exploración) → LBFGS (ajuste fino) (Raissi et al., 2019).

---

## 7. Sampler de geometría listo: círculos concéntricos + dominio (zanja/terreno)

> Objetivo: generar puntos internos por región (capas + suelo), puntos de interfaces (bandas alrededor de radios) y puntos de borde.

```python
# geom/sampler.py
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class CableGeometry:
    r_cond: float
    r_xlpe: float
    r_screen: float
    r_cover: float

@dataclass(frozen=True)
class Domain2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

@dataclass(frozen=True)
class CablePlacement:
    cx: float
    cy: float

def _rand_uniform(n: int, lo: float, hi: float, device=None) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand((n, 1), device=device)

def _in_annulus(xy: torch.Tensor, c: CablePlacement, r_in: float, r_out: float) -> torch.Tensor:
    dx = xy[:, 0:1] - c.cx
    dy = xy[:, 1:2] - c.cy
    r2 = dx*dx + dy*dy
    return (r2 >= r_in*r_in) & (r2 < r_out*r_out)

def _in_circle(xy: torch.Tensor, c: CablePlacement, r: float) -> torch.Tensor:
    dx = xy[:, 0:1] - c.cx
    dy = xy[:, 1:2] - c.cy
    return (dx*dx + dy*dy) < (r*r)

def sample_domain_points(dom: Domain2D, cable: CableGeometry, place: CablePlacement,
                        n_interior: int, n_interface: int, oversample: int = 5, device=None):
    # Interior por rejection sampling
    n_try = n_interior * oversample
    x = _rand_uniform(n_try, dom.xmin, dom.xmax, device=device)
    y = _rand_uniform(n_try, dom.ymin, dom.ymax, device=device)
    xy = torch.cat([x, y], dim=1)

    m_cond   = _in_circle(xy, place, cable.r_cond)
    m_xlpe   = _in_annulus(xy, place, cable.r_cond,   cable.r_xlpe)
    m_screen = _in_annulus(xy, place, cable.r_xlpe,   cable.r_screen)
    m_cover  = _in_annulus(xy, place, cable.r_screen, cable.r_cover)
    m_soil   = ~_in_circle(xy, place, cable.r_cover)

    def _take(mask: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.where(mask.squeeze(1))[0]
        if idx.numel() == 0:
            return torch.empty((0,2), device=device)
        if idx.numel() < n:
            ridx = idx[torch.randint(0, idx.numel(), (n,), device=device)]
        else:
            ridx = idx[torch.randperm(idx.numel(), device=device)[:n]]
        return xy[ridx]

    # Partición simple (ajústala)
    n_cond   = max(1, int(0.15 * n_interior))
    n_xlpe   = max(1, int(0.20 * n_interior))
    n_screen = max(1, int(0.10 * n_interior))
    n_cover  = max(1, int(0.10 * n_interior))
    n_soil   = max(1, n_interior - (n_cond+n_xlpe+n_screen+n_cover))

    pts = {
        "cond":   _take(m_cond,   n_cond),
        "xlpe":   _take(m_xlpe,   n_xlpe),
        "screen": _take(m_screen, n_screen),
        "cover":  _take(m_cover,  n_cover),
        "soil":   _take(m_soil,   n_soil),
    }

    # Interfaces: muestreo angular + banda radial
    eps = 0.002 * cable.r_cover
    angles = 2.0 * math.pi * torch.rand((n_interface, 1), device=device)

    def _ring_points(r: float) -> torch.Tensor:
        rr = r + eps * (2.0 * torch.rand((n_interface, 1), device=device) - 1.0)
        x = place.cx + rr * torch.cos(angles)
        y = place.cy + rr * torch.sin(angles)
        return torch.cat([x, y], dim=1)

    interfaces = {
        "r_cond":   _ring_points(cable.r_cond),
        "r_xlpe":   _ring_points(cable.r_xlpe),
        "r_screen": _ring_points(cable.r_screen),
        "r_cover":  _ring_points(cable.r_cover),
    }
    return pts, interfaces

def sample_boundary_points(dom: Domain2D, n_b: int, device=None):
    nh = n_b // 2
    nv = n_b - nh
    xh = _rand_uniform(nh, dom.xmin, dom.xmax, device=device)
    y_bottom = torch.full((nh, 1), dom.ymin, device=device)
    y_top    = torch.full((nh, 1), dom.ymax, device=device)

    yv = _rand_uniform(nv, dom.ymin, dom.ymax, device=device)
    x_left  = torch.full((nv, 1), dom.xmin, device=device)
    x_right = torch.full((nv, 1), dom.xmax, device=device)

    return {
        "bottom": torch.cat([xh, y_bottom], dim=1),
        "top":    torch.cat([xh, y_top], dim=1),
        "left":   torch.cat([x_left,  yv], dim=1),
        "right":  torch.cat([x_right, yv], dim=1),
    }

def sample_time(n_t: int, t0: float, t1: float, device=None):
    return _rand_uniform(n_t, t0, t1, device=device)
```

---

## 8. Solver PINN (plantilla PyTorch): estacionario + transitorio + k(x,y) variable

```python
# pinn/model.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, width: int = 128, depth: int = 6, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

```python
# materials/props.py
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class MaterialProps:
    k_cond: float
    k_xlpe: float
    k_screen: float
    k_cover: float
    k_soil: float

    rho_c_cond: float
    rho_c_xlpe: float
    rho_c_screen: float
    rho_c_cover: float
    rho_c_soil: float

def k_soil_variable(xy: torch.Tensor, k0: float, amp: float = 0.3):
    # Ejemplo suave para tests (validación de coeficientes variables)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return k0 * (1.0 + amp * torch.sin(2.0 * torch.pi * x) * torch.cos(2.0 * torch.pi * y))

def get_k_rhoc(region: str, xy: torch.Tensor, mp: MaterialProps, soil_variable: bool):
    if region == "cond":
        return mp.k_cond, mp.rho_c_cond
    if region == "xlpe":
        return mp.k_xlpe, mp.rho_c_xlpe
    if region == "screen":
        return mp.k_screen, mp.rho_c_screen
    if region == "cover":
        return mp.k_cover, mp.rho_c_cover
    if region == "soil":
        k = k_soil_variable(xy, mp.k_soil) if soil_variable else mp.k_soil
        return k, mp.rho_c_soil
    raise ValueError(region)
```

```python
# pinn/pde.py
import torch

def gradients(u: torch.Tensor, x: torch.Tensor):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

def laplace_variable_k(T: torch.Tensor, xy: torch.Tensor, k):
    # div(k grad T) en 2D. k escalar o (N,1)
    if not torch.is_tensor(k):
        k = torch.tensor(k, device=xy.device, dtype=xy.dtype).view(1,1)
    if k.ndim == 0:
        k = k.view(1,1)

    gT = gradients(T, xy)  # (N,2)
    k_ = k if k.shape[0] == xy.shape[0] else k.expand(xy.shape[0], 1)

    k_gT = torch.cat([k_ * gT[:, 0:1], k_ * gT[:, 1:2]], dim=1)
    d_k_gT_dx = gradients(k_gT[:, 0:1], xy)[:, 0:1]
    d_k_gT_dy = gradients(k_gT[:, 1:2], xy)[:, 1:2]
    return d_k_gT_dx + d_k_gT_dy  # (N,1)

def pde_residual_steady(T: torch.Tensor, xy: torch.Tensor, k, Q):
    # Estacionario: div(k gradT) + Q = 0
    if not torch.is_tensor(Q):
        Q = torch.tensor(Q, device=xy.device, dtype=xy.dtype).view(1,1)
    if Q.ndim == 0:
        Q = Q.view(1,1)
    Q_ = Q if Q.shape[0] == xy.shape[0] else Q.expand(xy.shape[0], 1)
    return laplace_variable_k(T, xy, k) + Q_

def pde_residual_transient(T: torch.Tensor, xyt: torch.Tensor, k, rho_c, Q):
    # Transitorio: div(k gradT) + Q = rho*c * dT/dt
    xy = xyt[:, 0:2]
    g = gradients(T, xyt)  # (N,3)
    dTdt = g[:, 2:3]

    div_term = laplace_variable_k(T, xy, k)

    if not torch.is_tensor(Q):
        Q = torch.tensor(Q, device=xyt.device, dtype=xyt.dtype).view(1,1)
    Q_ = Q if Q.shape[0] == xyt.shape[0] else Q.expand(xyt.shape[0], 1)

    if not torch.is_tensor(rho_c):
        rho_c = torch.tensor(rho_c, device=xyt.device, dtype=xyt.dtype).view(1,1)
    rho_c_ = rho_c if rho_c.shape[0] == xyt.shape[0] else rho_c.expand(xyt.shape[0], 1)

    return div_term + Q_ - rho_c_ * dTdt
```

```python
# pinn/losses.py
import torch

def mse(x: torch.Tensor):
    return torch.mean(x*x)

def dirichlet_loss(T_pred: torch.Tensor, T_target: torch.Tensor):
    return mse(T_pred - T_target)

def interface_T_loss(T_left: torch.Tensor, T_right: torch.Tensor):
    return mse(T_left - T_right)

def interface_flux_loss(flux_left: torch.Tensor, flux_right: torch.Tensor):
    return mse(flux_left - flux_right)
```

```python
# pinn/train.py
from dataclasses import dataclass
from torch.optim import Adam, LBFGS

@dataclass
class TrainConfig:
    lr: float = 1e-3
    adam_steps: int = 20000
    lbfgs_steps: int = 5000
    print_every: int = 200

def train(model, loss_fn, cfg: TrainConfig):
    model.train()
    opt = Adam(model.parameters(), lr=cfg.lr)

    for it in range(1, cfg.adam_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
        if it % cfg.print_every == 0:
            print(f"[Adam] it={it} loss={loss.item():.4e}")

    opt2 = LBFGS(model.parameters(), max_iter=cfg.lbfgs_steps, history_size=50, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return loss
    loss = opt2.step(closure)
    print(f"[LBFGS] final loss={float(loss):.4e}")
```

---

## 9. Workflow recomendado

### 9.1. Verificación (antes de aplicar a cable real)
1) MMS con k(x,y) suave para comprobar coeficientes variables (Raissi et al., 2019).  
2) Rectángulo con Dirichlet (solución por series).  
3) Cilindro/anillo multicapa radial (analítico) (Carslaw & Jaeger, 1959).

### 9.2. Validación (aplicación)
4) Caso cable XLPE en zanja: comparar contra FEM y/o caso de referencia publicado (Aras et al., 2005).  
5) Caso con 3 cables y/o interferencia térmica externa (Aras et al., 2005).

### 9.3. Transitorio
6) Agregar t: muestreo (x,y,t), IC y pruebas de escalón/ciclo de carga (Raissi et al., 2019).

---

## 10. Extensiones recomendadas (si el entrenamiento se traba)

### 10.1. XPINNs / domain decomposition
Si el PINN global sufre con discontinuidades fuertes, implementar una red por subdominio (capas + suelo) y acoples en interfaces (Jagtap & Karniadakis, 2020).

### 10.2. PINN paramétrico (surrogate)
Entrenar una red con inputs (x,y,t,p) donde p incluye k_suelo, profundidad, separación, corriente/pérdidas. Esto habilita exploración rápida de escenarios (Raissi et al., 2019).

### 10.3. Ampacidad por loop de corriente
Procedimiento externo típico:
- dado I → computar pérdidas → resolver T(x,y,t) → evaluar T_max en conductor
- ajustar I (bisección) hasta T_max = T_límite (IEC, 2023).

---

## 11. Referencias (APA 7)

- Aras, F., Oysu, C., & Yilmaz, G. (2005). *An assessment of the methods for calculating ampacity of underground power cables*. **Electric Power Components and Systems, 33**(12), 1385–1402. https://doi.org/10.1080/15325000590964425  
- Carslaw, H. S., & Jaeger, J. C. (1959). *Conduction of Heat in Solids* (2nd ed.). Oxford University Press.  
- International Electrotechnical Commission. (2023). *IEC 60287-1-1: Electric cables—Calculation of the current rating—Part 1-1: Current rating equations (100% load factor) and calculation of losses—General*. IEC.  
- Jagtap, A. D., & Karniadakis, G. E. (2020). Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. *Communications in Computational Physics, 28*(5), 2002–2041. https://doi.org/10.4208/cicp.OA-2020-0164  
- Neher, J. H., & McGrath, M. H. (1957). The calculation of the temperature rise and load capability of cable systems. *Transactions of the American Institute of Electrical Engineers, Part III: Power Apparatus and Systems, 76*, 752–772.  
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
