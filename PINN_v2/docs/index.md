# PINN_v2

Marco de simulación para **conducción de calor 2D en régimen permanente** en cables subterráneos.
Compara varias familias de PINN con una referencia de elementos finitos (FEniCSx) y deja un registro
completo de cada ejecución.

## Qué hace

- Resuelve $-\nabla\cdot(k\,\nabla T)=Q$ de forma explícita en cada capa del cable y del suelo.
- Entrena modelos PINN intercambiables: M0, M1, M2, CUSP, DCSNN, FBPINN y las variantes con atención RBA y SA.
- Usa **el mismo archivo de caso** para entrenar un PINN y para correr la solución FEM.
- Permite parametrizar el entrenamiento desde la línea de comandos o desde un archivo JSON, y buscar hiperparámetros.
- Guarda en cada corrida los resultados, la configuración, las semillas y una huella del código (manifest).

## Reglas que no se rompen

- Las capas se tratan una por una; no se usan resistencias térmicas equivalentes ni se reconstruye la temperatura del conductor por vía radial.
- Contacto perfecto en las interfaces. Robin nunca se convierte en Dirichlet.
- La referencia FEM y los datos de los artículos **nunca** son etiquetas de entrenamiento; solo se usan para evaluar.
- Ningún resultado se borra ni se sobrescribe: cada corrida usa una carpeta nueva.
- Un dato que falta no se inventa: el caso se rechaza o se marca como `supuesto` con una nota.

## Por dónde empezar

1. [Inicio rápido](quickstart.md): instalar y correr el primer entrenamiento.
2. [Formato de caso](cases.md): cómo se describe un problema (un solo JSON).
3. [Modelos](models.md) y [Entrenamiento](training.md): cambiar de modelo y ajustar parámetros.
4. [Búsqueda de hiperparámetros](search.md).
5. [Referencia FEM](fem.md): correr FEniCSx con los mismos datos.

Para trabajar sin comandos, usa la [interfaz web](frontend.md).

Si buscas un comando concreto, ve a la [referencia de la línea de comandos](cli.md).

## Estado

Los criterios de aceptación y las compuertas de convergencia del proyecto son **propuestos**, todavía
no confirmados por la persona responsable de la tesis. Ver [Resultados](results.md) y
[Limitaciones](limitations.md).