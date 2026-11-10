"""APA 7 references and verified pagination of the consulted versions."""
REFERENCES={
 'wu2022':'Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2022). *A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks* [Prepublicación, versión 1]. arXiv. https://arxiv.org/abs/2207.10289v1',
 'deryck2024':'De Ryck, T., & Mishra, S. (2024). *Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning* [Prepublicación, versión 1]. arXiv. https://arxiv.org/abs/2402.10926v1',
 'mishra2020':'Mishra, S., & Molinaro, R. (2020). *Estimates on the generalization error of physics informed neural networks (PINNs) for approximating PDEs* [Prepublicación, versión 1]. arXiv. https://arxiv.org/abs/2006.16144v1',
 'baratta2023':'Baratta, I. A., Dean, J. P., Dokken, J. S., Habera, M., Hale, J. S., Richardson, C. N., Rognes, M. E., Scroggs, M. W., Sime, N., & Wells, G. N. (2023). *DOLFINx: The next generation FEniCS problem solving environment* [Prepublicación, versión 1]. Zenodo. https://doi.org/10.5281/zenodo.10447666',
 'cigre2022':'CIGRÉ Working Group B1.56. (2022). *Power cable rating examples for calculation tool verification* (Technical Brochure No. 880). CIGRÉ.',
 'raissi2017':'Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). *Physics informed deep learning (Part I): Data-driven solutions of nonlinear partial differential equations* [Prepublicación, versión 1]. arXiv. https://arxiv.org/abs/1711.10561',
 'wang2020':'Wang, S., Teng, Y., & Perdikaris, P. (2020). *Understanding and mitigating gradient pathologies in physics-informed neural networks* [Prepublicación, versión 1]. arXiv. https://arxiv.org/abs/2001.04536',
 'cigre2025':'CIGRÉ Working Group B1.87. (2025). *Finite element analysis for cable rating calculations* (Technical Brochure No. 963). CIGRÉ.',
 'aras2005':'Aras, F., Oysu, C., & Yılmaz, G. (2005). An assessment of the methods for calculating ampacity of underground power cables. *Electric Power Components and Systems, 33*(12), 1385–1402. https://doi.org/10.1080/15325000590964425',
 'kim2025':'Kim, Y.-S., Cong, H. N., Dinh, B. H., & Kim, H.-K. (2025). Effect of ambient air and ground temperatures on heat transfer in underground power cable system buried in newly developed cable bedding material. *Geothermics, 125*, Article 103151. https://doi.org/10.1016/j.geothermics.2024.103151',
 'shukla2021':'Shukla, K., Jagtap, A. D., & Karniadakis, G. E. (2021). Parallel physics-informed neural networks via domain decomposition. *Journal of Computational Physics, 447*, Article 110683. https://doi.org/10.1016/j.jcp.2021.110683 (Versión consultada: https://arxiv.org/abs/2104.10013v3).',
}

def case_citations(name):
    keys=['raissi2017','cigre2025','wang2020','cigre2022','baratta2023','deryck2024','mishra2020']
    text='El entrenamiento mediante residuos y diferenciación automática se fundamenta en la versión consultada de Raissi et al. (2017: 4–5). La verificación del cálculo térmico sigue la distinción entre verificación y validación de CIGRÉ Working Group B1.87 (2025: 93–99). La comparación de pesos responde a dificultades de optimización documentadas por Wang et al. (2020: 9–10); no se implementa ni se atribuye a la campaña su algoritmo adaptativo.'
    text+=' La implementación FEM utiliza DOLFINx, cuyo entorno de solución y relación con FEniCSx describen Baratta et al. (2023: 1–3). Las versiones de software efectivamente ejecutadas se registran aparte.'
    text+=' La separación entre capacidad, muestreo y optimización sigue De Ryck y Mishra (2024: 24–27, 51–57). Las cotas que relacionan error, estabilidad y cuadratura requieren hipótesis específicas (Mishra y Molinaro, 2020: 7–8); las pendientes observadas en esta batería no se presentan como órdenes teóricos de PINN.'
    if name.startswith('aras'):
        keys.append('aras2005');text+=' Las dimensiones del cable proceden de Aras et al. (2005: 1390). El dominio, la conductividad de base y las pérdidas del caso común son adaptaciones declaradas.'
    elif name.startswith('kim'):
        keys.append('kim2025');text+=' Los datos geométricos y eléctricos se contrastan con Kim et al. (2025: 3–5), y las propiedades de relleno con su discusión de materiales (2025: 10–12). La referencia FEM de este cuaderno resuelve la adaptación común, no reproduce las temperaturas publicadas.'
    elif name=='annulus':
        text+=' El anillo adapta el ejemplo analítico de CIGRÉ Working Group B1.87 (2025: 52–55), usando exactamente 40 W/m.'
    else:
        text+=' La solución manufacturada o el escenario XLPE específico es de elaboración propia; sus valores no se atribuyen a un experimento publicado.'
    if name.startswith(('aras','kim','xlpe')):
        keys.append('cigre2022');text+=' La resistencia DC se actualiza con la temperatura individual del conductor, siguiendo la ley lineal y el coeficiente del cobre documentados en CIGRÉ Working Group B1.56 (2022: 132). Las pérdidas AC adicionales de esa fuente quedan fuera del modelo DC ejecutado.'
    if 'interface' in name or 'layered_y' in name or 'discrete' in name:
        keys.append('shukla2021');text+=' La continuidad entre subredes se relaciona con Shukla et al. (2021: 1, 5–6, paginación de la versión arXiv consultada).'
    if name in ['kim_layered','xlpe_backfill']:
        keys.append('wu2022');text+=' La comparación de renovación aleatoria y distribución por residuo se fundamenta en Wu et al. (2022: 6–8). El muestreo mixto con anclajes geométricos y el indicador de gradiente térmico son adaptaciones propias, no una reproducción exacta de RAD.'
    refs='\n\n'.join(REFERENCES[k] for k in sorted(set(keys),key=lambda k:REFERENCES[k]))
    return text,refs
