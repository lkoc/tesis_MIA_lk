# Capitulo I: resultado capturado de SlopTotal

**Captura:** 17 de noviembre de 2026, 15:13:21 UTC. **Paginas PDF:** 12 a 28. **Texto identificado por el portal:** 6566 palabras, espanol.

Informe: <https://sloptotal.com/report/?id=3de8ee861873>. Ticket original: `a4878406083c40fa`.

## Resultado mostrado

El portal muestra **96,0/100**, etiqueta **Slop detected** y cabecera **Complete**. Tambien indica que dos de sus 23 motores senalan el texto y que su puntuacion agrega los motores mediante ponderaciones de calibracion. No se interpreta ese valor como porcentaje demostrado de texto escrito por IA.

La tabla solo presenta puntuaciones numericas para **14 de 23 motores**. Los nueve restantes muestran **Waiting**. Por ello, se conserva como informe disponible con cobertura parcial; la cabecera Complete no permite afirmar que todos los motores devolvieron un resultado.

| Motor | Puntuacion mostrada |
|---|---:|
| TMR Detector | 95,2 |
| Structural | 51,3 |
| E5-Small | 37,5 |
| Vocabulary | 27,3 |
| Burstiness | 21,9 |
| OpenAI Detector | 14,3 |
| Formulaic | 4,8 |
| Readability | 1,0 |
| SuperAnnotate | 0,1 |
| Fast-DetectGPT | 0,0 |
| Perplexity | 0,0 |
| ChatGPT Detector | 0,0 |
| Linguistic | 0,0 |
| Sentiment | 0,0 |

Sin puntuacion: ReMoDetect, Binoculars, Cross-Perplexity, Fakespot, BERT-tiny RAID, Desklib DeBERTa, GLTR, Log-Rank y DivEye. No se convierten esos estados en ceros.

## Evidencia y limites

Los datos visibles se conservan en [01_planteamiento_resultados.json](01_planteamiento_resultados.json). Esa transcripcion estructurada se distingue de la [exportacion JSON original](01_planteamiento_exportacion_original.json), obtenida despues mediante el boton Download JSON y la captura del archivo generado en el navegador.

La exportacion corresponde al mismo identificador `3de8ee861873`, pero devuelve **99,3/100** y solo seis motores: Structural Analysis (0,513), Vocabulary Richness (0,273), Formulaic Patterns (0,048), Readability Uniformity (0,01), Sentiment & Hedging (0) y Linguistic Markers (0). Las puntuaciones individuales del JSON usan escala 0 a 1. No coincide con el agregado 96,0 y los 14 motores de la captura visible. Se conservan ambas evidencias sin fusionarlas ni escoger una como verdad definitiva. No se ha establecido la causa de la discrepancia.

El texto contiene tablas, formulas, cifras, referencias y encabezados extraidos del PDF. No se conoce aun el truncamiento interno por motor ni una calibracion especifica para esta prosa academica en espanol. No se modifico el plan ni se respondio al cuestionario de autoria del portal. Este resultado describe el comportamiento del servicio, no un dictamen de plagio, autoria o calidad de la tesis.