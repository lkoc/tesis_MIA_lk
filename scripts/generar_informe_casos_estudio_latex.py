from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Entrega_Google_Drive" / "02_ejemplos_datos_objeto_estudio"
MD_PATH = OUT_DIR / "Informe_datos_casos_estudio_papers.md"
TEX_PATH = OUT_DIR / "Informe_datos_casos_estudio_papers.tex"


PREAMBLE = r"""% !TeX program = lualatex
% Compilacion recomendada:
% latexmk -lualatex Informe_datos_casos_estudio_papers.tex
\RequirePackage{fix-cm}
\documentclass[11pt,a4paper,oneside]{report}

\usepackage{fontspec}
\setmainfont{Arial}
\setsansfont{Arial}
\usepackage[spanish,es-nodecimaldot,es-noshorthands]{babel}
\usepackage{geometry}
\geometry{
  a4paper,
  left=3cm,
  top=3cm,
  right=2.5cm,
  bottom=2.5cm,
  headheight=14pt,
  headsep=0.6cm
}
\usepackage{setspace}
\onehalfspacing
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{calc}
\usepackage{ragged2e}
\usepackage{enumitem}
\usepackage{etoolbox}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage[hidelinks]{hyperref}
\usepackage{bookmark}

\definecolor{unired}{HTML}{8B1A1A}
\definecolor{darkblue}{HTML}{17365D}
\definecolor{lightgray}{HTML}{F2F2F2}

\hypersetup{
  pdftitle={Informe de datos de casos de estudio extraídos de papers},
  pdfauthor={Luis Enrique Koc Góngora y Herbert Antonio Meléndez García},
  pdfsubject={Parte 2 - Ejemplos de datos recolectados del objeto de estudio}
}

\newcommand{\tesisbodyfont}{\fontsize{11}{13.6}\selectfont}
\setlength{\parindent}{1.25cm}
\setlength{\parskip}{12pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.18}
\setlength{\LTpre}{0.45\baselineskip}
\setlength{\LTpost}{0.65\baselineskip}
\AtBeginEnvironment{longtable}{%
  \footnotesize
  \setlength{\tabcolsep}{3pt}%
  \setlength{\parindent}{0pt}%
}
\setlist{
  leftmargin=1.1cm,
  before=\begin{spacing}{1}\tesisbodyfont,
  after=\end{spacing},
  topsep=.5\baselineskip,
  itemsep=.35\baselineskip,
  parsep=0pt,
  partopsep=0pt
}

\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\titleformat{\section}
  {\normalfont\bfseries\fontsize{11}{13.2}\selectfont}
  {\thesection}{0.75em}{}
\titlespacing*{\section}{0pt}{1.0\baselineskip}{0.25\baselineskip}
\titleformat{\subsection}
  {\normalfont\bfseries\fontsize{11}{13.2}\selectfont}
  {\thesubsection}{0.75em}{}
\titlespacing*{\subsection}{0pt}{0.75\baselineskip}{0.25\baselineskip}
\titleformat{\subsubsection}
  {\normalfont\bfseries\fontsize{11}{13.2}\selectfont}
  {\thesubsubsection}{0.75em}{}
\titlespacing*{\subsubsection}{0pt}{0.5\baselineskip}{0.25\baselineskip}

\makeatletter
\renewcommand*\l@section{\@dottedtocline{1}{1.5em}{3.0em}}
\makeatother
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{2}
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}

\fancypagestyle{tesis}{
  \fancyhf{}
  \fancyhead[R]{\normalfont\fontsize{11}{13.2}\selectfont\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[R]{\normalfont\fontsize{11}{13.2}\selectfont\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\pagestyle{tesis}

\newcommand{\tituloTesis}{Diseño y evaluación de una red neuronal informada por física para estimar la temperatura y la ampacidad en cables enterrados con heterogeneidad térmica}
\newcommand{\tituloInforme}{Informe de datos de casos de estudio extraídos de papers}
\newcommand{\subtituloInforme}{Parte 2 -- Ejemplos de datos recolectados del objeto de estudio}

\begin{document}
\tesisbodyfont
\pagenumbering{arabic}

\begin{center}
{\fontsize{9}{10.8}\selectfont\MakeUppercase{\tituloTesis}\par}
\vspace{0.35cm}
{\fontsize{14}{16.8}\selectfont\bfseries\MakeUppercase{\tituloInforme}\par}
\vspace{0.25cm}
{\fontsize{11}{13.2}\selectfont\bfseries \subtituloInforme\par}
\end{center}

\vspace{0.45cm}
\noindent\textbf{Documento base:} Plan de tesis sobre el sistema cable--instalación--entorno térmico en cables eléctricos subterráneos.

\noindent\textbf{Propósito del informe:} organizar, en formato narrativo y tabular, los datos de instalación de cables y casos de estudio extraídos de los papers usados en el documento.
"""


ENDING = r"""
\end{document}
"""


def strip_top_title(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    if lines and lines[0].startswith("**Parte 2"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def widen_summary_table(body: str) -> str:
    start = body.find(r"\begin{longtable}[]{@{}")
    if start < 0:
        return body
    top_rule = body.find(r"\toprule", start)
    if top_rule < 0:
        return body
    custom = r"""\begin{longtable}[]{@{}
  >{\RaggedRight\arraybackslash}p{0.11\linewidth}
  >{\RaggedRight\arraybackslash}p{0.14\linewidth}
  >{\RaggedRight\arraybackslash}p{0.36\linewidth}
  >{\RaggedRight\arraybackslash}p{0.34\linewidth}@{}}
"""
    return body[:start] + custom + body[top_rule:]


def main() -> None:
    source = MD_PATH.read_text(encoding="utf-8")
    body_markdown = strip_top_title(source)
    proc = subprocess.run(
        [
            "pandoc",
            "--from",
            "markdown+pipe_tables",
            "--to",
            "latex",
            "--shift-heading-level-by=-1",
        ],
        input=body_markdown,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    body = widen_summary_table(proc.stdout.strip())
    for key, wrapped in {
        "kim2025": r"kim\-2025",
        "khumalo2025": r"khumalo\-2025",
        "aldulaimi2024": r"aldulaimi\-2024",
        "atoccsa2024": r"atoccsa\-2024",
        "oclon2015": r"oclon\-2015",
        "aras2005": r"aras\-2005",
    }.items():
        body = body.replace(key, wrapped)
    TEX_PATH.write_text(PREAMBLE + "\n\n" + body + "\n" + ENDING, encoding="utf-8")
    print(TEX_PATH)


if __name__ == "__main__":
    main()
