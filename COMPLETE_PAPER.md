% Template for ICASSP-2026 paper; to be used with:
%          spconf.sty  - ICASSP/ICIP LaTeX style file, and
%          IEEEbib.bst - IEEE bibliography style file.
% --------------------------------------------------------------------------
\documentclass{article}
\usepackage{spconf,amsmath,graphicx,hyperref}

\usepackage{tabularray}
\usepackage{paralist}
\usepackage[ngerman]{babel}
\usepackage{microtype}
% Title.
% ------
\title{ALS-Schweregradklassifikation mit Wav2Vec 2.0}

\name{
  \begin{tabular}{cc}
    Sandro Diakourakis & Fabian Drzimalla \\
    \texttt{1115059} & \texttt{1114573}
  \end{tabular}
}
\address{}


\begin{document}
%\ninept
%
\maketitle
%
\begin{abstract}
Diese Arbeit untersucht die automatische Bestimmung des ALS-Schweregrads anhand von acht Sprachaufnahmen pro Patient mittels Wav2Vec-2.0-basierter Repräsentationen. Verglichen werden eine Support Vector Machine mit gemittelten Wav2Vec-2.0-Base-Features und ein neuronales Late-Fusion-Netzwerk, das Multi-Layer-Repräsentationen aus Wav2Vec-2.0-Large mit file-spezifischen Teilnetzen kombiniert. Das neuronale Modell erzielt auf dem Validierungssatz einen F1-Macro-Score von 0,594 und übertrifft damit die SVM-Baseline (0,45) deutlich. Die Ergebnisse zeigen, dass die differenzierte Modellierung unterschiedlicher phonetischer Aufgaben durch separate Netzwerkkomponenten zu ausdrucksstärkeren Repräsentationen führt als einfache Mittelungsstrategien. Fehlklassifikationen treten vorwiegend zwischen benachbarten Schweregraden auf, was dem graduellen Krankheitsverlauf entspricht und die Kohärenz der gelernten Merkmale unterstreicht.
\end{abstract}
%
\begin{keywords}
Amyotrophe Lateralsklerose, Sprachklassifikation, Wav2Vec 2.0, Late Fusion, Deep Learning
\end{keywords}
%

\section{Methodik}
\label{sec:method}

\subsection{Feature-Extraktion mit Wav2Vec 2.0}

Für beide Modelle dient Wav2Vec 2.0 als Feature-Extractor, der kontextualisierte Repräsentationen direkt aus Rohaudio erzeugt. Die Audiodaten werden auf 16\,kHz resampled, normalisiert und anschließend vom Wav2Vec-Prozessor in Transformer-kompatible Eingaben überführt. Die Modelle unterscheiden sich in der verwendeten Wav2Vec-Variante und der Tiefe der Feature-Extraktion \cite{DBLP:journals/corr/abs-2006-11477}.

\subsection{Late-Fusion-Strategie}

Jeder Patient wird durch acht unterschiedliche Sprachaufnahmen repräsentiert. Beide Ansätze verarbeiten jede Datei separat über Wav2Vec~2.0 und fusionieren die resultierenden Repräsentationen anschließend \cite{LateFusion}. Der SVM-Ansatz nutzt eine einfache Mittelwertbildung über alle Feature-Vektoren, während das neuronale Modell file-spezifische Teilnetze einsetzt und die erzeugten Repräsentationen zu einem gemeinsamen Vektor kombiniert.


\section{Modelle}
\label{sec:pagestyle}

\subsection{Support Vector Machine mit Late Fusion}

Der SVM-Ansatz \cite{SVM} nutzt Wav2Vec 2.0-Base als festen Feature-Extractor. Für jede der acht Audiodateien eines Patienten werden die Hidden States über die Zeit gemittelt, wodurch acht 768-dimensionale Feature-Vektoren entstehen. Diese werden per arithmetischer Mittelung zu einer einzigen Patientenrepräsentation fusioniert, die als Eingang für die Support Vector Machine dient. Untersucht wurden ein linearer und ein RBF-Kernel, um sowohl lineare als auch nichtlineare Entscheidungsgrenzen abzudecken.

Die Hyperparameterbestimmung erfolgte mittels Grid Search \cite{gridsearch} und fünffacher Kreuzvalidierung \cite{kfoldcrossvalidation} auf dem Trainingsset. Variiert wurden der Regularisierungsparameter~$C$, die Kernelwahl sowie der Gamma-Wert für den RBF-Kernel. Optimiert wurde auf den F1-Macro-Score \cite{Metrics}, der die stark unausgewogene Klassenverteilung berücksichtigt und alle Schweregrade gleichgewichtet behandelt.

Das finale Modell wurde mit den optimalen Parametern auf den fusionierten Wav2Vec-Features trainiert. Der SVM-Ansatz bildet damit eine kompakte, nicht fine-getunte Baseline, die allein auf aggregierten Wav2Vec-Features operiert.


\subsection{Neuronales Late-Fusion-Netzwerk}

Der neuronale Ansatz erweitert die reine Feature-Mittelung des SVM-Modells durch eine mehrstufige Late-Fusion-Architektur. Ziel ist es, die acht heterogenen Sprachaufnahmen eines Patienten separat zu modellieren und deren Informationsgehalt differenziert in die Klassifikation einzubeziehen.

\textbf{Multi-Layer Feature-Extraktion.}
Als Grundlage dient Wav2Vec 2.0-Large, aus dem Merkmalsrepräsentationen aus fünf unterschiedlichen Transformer-Layern (6, 9, 12, 15, 18) extrahiert werden. Die Layer werden nach Mean Pooling über die Zeit konkateniert, wodurch pro Datei ein 5120-dimensionaler Feature-Vektor entsteht. Diese Multi-Layer-Strategie kombiniert akustische und zunehmend abstrakte Sprachmerkmale und liefert deutlich reichhaltigere Repräsentationen als ein einzelner Transformer-Layer \cite{DBLP:journals/corr/abs-2006-11477}.

\textbf{File-Processor-Netzwerke.}
Für jede der acht Audiodateien wird ein eigenes Feedforward-Netzwerk eingesetzt, das die 5120-dimensionalen Wav2Vec-Features auf eine kompakte 128-dimensionale Repräsentation projiziert. Die File-Processors besitzen unabhängige Gewichte und können dadurch unterschiedliche phonetische Aufgaben (Vokale vs.\ Silbenwiederholungen) modellieren. LayerNorm \cite{ba2016layernormalization} und Dropout \cite{dropout} stabilisieren das Training \cite{DBLP:journals/corr/abs-2006-11477}.

\textbf{Fusion und Klassifikation.}
Die acht file-spezifischen Repräsentationen werden zu einem 1024-dimensionalen Vektor konkateniert, der alle aufnahmespezifischen Informationen vollständig erhält. Ein zweistufiger Klassifikator (Linear~$\rightarrow$~256, ReLU, Dropout, Linear~$\rightarrow$~5) erzeugt daraus die finalen Klassenlogits. Die Gesamtarchitektur umfasst rund 11\,M trainierbare Parameter.

\textbf{Training.}
Optimiert wurde mit AdamW \cite{AdamW} (Learning Rate: $5{\times}10^{-4}$, Weight Decay: $10^{-4}$), Cross-Entropy-Loss \cite{cross-entropy-loss} und automatischen Klassengewichten zur Kompensation der starken Imbalance. Die Batch Size betrug 24, Dropout wurde auf 0.3 gesetzt. Gradient Clipping \cite{GradientClipping} (Max Norm: 1.0) stabilisierte den Trainingsverlauf, Early Stopping mit Patience 10 verhinderte Overfitting. Der Wav2Vec-Feature-Extractor blieb während des gesamten Trainings eingefroren; trainiert wurden ausschließlich die File-Processor- und Klassifikator-Komponenten. Das Modell konvergierte nach 28 Epochen und erzielte einen Validierungs-F1-Macro \cite{Metrics} von 59.4\%.

Dieser Ansatz nutzt damit sowohl die Tiefe des Transformer-Modells als auch die Heterogenität der acht Aufnahmen und bietet eine deutlich ausdrucksstärkere Repräsentation als einfache Mittelungsstrategien.

\section{Ergebnisse}
\label{sec:results}

\subsection{Support Vector Machine}

Der optimierte SVM-Ansatz (RBF-Kernel, $C{=}10$, $\gamma{=}\mathrm{scale}$) erreichte auf dem Validierungssatz einen F1-Macro \cite{Metrics} von \textbf{0.45}. Die Konfusionsmatrix des SVM-Modells (Abb.~\ref{fig:svm}) zeigt eine deutliche Schwerpunktbildung auf den mittleren und höheren Schweregraden. Klasse~1 wird nur teilweise korrekt erkannt (2/3), während Klasse~2 überwiegend als Klasse~3 fehlklassifiziert wird. Die größte Unsicherheit besteht in Klasse~3, die breit über die Klassen~2–4 verteilt vorhergesagt wird. Klasse~4 weist eine heterogene Verteilung mit starken Verwechslungen zu Klasse~5 auf. Klasse~5 wird am zuverlässigsten erkannt (10 korrekte Zuordnungen), zeigt jedoch ebenfalls systematische Verwechslungen mit Klasse~4.

\begin{figure}[h]
    \centering
    \includegraphics[scale=.24]{SVM_ConfusionMatrix}
    \caption{Konfusionsmatrix des SVM-Modells.}
    \label{fig:svm}
\end{figure}

\subsection{Neuronales Late-Fusion-Netzwerk}

Das neuronale Modell erzielte auf dem unabhängigen Validierungssatz eine deutlich höhere Leistung mit einem F1-Macro \cite{Metrics} von \textbf{0.594}, einer Accuracy von \textbf{0.717} und einem F1-Weighted \cite{Metrics} von \textbf{0.718} und übertrifft damit den SVM-Ansatz klar.

Die Konfusionsmatrix des NN-Modells (Abb.~\ref{fig:NN}) zeigt, dass höhere Schweregrade (Klassen~3 und~4) mit hoher Zuverlässigkeit erkannt werden, während die seltenen Klassen erwartungsgemäß geringere Trefferquoten aufweisen. Die Fehlklassifikationen liegen überwiegend zwischen benachbarten Klassen, was dem graduellen Verlauf der ALS und den überlappenden akustischen Merkmalen entspricht. Größere Sprünge zwischen distanten Schweregraden treten nicht auf und unterstreichen die Kohärenz der gelernten Repräsentationen.

\begin{figure}[h]
    \centering
    \includegraphics[scale=.33]{NN_confusion_matrix.png}
    \caption{Konfusionsmatrix des neuronalen Modells.}
    \label{fig:NN}
\end{figure}

% References should be produced using the bibtex program from suitable
% BiBTeX files (here: strings, refs, manuals). The IEEEbib.bst bibliography
% style file from IEEE produces unsorted bibliography list.
% -------------------------------------------------------------------------
\bibliographystyle{IEEEbib}
\bibliography{strings,refs}

\end{document}
