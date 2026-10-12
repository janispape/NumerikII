# Numerik II – Löser für gewöhnliche und partielle Differentialgleichungen
 
Python-Implementierungen numerischer Verfahren für Differentialgleichungen. Das Repository entstand im Frühjahr 2025 in der Vorlesung **Numerische Mathematik II** im Studiengang *Angewandte Mathematik und Informatik* an der **FH Aachen** (in Kooperation mit dem Forschungszentrum Jülich). Der zweite Teil (Parareal, PDE-Löser, stochastische PDE) ist **freiwillig nach der Vorlesung** aus eigenem Interesse entstanden.
 
## Inhalt
 
### Teil 1: Vorlesung und Übungsblätter
 
Gewöhnliche Differentialgleichungen (ODEs):
 
| Thema | Dateien |
|---|---|
| Explizites Euler-Verfahren | `euler.py` |
| Runge-Kutta-Verfahren, Butcher-Tableaus | `rk.py`, `rungekutta.py`, `butcher.py` |
| Lineare Mehrschrittverfahren | `mehrschritt.py`, `LMMexp.py` |
| Schrittweitensteuerung | siehe Übungsblätter |
| Richtungsfeld | `Richtungsfeld.py` |
| Übungsblätter 1–5 | `Blatt1-Aufgabe3.py`, `Blatt1-Aufgabe4.py`, `Blatt2.py` bis `Blatt5.py` |
 
### Teil 2: Eigene Weiterführung
 
| Thema | Dateien |
|---|---|
| Parareal-Verfahren (parallele Zeitintegration) | `parareal_vanilla.py` |
| Eindimensionaler PDE-Löser (Wärmeleitungsgleichung) | `mein_erster_pde_solver.py` |
| Zweidimensionaler PDE-Löser | `mein_erster_pde_solver_zwei_dim.py` |
| Euler-Maruyama-Verfahren für stochastische Differentialgleichungen | `EulerMaruyama.py` |
| Stochastische Wärmeleitungsgleichung (Animationen) | siehe `animations/` |
 
## Animationen
 
Im Ordner [`animations/`](animations/) liegen GIFs der Lösungen der Wärmeleitungsgleichung mit verschiedenen Anfangsbedingungen und Wärmequellen sowie der stochastischen Wärmeleitungsgleichung mit unterschiedlich starkem Rauschen.
 
## Ausführen
 
Voraussetzung: Python 3 mit NumPy und Matplotlib.
 
```bash
pip install numpy matplotlib
python euler.py
```
 
Die einzelnen Skripte sind eigenständig und können direkt gestartet werden.
 
## Hinweis
 
Die Skripte `Blatt1-Aufgabe3.py` bis `Blatt5.py` sind Lösungen zu Übungsaufgaben der Vorlesung und daher auf die jeweilige Aufgabenstellung zugeschnitten.
 
