import numpy as np

from collections import namedtuple

def LMMexp(f, T, u0, k, alpha, beta, h):

    """
    IN:
        f(t, u) : muss array auf array abbilden
        T       : [t0,te]
        u0      : Startwert (array)
        k       : Schrittanzahl
        alpha   : Koeffizienten der u_i, Funktion von k
        beta    : Koeffizienten der f_i, Funktion von k
        h       : Zeitschrittweite

    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.h    : Zeitschrittweite
        erg.nfev : Anzahl f-Auswertungen

    t0 und te sind sicher im Output
    """

    # Start- und Endzeit ermitteln
    t0 = T[0]
    te = T[1]

    # Anfangswert ggf. auf Zeilenvektor transformieren
    u0 = np.array([u0]).flatten()
    n = len(u0)

    # Anfangszeit und -wert in Lösungsliste eintragen
    tv = [t0]
    uv = [u0]

    # eigentliches Verfahren
    t = t0
    u = u0

    # ttol für Endzeiterkennung
    ttol = 1e-14

    # f-Werte speichern
    f_werte = [f(t0, u0)]
    nfev = 1

    alpha_matrix = []  # Liste statt Matrix
    beta_matrix = []
    for i in range(1, k + 1):
        alpha_matrix.append(alpha(i))
        beta_matrix.append(beta(i))

    while t < te - ttol:
        # Schrittweite ggf. anpassen für letzten Schritt
        h_local = min(h, te - t)

        # Aktuelle Gedächtnislänge bestimmen
        current_k = min(len(tv), k)

        # Koeffizienten für aktuelle Gedächtnislänge holen
        alpha_k = alpha_matrix[current_k - 1]  # -1 wegen 0-basierter Indizierung
        beta_k = beta_matrix[current_k - 1]

        # Sicherstellen dass es ein explizites Verfahren ist
        assert beta_k[-1] == 0, "Verfahren muss explizit sein (beta_k = 0)"
        assert alpha_k[-1] != 0, "alpha_k darf nicht 0 sein"

        # Summen für die Verfahrensvorschrift initialisieren
        sum_alpha = np.zeros(n)
        sum_beta = np.zeros(n)

        # Summen für die Verfahrensvorschrift initialisieren
        sum_alpha = np.zeros(n)
        sum_beta = np.zeros(n)

        # Summen berechnen
        for j in range(current_k):
            idx = -(j + 1)  # Index für vorherige Werte
            sum_alpha += alpha_k[j] * uv[-current_k + j]
            sum_beta += beta_k[j] * f_werte[-current_k + j]

        # Neuen Wert berechnen (für Adams-Bashforth)
        u_new = -(sum_alpha - h_local * sum_beta) * alpha_k[-1]
        t_new = t + h_local

        # f-Auswertung für den neuen Punkt
        f_new = f(t_new, u_new)
        nfev += 1

        # Werte anhängen
        t = t_new
        u = u_new
        tv.append(t)
        uv.append(u)
        f_werte.append(f_new)

    # Ergebnis zusammenstellen
    erg = namedtuple('erg', ['t', 'y', 'h', 'nfev'])
    return erg(np.array(tv), np.array(uv).T, h, nfev)

def LMMexp_PEC(f, T, u0, k, alphap, betap, alphac, betac, h, mc):

    """
    IN:
        f(t, u) : muss array auf array abbilden
        T       : [t0,te]
        u0      : Startwert (array)
        k       : Schrittanzahl
        alphap  : Koeffizienten der u_i, Funktion von k, Prädiktor
        betap   : Koeffizienten der f_i, Funktion von k, Prädiktor
        alphac  : Koeffizienten der u_i, Funktion von k, Korrektor
        betac   : Koeffizienten der f_i, Funktion von k, Korrektor
        h       : Zeitschrittweite
        mc      : Anzahl der Korrekturen

    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.h    : Zeitschrittweite
        erg.nfev : Anzahl f-Auswertungen

    t0 und te sind sicher im Output
    """

    # Start- und Endzeit ermitteln
    t0 = T[0]
    te = T[1]

    # Anfangswert ggf. auf Zeilenvektor transformieren
    u0 = np.array([u0]).flatten()
    n = len(u0)

    # Anfangszeit und -wert in Lösungsliste eintragen
    tv = [t0]
    uv = [u0]

    # eigentliches Verfahren
    t = t0
    u = u0

    # ttol für Endzeiterkennung
    ttol = 1e-14

    # f-Werte speichern
    f_werte = [f(t0, u0)]
    nfev = 1

    alpha_p_matrix = []
    beta_p_matrix = []
    alpha_c_matrix = []
    beta_c_matrix = []
    for i in range(1, k + 1):
        alpha_p_matrix.append(alphap(i))
        beta_p_matrix.append(betap(i))
        alpha_c_matrix.append(alphac(i))
        beta_c_matrix.append(betac(i))

    while t < te - ttol:
        # Schrittweite ggf. anpassen für letzten Schritt
        h_local = min(h, te - t)

        # Aktuelle Gedächtnislänge bestimmen
        current_k = min(len(tv), k)

        # Koeffizienten für aktuelle Gedächtnislänge holen
        alpha_p_k = alpha_p_matrix[current_k - 1]  # -1 wegen 0-basierter Indizierung
        beta_p_k = beta_p_matrix[current_k - 1]
        alpha_c_k = alpha_c_matrix[current_k - 1]
        beta_c_k = beta_c_matrix[current_k - 1]

        # Sicherstellen dass es ein explizites Verfahren ist
        assert beta_p_k[-1] == 0, "Verfahren muss explizit sein (beta_k = 0)"
        assert alpha_p_k[-1] != 0, "alpha_k darf nicht 0 sein"

        # Erst Adams-Bashfort:
        # Summen für die Verfahrensvorschrift initialisieren
        sum_alpha_p = np.zeros(n)
        sum_beta_p = np.zeros(n)

        # Summen berechnen
        for j in range(current_k):
            sum_alpha_p += alpha_p_k[j] * uv[-current_k + j]
            sum_beta_p += beta_p_k[j] * f_werte[-current_k + j]

        # Neuen Wert berechnen (für Adams-Bashforth)
        u_new = np.zeros((mc, len(u0)))
        u_new[0, :] = -(sum_alpha_p - h_local * sum_beta_p) * alpha_p_k[-1]
        t_new = t + h_local

        # f-Auswertung für den neuen Punkt
        f_new = f(t_new, u_new[0])
        nfev += 1

        # Werte anhängen
        t = t_new
        u = u_new[0]
        tv.append(t)
        uv.append(u)
        f_werte.append(f_new)


        for l in range(1,mc):
            # Summen für die Verfahrensvorschrift initialisieren
            sum_alpha_c = np.zeros(n)
            sum_beta_c = np.zeros(n)
            for j in range(current_k):
                idx = -(j + 1)
                sum_alpha_c += alpha_c_k[j] * uv[-current_k + j]
                sum_beta_c += beta_c_k[j] * f_werte[-current_k + j]

            # u^{(l)} berechnen
            u_new[l, :] = -(sum_alpha_c - h_local * sum_beta_c) * alpha_c_k[-1]
            f_new = f(t_new, u_new[l, :])
            nfev += 1

            # den letzten Eintrag (also u_i+1^{(l-1)}) durch u_i+1^{(l)} ersetzen.
            uv[-1] = u_new[l, :]
            f_werte[-1] = f_new

    # Ergebnis zusammenstellen
    erg = namedtuple('erg', ['t', 'y', 'h', 'nfev'])
    return erg(np.array(tv), np.array(uv).T, h, nfev)