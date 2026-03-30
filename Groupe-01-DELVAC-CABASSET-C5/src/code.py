import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "XOM", "JNJ"]

SECTORS = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "Finance": ["JPM", "GS"],
    "Energy": ["XOM"],
    "Health": ["JNJ"],
}

START = "2021-01-01"
END = "2024-12-31"
RF = 0.04
TAU = 0.05
DELTA = 2.5


def get_data():
    print(f"[INFO] Téléchargement Yahoo Finance ({START} → {END})…")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False
    )["Close"].dropna()
    print(f"[OK]   {len(raw)} jours, {len(TICKERS)} actifs.")
    return raw


def compute_stats(prices):
    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252
    return mu, Sigma, returns


def equilibrium_returns(Sigma, delta=DELTA):
    n = Sigma.shape[0]
    w_eq = np.ones(n) / n
    return delta * Sigma.values @ w_eq


def build_omega(P, Q, confidences, Sigma):
    k = len(Q)
    P_ = np.array(P)
    S = Sigma.values
    omega_diag = []
    for i in range(k):
        p_i = P_[i]
        var_p = p_i @ S @ p_i
        c = confidences[i]
        factor = (1 - c) / max(c, 1e-6)
        omega_diag.append(factor * var_p + 1e-8)
    return np.diag(omega_diag)


def black_litterman(mu_eq, Sigma, P, Q, Omega):
    S = Sigma.values
    tS = TAU * S
    P_ = np.array(P)
    Q_ = np.array(Q)
    tS_inv = np.linalg.inv(tS)
    O_inv = np.linalg.inv(Omega)
    M = np.linalg.inv(tS_inv + P_.T @ O_inv @ P_)
    mu_bl = M @ (tS_inv @ mu_eq + P_.T @ O_inv @ Q_)
    Sigma_bl = S + M
    return mu_bl, Sigma_bl


def optimize_portfolio(mu, Sigma, tickers, sectors=SECTORS,
                       sector_cap=0.40, n_samples=50_000):
    mu_ = np.array(mu) if not isinstance(mu, np.ndarray) else mu
    S_ = Sigma.values if hasattr(Sigma, "values") else np.array(Sigma)
    n = len(mu_)
    sector_idx = {}
    for sec, names in sectors.items():
        sector_idx[sec] = [tickers.index(t) for t in names if t in tickers]
    best_w, best_sharpe = np.ones(n) / n, -np.inf
    for _ in range(n_samples):
        w = np.random.dirichlet(np.ones(n))
        feasible = all(
            w[idx].sum() <= sector_cap
            for idx in sector_idx.values()
            if idx
        )
        if not feasible:
            continue
        ret = w @ mu_
        vol = np.sqrt(w @ S_ @ w)
        sh = (ret - RF) / vol if vol > 0 else -np.inf
        if sh > best_sharpe:
            best_sharpe = sh
            best_w = w.copy()
    return best_w


def efficient_frontier(mu, Sigma, n_portfolios=8_000):
    mu_ = np.array(mu) if not isinstance(mu, np.ndarray) else mu
    S_ = Sigma.values if hasattr(Sigma, "values") else np.array(Sigma)
    n = len(mu_)
    all_vols, all_rets, all_sharpes = [], [], []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r = w @ mu_
        v = np.sqrt(w @ S_ @ w)
        sh = (r - RF) / v if v > 0 else 0
        all_vols.append(v)
        all_rets.append(r)
        all_sharpes.append(sh)
    return np.array(all_vols), np.array(all_rets), np.array(all_sharpes)


def portfolio_metrics(w, mu, Sigma):
    mu_ = np.array(mu) if not isinstance(mu, np.ndarray) else mu
    S_ = Sigma.values if hasattr(Sigma, "values") else np.array(Sigma)
    ret = w @ mu_
    vol = np.sqrt(w @ S_ @ w)
    sh = (ret - RF) / vol if vol > 0 else 0
    return ret, vol, sh


def run():
    print("=" * 60)
    print("  BLACK-LITTERMAN – NIVEAU BON")
    print("  Vues avec confiance + Contraintes + Frontière efficace")
    print("=" * 60)

    prices = get_data()
    tickers = list(prices.columns)
    mu_hist, Sigma, returns = compute_stats(prices)

    mu_eq = equilibrium_returns(Sigma)

    n = len(tickers)
    idx = {t: tickers.index(t) for t in tickers}
    P, Q, view_labels, confidences = [], [], [], []

    def add_view(label, p_row, q_val, conf):
        P.append(p_row)
        Q.append(q_val)
        view_labels.append(label)
        confidences.append(conf)

    def abs_view(ticker, q, conf, label):
        row = np.zeros(n)
        row[idx[ticker]] = 1.0
        add_view(label, row, q, conf)

    def rel_view(ticker_a, ticker_b, q, conf, label):
        row = np.zeros(n)
        row[idx[ticker_a]] = 1.0
        row[idx[ticker_b]] = -1.0
        add_view(label, row, q, conf)

    abs_view("AAPL", 0.10, 0.80, "AAPL +10% (conf 80%)")
    abs_view("MSFT", 0.08, 0.65, "MSFT +8%  (conf 65%)")
    abs_view("XOM", -0.05, 0.50, "XOM  -5%  (conf 50%)")
    rel_view("JPM", "GOOGL", 0.04, 0.40, "JPM > GOOGL +4% (conf 40%)")

    Omega = build_omega(P, Q, confidences, Sigma)

    print("\n── Vues de l'investisseur ──")
    for lbl, q, c in zip(view_labels, Q, confidences):
        print(f"  {lbl:<35s}  q={q:+.2%}  conf={c:.0%}")

    mu_bl, Sigma_bl_arr = black_litterman(mu_eq, Sigma, P, Q, Omega)
    Sigma_bl = pd.DataFrame(Sigma_bl_arr, index=tickers, columns=tickers)

    print("\n── Rendements (annualisés) ──")
    print(f"  {'Actif':8s}  {'Historique':>12s}  {'Équilibre':>10s}  {'BL Post.':>10s}")
    print("  " + "-" * 48)
    for i, t in enumerate(tickers):
        print(f"  {t:8s}  {mu_hist[t]:+11.2%}  {mu_eq[i]:+9.2%}  {mu_bl[i]:+9.2%}")

    print("\n── Optimisation sous contraintes (secteur ≤ 40 %) ──")
    print("  [Markowitz] recherche en cours…")
    w_mk = optimize_portfolio(mu_hist.values, Sigma, tickers)
    print("  [BL]        recherche en cours…")
    w_bl = optimize_portfolio(mu_bl, Sigma_bl, tickers)

    r_mk, v_mk, sh_mk = portfolio_metrics(w_mk, mu_hist.values, Sigma)
    r_bl, v_bl, sh_bl = portfolio_metrics(w_bl, mu_bl, Sigma_bl)

    print("\n── Métriques des portefeuilles ──")
    print(f"  {'Stratégie':15s}  {'Rend.':>8s}  {'Vol.':>8s}  {'Sharpe':>8s}")
    print("  " + "-" * 44)
    for label, r, v, sh in [
        ("Markowitz", r_mk, v_mk, sh_mk),
        ("BL+Conf", r_bl, v_bl, sh_bl),
    ]:
        print(f"  {label:15s}  {r:+7.2%}  {v:7.2%}  {sh:7.2f}")

    print("\n── Allocations (%) ──")
    print(f"  {'Actif':8s}  {'Secteur':10s}  {'Markowitz':>10s}  {'BL+Conf':>10s}")
    print("  " + "-" * 46)
    sector_map = {t: s for s, ts in SECTORS.items() for t in ts}
    for i, t in enumerate(tickers):
        sec = sector_map.get(t, "?")
        print(f"  {t:8s}  {sec:10s}  {w_mk[i]:9.1%}   {w_bl[i]:9.1%}")

    print("\n── Vérification contrainte secteur (≤ 40 %) ──")
    for sec, names in SECTORS.items():
        idxs = [tickers.index(t) for t in names if t in tickers]
        s_mk = sum(w_mk[i] for i in idxs)
        s_bl = sum(w_bl[i] for i in idxs)
        ok_mk = "✔" if s_mk <= 0.401 else "✘"
        ok_bl = "✔" if s_bl <= 0.401 else "✘"
        print(f"  {sec:10s}  MK={s_mk:.1%} {ok_mk}   BL={s_bl:.1%} {ok_bl}")

    print("\nCalcul des frontières efficaces…")
    v_mk_f, r_mk_f, sh_mk_f = efficient_frontier(mu_hist.values, Sigma)
    v_bl_f, r_bl_f, sh_bl_f = efficient_frontier(mu_bl, Sigma_bl)

    print("\nOuverture des 6 graphiques…")
    _plot_rendements(tickers, mu_hist.values, mu_eq, mu_bl)
    _plot_confiances(view_labels, confidences)
    _plot_allocations(tickers, w_mk, w_bl)
    _plot_frontier_markowitz(v_mk_f, r_mk_f, sh_mk_f, (v_mk, r_mk, sh_mk))
    _plot_frontier_bl(v_bl_f, r_bl_f, sh_bl_f, (v_bl, r_bl, sh_bl))
    _plot_recap(stats_mk=(v_mk, r_mk, sh_mk), stats_bl=(v_bl, r_bl, sh_bl))

    plt.show()
    print("\n✔  Tous les graphiques sont affichés.")


BG = "#0F1117"
PAN = "#1A1D2E"
C1 = "#3BB273"
C2 = "#2E86AB"
C3 = "#F4A261"
C4 = "#E84855"
GREY = "#888"


def _new_fig(title, figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PAN)
    fig.canvas.manager.set_window_title(title)
    return fig, ax


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(color="#333", linewidth=0.5, linestyle="--", alpha=0.6)


def _plot_rendements(tickers, mu_hist, mu_eq, mu_bl):
    fig, ax = _new_fig("A – Rendements : Prior → Posteriori")
    x = np.arange(len(tickers))
    w = 0.22
    ax.bar(x - w, mu_hist * 100, w, label="Historique", color=C4, alpha=0.85)
    ax.bar(x, mu_eq * 100, w, label="Équilibre", color=C3, alpha=0.85)
    ax.bar(x + w, mu_bl * 100, w, label="BL+Conf", color=C1, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=35, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_ylabel("Rendement ann. (%)", color="white")
    ax.set_title("Rendements : Prior → Posteriori", color="white", fontweight="bold")
    ax.legend(facecolor=PAN, labelcolor="white", fontsize=9)
    _style_ax(ax)
    fig.tight_layout()


def _plot_confiances(view_labels, confidences):
    fig, ax = _new_fig("B – Vues & niveaux de confiance", figsize=(9, 5))
    y_pos = np.arange(len(view_labels))
    cols = [C1 if c >= 0.6 else C3 if c >= 0.45 else C4 for c in confidences]
    ax.barh(y_pos, [c * 100 for c in confidences], color=cols, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(view_labels, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_xlabel("Confiance (%)", color="white")
    ax.set_title("Vues & niveaux de confiance", color="white", fontweight="bold")
    ax.axvline(50, color=GREY, ls="--", lw=0.8)
    _style_ax(ax)
    fig.tight_layout()


def _plot_allocations(tickers, w_mk, w_bl):
    fig, ax = _new_fig("C – Allocations optimales (contraintes)")
    x = np.arange(len(tickers))
    ax.bar(x - 0.2, w_mk * 100, 0.38, label="Markowitz", color=C2, alpha=0.85)
    ax.bar(x + 0.2, w_bl * 100, 0.38, label="BL+Conf", color=C1, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=35, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_ylabel("Poids (%)", color="white")
    ax.set_title("Allocations optimales (contraintes)", color="white", fontweight="bold")
    ax.legend(facecolor=PAN, labelcolor="white", fontsize=9)
    _style_ax(ax)
    fig.tight_layout()


def _plot_frontier_markowitz(v_mk_f, r_mk_f, sh_mk_f, stats_mk):
    fig, ax = _new_fig("D – Frontière efficace Markowitz")
    sc = ax.scatter(v_mk_f * 100, r_mk_f * 100, c=sh_mk_f, cmap="plasma", s=4, alpha=0.5)
    cb = plt.colorbar(sc, ax=ax, label="Sharpe")
    cb.ax.yaxis.set_tick_params(color="white")
    cb.set_label("Sharpe", color="white")
    v_mk, r_mk, sh_mk = stats_mk
    ax.scatter(v_mk * 100, r_mk * 100, s=180, color="white",
               zorder=5, edgecolors=C2, linewidths=2,
               label=f"Tangent (Sh={sh_mk:.2f})")
    ax.set_xlabel("Volatilité ann. (%)", color="white")
    ax.set_ylabel("Rendement ann. (%)", color="white")
    ax.set_title("Frontière efficace – Markowitz", color="white", fontweight="bold")
    ax.legend(facecolor=PAN, labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    _style_ax(ax)
    fig.tight_layout()


def _plot_frontier_bl(v_bl_f, r_bl_f, sh_bl_f, stats_bl):
    fig, ax = _new_fig("E – Frontière efficace BL+Conf")
    sc = ax.scatter(v_bl_f * 100, r_bl_f * 100, c=sh_bl_f, cmap="plasma", s=4, alpha=0.5)
    cb = plt.colorbar(sc, ax=ax, label="Sharpe")
    cb.ax.yaxis.set_tick_params(color="white")
    cb.set_label("Sharpe", color="white")
    v_bl, r_bl, sh_bl = stats_bl
    ax.scatter(v_bl * 100, r_bl * 100, s=180, color="white",
               zorder=5, edgecolors=C1, linewidths=2,
               label=f"Tangent (Sh={sh_bl:.2f})")
    ax.set_xlabel("Volatilité ann. (%)", color="white")
    ax.set_ylabel("Rendement ann. (%)", color="white")
    ax.set_title("Frontière efficace – BL+Conf", color="white", fontweight="bold")
    ax.legend(facecolor=PAN, labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    _style_ax(ax)
    fig.tight_layout()


def _plot_recap(stats_mk, stats_bl):
    fig, ax = _new_fig("F – Récapitulatif des performances", figsize=(7, 4))
    ax.axis("off")
    v_mk, r_mk, sh_mk = stats_mk
    v_bl, r_bl, sh_bl = stats_bl
    rows = [
        ["Markowitz", f"{r_mk:+.2%}", f"{v_mk:.2%}", f"{sh_mk:.2f}"],
        ["BL+Conf", f"{r_bl:+.2%}", f"{v_bl:.2%}", f"{sh_bl:.2f}"],
    ]
    col_labels = ["Stratégie", "Rend.", "Vol.", "Sharpe"]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center",
                   bbox=[0.05, 0.35, 0.9, 0.40])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#252841" if row == 0 else PAN)
        cell.set_text_props(color="white")
        cell.set_edgecolor("#555")
    ax.set_title("Récapitulatif des performances", color="white", fontweight="bold", pad=12)
    ax.text(0.5, 0.1,
            "Contrainte : poids ≥ 0  |  Σwᵢ = 1  |  Secteur ≤ 40 %",
            color=GREY, fontsize=9, ha="center", va="center",
            transform=ax.transAxes)
    fig.tight_layout()


if __name__ == "__main__":
    run()
