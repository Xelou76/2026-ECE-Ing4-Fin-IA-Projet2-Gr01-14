"""
portfolio.py
============
Construction de portefeuille et backtest.

Stratégies implémentées :
  1. Equal Weight         — référence naïve
  2. Markowitz classique  — optimisation moyenne-variance
  3. GNN Portfolio        — poids issus des prédictions GNN
  4. GNN + RL             — allocation par Reinforcement Learning (niveau excellent)

Usage:
    from src.portfolio import PortfolioBuilder, Backtester
    builder   = PortfolioBuilder(returns, predictions)
    weights   = builder.markowitz_weights()
    backtester = Backtester(returns)
    results    = backtester.run_all()
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[portfolio] scipy non disponible — Markowitz désactivé")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Métriques de performance
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(portfolio_returns: pd.Series, rf: float = 0.0) -> dict:
    """
    Calcule les métriques standard de performance d'un portefeuille.

    Paramètres
    ----------
    portfolio_returns : Series — rendements journaliers du portefeuille
    rf               : float  — taux sans risque journalier (défaut 0)

    Retourne
    --------
    dict avec :
      - rendement_annualise : μ × 252
      - volatilite_annualisee : σ × √252
      - sharpe_ratio : (μ - rf) / σ × √252
      - max_drawdown : perte max depuis un pic
      - sortino_ratio : Sharpe avec σ downside seulement
      - calmar_ratio : rendement annualisé / |max_drawdown|

    Note sur le Sharpe Ratio
    ------------------------
        SR = (R_p - R_f) / σ_p × √252
    C'est le rendement excédentaire par unité de risque.
    SR > 1 est considéré bon, > 2 est excellent.
    """
    ann = 252
    excess = portfolio_returns - rf

    mu    = portfolio_returns.mean()
    sigma = portfolio_returns.std()

    # Rendement annualisé
    ann_return = mu * ann

    # Volatilité annualisée
    ann_vol = sigma * np.sqrt(ann)

    # Sharpe Ratio
    sharpe = (excess.mean() / (sigma + 1e-10)) * np.sqrt(ann)

    # Maximum Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns   = (cum_returns - rolling_max) / rolling_max
    max_dd      = drawdowns.min()

    # Sortino Ratio (pénalise uniquement les rendements négatifs)
    downside_returns = portfolio_returns[portfolio_returns < rf]
    downside_std     = downside_returns.std() * np.sqrt(ann)
    sortino = (ann_return - rf * ann) / (downside_std + 1e-10)

    # Calmar Ratio
    calmar = ann_return / (abs(max_dd) + 1e-10)

    return {
        'Rendement annualisé (%)' : round(ann_return * 100, 2),
        'Volatilité annualisée (%)': round(ann_vol * 100, 2),
        'Sharpe Ratio'            : round(sharpe, 3),
        'Max Drawdown (%)'        : round(max_dd * 100, 2),
        'Sortino Ratio'           : round(sortino, 3),
        'Calmar Ratio'            : round(calmar, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Construction de portefeuille
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioBuilder:
    """
    Transforme des prédictions de rendements en poids de portefeuille.

    Paramètres
    ----------
    returns     : pd.DataFrame — rendements historiques (pour Markowitz)
    predictions : ndarray      — scores de rendement prédit par le GNN
    """

    def __init__(
        self,
        returns     : pd.DataFrame,
        predictions : np.ndarray = None,
    ):
        self.returns     = returns
        self.predictions = predictions
        self.n           = returns.shape[1]
        self.tickers     = list(returns.columns)

    # ── 2.1 Equal Weight ────────────────────────────────────────────────────
    def equal_weight(self) -> np.ndarray:
        """
        Portefeuille équipondéré : w_i = 1/N pour tout i.
        Référence naïve mais souvent difficile à battre (DeMiguel et al. 2009).
        """
        return np.ones(self.n) / self.n

    # ── 2.2 Markowitz (Mean-Variance) ───────────────────────────────────────
    def markowitz_weights(
        self,
        expected_returns : np.ndarray = None,
        risk_aversion    : float      = 1.0,
        long_only        : bool       = True,
    ) -> np.ndarray:
        """
        Optimisation moyenne-variance de Markowitz.

        Problème :
            max  w^T μ - (λ/2) w^T Σ w
            s.t. Σ w_i = 1
                 w_i ≥ 0  (si long_only=True)

        où :
            μ  = rendements attendus (prédictions GNN ou moyenne historique)
            Σ  = matrice de covariance estimée
            λ  = aversion au risque

        Note sur l'estimation de Σ
        ---------------------------
        L'estimateur de Ledoit-Wolf est utilisé quand disponible.
        Il réduit l'erreur d'estimation de la covariance en "shrinkant"
        vers la matrice identité (problème classique quand n_actifs > n_jours).

        Paramètres
        ----------
        expected_returns : ndarray (n,) — si None, utilise moyenne historique
        risk_aversion    : float  — λ (plus élevé = plus prudent)
        long_only        : bool   — contrainte de positions longues uniquement
        """
        if not HAS_SCIPY:
            print("[PortfolioBuilder] scipy manquant → fallback Equal Weight")
            return self.equal_weight()

        # Rendements attendus
        if expected_returns is None:
            mu = self.returns.mean().values
        else:
            mu = expected_returns

        # Matrice de covariance (Ledoit-Wolf si sklearn disponible)
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(self.returns.values)
            cov = lw.covariance_
        except ImportError:
            cov = self.returns.cov().values

        # Optimisation
        n = self.n

        def neg_sharpe(w):
            port_return = w @ mu
            port_var    = w @ cov @ w
            return -(port_return - 0.0) / (np.sqrt(port_var) + 1e-10)

        def neg_utility(w):
            return -(w @ mu - (risk_aversion / 2) * (w @ cov @ w))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n if long_only else [(-0.3, 1)] * n

        w0 = np.ones(n) / n  # départ : equal weight

        result = minimize(
            neg_utility, w0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'ftol': 1e-9, 'maxiter': 1000},
        )

        if not result.success:
            print(f"[PortfolioBuilder] Markowitz non convergé : {result.message}")
            return self.equal_weight()

        w = result.x
        w = np.maximum(w, 0)         # sécurité : pas de poids négatifs
        w = w / w.sum()              # renormaliser
        return w.astype(np.float32)

    # ── 2.3 GNN Portfolio ───────────────────────────────────────────────────
    def gnn_softmax_weights(
        self,
        temperature : float = 1.0,
    ) -> np.ndarray:
        """
        Convertit les scores GNN en poids via softmax.

        w_i = exp(score_i / T) / Σ_j exp(score_j / T)

        Paramètres
        ----------
        temperature : float — T (>1 = plus lisse/diversifié, <1 = plus concentré)
        """
        if self.predictions is None:
            raise RuntimeError("Fournir des predictions au constructeur.")

        scores = self.predictions.flatten()
        # Softmax avec température
        scores_t = scores / temperature
        scores_t -= scores_t.max()   # stabilité numérique
        exp_s = np.exp(scores_t)
        w = exp_s / exp_s.sum()
        return w.astype(np.float32)

    def gnn_markowitz_weights(
        self,
        risk_aversion : float = 1.0,
    ) -> np.ndarray:
        """
        Markowitz avec les prédictions GNN comme rendements attendus.
        Combine la structure relationnelle (GNN) et l'optimisation classique.
        """
        if self.predictions is None:
            raise RuntimeError("Fournir des predictions au constructeur.")

        return self.markowitz_weights(
            expected_returns = self.predictions.flatten(),
            risk_aversion    = risk_aversion,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent RL pour allocation de portefeuille (niveau excellent)
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioRLAgent(nn.Module):
    """
    Agent de Reinforcement Learning pour l'allocation de portefeuille.

    Architecture : Policy Network (Actor-Critic simplifié)

    État (state)  : features du GNN (embeddings des nœuds) + poids actuels
    Action        : nouveaux poids de portefeuille (sortie softmax)
    Récompense    : Sharpe ratio sur la période

    Inspiration : Zhang (2026) — framework multi-agent adapté en mono-agent.

    Paramètres
    ----------
    n_assets    : int — nombre d'actifs
    state_dim   : int — dimension de l'état (features GNN + poids)
    hidden      : int — taille des couches cachées
    """

    def __init__(
        self,
        n_assets  : int,
        state_dim : int,
        hidden    : int = 128,
    ):
        super().__init__()
        self.n_assets = n_assets

        # Réseau de politique (policy network)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_assets),
        )

        # Réseau de valeur (value network pour A2C)
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor):
        """
        Paramètres
        ----------
        state : Tensor (batch × state_dim)

        Retourne
        --------
        weights : Tensor (batch × n_assets) — poids softmax
        value   : Tensor (batch × 1)        — estimation de valeur
        """
        logits  = self.policy(state)
        weights = torch.softmax(logits, dim=-1)  # somme à 1, non négatifs
        value   = self.value(state)
        return weights, value


class PortfolioEnv:
    """
    Environnement gym-like pour l'entraînement RL.

    À chaque pas de temps :
      - État  : embeddings GNN normalisés + poids courants
      - Action: nouveaux poids
      - Récompense : Sharpe sur la fenêtre suivante (20 jours)
    """

    def __init__(
        self,
        returns         : pd.DataFrame,
        gnn_embeddings  : np.ndarray,   # (n_jours × n_actifs × embed_dim)
        window          : int   = 20,
        transaction_cost: float = 0.001,
    ):
        self.returns          = returns.values
        self.gnn_embeddings   = gnn_embeddings
        self.window           = window
        self.transaction_cost = transaction_cost
        self.n_assets         = returns.shape[1]
        self.n_steps          = len(returns) - window
        self.reset()

    def reset(self):
        self.t          = self.window
        self.weights    = np.ones(self.n_assets) / self.n_assets
        return self._get_state()

    def _get_state(self):
        """
        État = flatten des embeddings GNN du timestep courant + poids actuels.
        """
        if self.gnn_embeddings.ndim == 3:
            embed = self.gnn_embeddings[self.t].flatten()
        else:
            embed = self.gnn_embeddings.flatten()
        state = np.concatenate([embed, self.weights])
        return state.astype(np.float32)

    def step(self, new_weights: np.ndarray):
        """
        Paramètres
        ----------
        new_weights : ndarray (n_assets,) — poids proposés par l'agent

        Retourne
        --------
        (next_state, reward, done, info)
        """
        # Coût de transaction (tournover × coût)
        turnover = np.abs(new_weights - self.weights).sum()
        tc_cost  = turnover * self.transaction_cost

        # Rendement du portefeuille sur la fenêtre suivante
        if self.t + self.window <= len(self.returns):
            future_returns = self.returns[self.t : self.t + self.window]
            port_returns   = future_returns @ new_weights

            mu    = port_returns.mean()
            sigma = port_returns.std() + 1e-8
            sharpe_reward = (mu / sigma) * np.sqrt(252)
        else:
            sharpe_reward = 0.0

        reward = sharpe_reward - tc_cost * 10  # pénalité de transaction

        self.weights = new_weights
        self.t      += 1
        done         = self.t >= self.n_steps

        return self._get_state(), reward, done, {'sharpe': sharpe_reward}


class RLTrainer:
    """
    Entraîne l'agent RL avec l'algorithme REINFORCE (policy gradient).

    Algorithme (simplifié)
    ----------------------
    Pour chaque épisode :
      1. Jouer un épisode complet avec la politique courante
      2. Calculer les retours cumulés G_t = Σ_{k≥t} γ^{k-t} r_k
      3. Mettre à jour : θ ← θ + α ∇_θ log π(a_t|s_t) G_t
    """

    def __init__(
        self,
        agent    : PortfolioRLAgent,
        env      : PortfolioEnv,
        lr       : float = 3e-4,
        gamma    : float = 0.99,
        device   : str   = None,
    ):
        self.agent  = agent
        self.env    = env
        self.gamma  = gamma
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent  = agent.to(self.device)
        self.opt    = torch.optim.Adam(agent.parameters(), lr=lr)

        self.episode_rewards = []

    def train(self, n_episodes: int = 100, verbose_every: int = 10):
        """Entraînement REINFORCE."""
        print(f"[RLTrainer] Entraînement {n_episodes} épisodes sur {self.device}")

        for ep in range(1, n_episodes + 1):
            state    = self.env.reset()
            states, actions, rewards = [], [], []
            done = False

            while not done:
                s_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                weights, _ = self.agent(s_tensor)
                w = weights.detach().cpu().numpy().flatten()

                next_state, reward, done, _ = self.env.step(w)

                states.append(s_tensor)
                actions.append(weights)
                rewards.append(reward)
                state = next_state

            # Calculer les retours cumulés
            G, returns_list = 0.0, []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns_list.insert(0, G)

            returns_t = torch.tensor(returns_list, dtype=torch.float).to(self.device)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Loss REINFORCE
            loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for w_t, G_t in zip(actions, returns_t):
                log_prob = torch.log(w_t + 1e-8).sum()
                loss     = loss - log_prob * G_t

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.opt.step()

            ep_reward = sum(rewards)
            self.episode_rewards.append(ep_reward)

            if ep % verbose_every == 0:
                avg = np.mean(self.episode_rewards[-verbose_every:])
                print(f"  Épisode {ep:4d}/{n_episodes} | "
                      f"Récompense moy. : {avg:.4f}")

    def get_weights(self, state: np.ndarray) -> np.ndarray:
        """Prédit les poids pour un état donné."""
        self.agent.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            w, _ = self.agent(s)
        return w.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Backtester
# ─────────────────────────────────────────────────────────────────────────────
class Backtester:
    """
    Backteste et compare plusieurs stratégies de portefeuille.

    Paramètres
    ----------
    returns       : pd.DataFrame — rendements journaliers (période de test)
    rebalance_freq: int          — fréquence de rééquilibrage en jours (défaut 20)
    """

    def __init__(
        self,
        returns        : pd.DataFrame,
        rebalance_freq : int = 20,
    ):
        self.returns        = returns
        self.rebalance_freq = rebalance_freq
        self.results        = {}   # nom → Series de rendements du portefeuille

    def add_strategy(
        self,
        name    : str,
        weights : np.ndarray,
    ):
        """
        Ajoute une stratégie avec des poids fixes (pas de rééquilibrage dynamique).

        Note
        ----
        Dans un backtest réel, les poids seraient recalculés à chaque
        période de rééquilibrage. Pour la simplicité de ce projet universitaire,
        on utilise des poids fixes calculés sur la période d'entraînement.
        """
        port_returns = (self.returns * weights).sum(axis=1)
        self.results[name] = port_returns
        return port_returns

    def compute_all_metrics(self) -> pd.DataFrame:
        """
        Calcule toutes les métriques pour toutes les stratégies.

        Retourne
        --------
        DataFrame (stratégies × métriques)
        """
        rows = {}
        for name, port_ret in self.results.items():
            rows[name] = compute_metrics(port_ret)
        return pd.DataFrame(rows).T

    def plot_cumulative_returns(self, figsize=(12, 6)):
        """
        Trace les rendements cumulatifs de toutes les stratégies.
        C'est le graphique principal de la soutenance.
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = {
            'GNN + RL':         '#3266ad',
            'GAT + Markowitz':  '#1D9E75',
            'GCN Softmax':      '#7F77DD',
            'Markowitz':        '#E85D24',
            'Equal Weight':     '#888780',
        }
        styles = {
            'GNN + RL':         '-',
            'GAT + Markowitz':  '-',
            'GCN Softmax':      '--',
            'Markowitz':        '--',
            'Equal Weight':     ':',
        }

        for name, port_ret in self.results.items():
            cum = (1 + port_ret).cumprod()
            color = colors.get(name, '#3266ad')
            ls    = styles.get(name, '-')
            ax.plot(cum.index, cum.values,
                    label=f"{name} ({cum.iloc[-1]:.2f}×)",
                    color=color, linestyle=ls, linewidth=2.0)

        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Valeur du portefeuille (base 1)', fontsize=11)
        ax.set_title('Rendements cumulatifs — comparaison des stratégies',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        plt.tight_layout()
        return fig

    def plot_drawdowns(self, figsize=(12, 5)):
        """Trace les drawdowns pour identifier les périodes de stress."""
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['#3266ad', '#1D9E75', '#7F77DD', '#E85D24', '#888780']

        for (name, port_ret), color in zip(self.results.items(), colors):
            cum = (1 + port_ret).cumprod()
            peak = cum.cummax()
            dd   = (cum - peak) / peak * 100
            ax.fill_between(dd.index, dd.values, 0,
                            alpha=0.3, color=color, label=name)
            ax.plot(dd.index, dd.values, color=color, linewidth=0.8)

        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdowns par stratégie', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig

    def plot_metrics_comparison(self, figsize=(10, 5)):
        """Barplot comparatif des métriques clés."""
        metrics_df = self.compute_all_metrics()

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        cols   = ['Rendement annualisé (%)', 'Volatilité annualisée (%)', 'Sharpe Ratio']
        colors = ['#1D9E75', '#E85D24', '#3266ad']
        titles = ['Rendement annualisé (%)', 'Volatilité annualisée (%)', 'Sharpe Ratio']

        for ax, col, color, title in zip(axes, cols, colors, titles):
            vals = metrics_df[col].astype(float)
            bars = ax.bar(vals.index, vals.values, color=color, alpha=0.8,
                          edgecolor='white', linewidth=0.5)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index, rotation=30, ha='right', fontsize=8)
            ax.grid(True, alpha=0.25, axis='y')

            # Annoter les barres
            for bar, val in zip(bars, vals.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + abs(vals.values).max() * 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Comparaison des stratégies de portefeuille',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
