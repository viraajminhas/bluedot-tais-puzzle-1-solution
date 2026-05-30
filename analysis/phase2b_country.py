"""Phase 2b: characterise *country* representation specifically.

Goals:
  1. Identify which country word is in each F=1 text (build a per-text country tag).
  2. Visualise h2 colour-coded by country identity.
  3. Compute per-country centroids; test "distance-to-nearest-country-centroid"
     as a non-linear discriminator.
  4. Compare F=1 vs F=0 covariance structure to understand the manifold.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_NAMES, get_activations, load_split

FIG_DIR = Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)

COUNTRY_IDX = FEATURE_NAMES.index("country")  # 5


# ---- Country list (compact: ISO short names + common variants) ----------- #
COUNTRY_LIST = [
    # A
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Armenia", "Australia", "Austria", "Azerbaijan",
    # B
    "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Botswana", "Brazil", "Brunei",
    "Bulgaria", "Burkina Faso", "Burundi",
    # C
    "Cambodia", "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia",
    "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia",
    # D-F
    "Denmark", "Djibouti", "Dominica", "Ecuador", "Egypt", "Eritrea",
    "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    # G-I
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
    "Guatemala", "Guinea", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland",
    "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    # J-L
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kosovo",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia",
    "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    # M-O
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "Nicaragua", "Niger", "Nigeria", "Norway",
    "Oman",
    # P-R
    "Pakistan", "Palau", "Panama", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    # S
    "Samoa", "Senegal", "Serbia", "Seychelles", "Singapore", "Slovakia",
    "Slovenia", "Somalia", "Spain", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Syria",
    # T-Z
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga", "Tunisia",
    "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "Uruguay",
    "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia",
    "Zimbabwe",
    # Common variants
    "USA", "UK", "South Korea", "North Korea", "South Africa",
    "United States", "United Kingdom", "Saudi Arabia",
    "New Zealand", "El Salvador", "Sri Lanka",
    "Trinidad and Tobago", "Antigua",
    "Ivory Coast", "Sierra Leone", "Cape Verde", "Marshall Islands",
    "Equatorial Guinea", "Solomon Islands", "Dominican Republic",
    "Papua New Guinea", "Sao Tome", "Timor-Leste",
    "Central African Republic", "Czech Republic",
]
# Sort longer first so multi-word names match before single-word substrings.
COUNTRY_LIST = sorted(set(COUNTRY_LIST), key=lambda s: -len(s))


# ---- Continent groupings for clean visualisation ------------------------- #
CONTINENT = {
    # Africa
    "Algeria": "Africa", "Angola": "Africa", "Benin": "Africa",
    "Botswana": "Africa", "Burkina Faso": "Africa", "Burundi": "Africa",
    "Cameroon": "Africa", "Cape Verde": "Africa", "Central African Republic": "Africa",
    "Chad": "Africa", "Comoros": "Africa", "Congo": "Africa",
    "Djibouti": "Africa", "Egypt": "Africa", "Equatorial Guinea": "Africa",
    "Eritrea": "Africa", "Eswatini": "Africa", "Ethiopia": "Africa",
    "Gabon": "Africa", "Gambia": "Africa", "Ghana": "Africa",
    "Guinea": "Africa", "Ivory Coast": "Africa", "Kenya": "Africa",
    "Lesotho": "Africa", "Liberia": "Africa", "Libya": "Africa",
    "Madagascar": "Africa", "Malawi": "Africa", "Mali": "Africa",
    "Mauritania": "Africa", "Mauritius": "Africa", "Morocco": "Africa",
    "Mozambique": "Africa", "Namibia": "Africa", "Niger": "Africa",
    "Nigeria": "Africa", "Rwanda": "Africa", "Sao Tome": "Africa",
    "Senegal": "Africa", "Seychelles": "Africa", "Sierra Leone": "Africa",
    "Somalia": "Africa", "South Africa": "Africa", "Sudan": "Africa",
    "Tanzania": "Africa", "Togo": "Africa", "Tunisia": "Africa",
    "Uganda": "Africa", "Zambia": "Africa", "Zimbabwe": "Africa",
    # Asia
    "Afghanistan": "Asia", "Armenia": "Asia", "Azerbaijan": "Asia",
    "Bahrain": "Asia", "Bangladesh": "Asia", "Bhutan": "Asia", "Brunei": "Asia",
    "Cambodia": "Asia", "China": "Asia", "Cyprus": "Asia", "Georgia": "Asia",
    "India": "Asia", "Indonesia": "Asia", "Iran": "Asia", "Iraq": "Asia",
    "Israel": "Asia", "Japan": "Asia", "Jordan": "Asia", "Kazakhstan": "Asia",
    "Kuwait": "Asia", "Kyrgyzstan": "Asia", "Laos": "Asia", "Lebanon": "Asia",
    "Malaysia": "Asia", "Maldives": "Asia", "Mongolia": "Asia", "Myanmar": "Asia",
    "Nepal": "Asia", "North Korea": "Asia", "Oman": "Asia", "Pakistan": "Asia",
    "Philippines": "Asia", "Qatar": "Asia", "Saudi Arabia": "Asia",
    "Singapore": "Asia", "South Korea": "Asia", "Sri Lanka": "Asia",
    "Syria": "Asia", "Taiwan": "Asia", "Tajikistan": "Asia", "Thailand": "Asia",
    "Timor-Leste": "Asia", "Turkey": "Asia", "Turkmenistan": "Asia",
    "Uzbekistan": "Asia", "Vietnam": "Asia", "Yemen": "Asia",
    # Europe
    "Albania": "Europe", "Andorra": "Europe", "Austria": "Europe",
    "Belarus": "Europe", "Belgium": "Europe", "Bulgaria": "Europe",
    "Croatia": "Europe", "Czechia": "Europe", "Czech Republic": "Europe",
    "Denmark": "Europe", "Estonia": "Europe", "Finland": "Europe",
    "France": "Europe", "Germany": "Europe", "Greece": "Europe",
    "Hungary": "Europe", "Iceland": "Europe", "Ireland": "Europe",
    "Italy": "Europe", "Kosovo": "Europe", "Latvia": "Europe",
    "Liechtenstein": "Europe", "Lithuania": "Europe", "Luxembourg": "Europe",
    "Malta": "Europe", "Moldova": "Europe", "Monaco": "Europe",
    "Montenegro": "Europe", "Netherlands": "Europe", "Norway": "Europe",
    "Poland": "Europe", "Portugal": "Europe", "Romania": "Europe",
    "Russia": "Europe", "Serbia": "Europe", "Slovakia": "Europe",
    "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe",
    "Switzerland": "Europe", "UK": "Europe", "Ukraine": "Europe",
    "United Kingdom": "Europe",
    # Americas
    "Antigua": "Americas", "Argentina": "Americas", "Bahamas": "Americas",
    "Barbados": "Americas", "Belize": "Americas", "Bolivia": "Americas",
    "Brazil": "Americas", "Canada": "Americas", "Chile": "Americas",
    "Colombia": "Americas", "Costa Rica": "Americas", "Cuba": "Americas",
    "Dominica": "Americas", "Dominican Republic": "Americas", "Ecuador": "Americas",
    "El Salvador": "Americas", "Grenada": "Americas", "Guatemala": "Americas",
    "Guyana": "Americas", "Haiti": "Americas", "Honduras": "Americas",
    "Jamaica": "Americas", "Mexico": "Americas", "Nicaragua": "Americas",
    "Panama": "Americas", "Paraguay": "Americas", "Peru": "Americas",
    "Suriname": "Americas", "Trinidad and Tobago": "Americas",
    "United States": "Americas", "USA": "Americas", "Uruguay": "Americas",
    "Venezuela": "Americas",
    # Oceania
    "Australia": "Oceania", "Fiji": "Oceania", "Kiribati": "Oceania",
    "Marshall Islands": "Oceania", "Micronesia": "Oceania", "Nauru": "Oceania",
    "New Zealand": "Oceania", "Palau": "Oceania", "Papua New Guinea": "Oceania",
    "Samoa": "Oceania", "Solomon Islands": "Oceania", "Tonga": "Oceania",
    "Tuvalu": "Oceania", "Vanuatu": "Oceania",
}


def tag_countries(texts: list[str]) -> list[str | None]:
    """For each text, return the (first) matched country name, else None."""
    tagged = []
    for t in texts:
        match = None
        for c in COUNTRY_LIST:
            if c in t:
                match = c
                break
        tagged.append(match)
    return tagged


def run():
    train = get_activations("train")
    test = get_activations("test")
    texts_tr, _, _ = load_split("train")
    texts_te, _, _ = load_split("test")

    X_tr = train["h2"]
    X_te = test["h2"]
    y_tr = train["labels"][:, COUNTRY_IDX]
    y_te = test["labels"][:, COUNTRY_IDX]

    # ----- 1. Tag countries ------------------------------------------------ #
    print("Tagging countries via string match...")
    country_tr = tag_countries(texts_tr)
    country_te = tag_countries(texts_te)
    n_pos = int(y_tr.sum())
    n_tag = sum(1 for c in country_tr if c is not None)
    n_pos_tag = sum(1 for c, y in zip(country_tr, y_tr) if c is not None and y == 1)
    print(f"  train: country=1 in labels: {n_pos}, country-word found in text: {n_tag}, "
          f"both: {n_pos_tag}")
    counts = Counter([c for c in country_tr if c])
    print(f"  unique countries identified: {len(counts)}; top 10: {counts.most_common(10)}")

    # Examples where we *missed* tagging despite y=1 (so we can spot gaps in our list).
    missed = [t for c, y, t in zip(country_tr, y_tr, texts_tr) if y == 1 and c is None]
    print(f"  missed (y=1 but no country tag): {len(missed)} examples; sample:")
    for ex in missed[:5]:
        print(f"     - {ex[:120]}")

    # ----- 2. PCA on F=1 examples, colour by continent -------------------- #
    Xp_tr = X_tr[y_tr == 1]
    Xn_tr = X_tr[y_tr == 0]
    mu = X_tr.mean(0)
    pca = PCA(n_components=4).fit(Xp_tr - mu)
    P_p = pca.transform(Xp_tr - mu)
    P_n = pca.transform(Xn_tr - mu)
    countries_p = [c for c, y in zip(country_tr, y_tr) if y == 1]
    continents_p = [CONTINENT.get(c, "Unknown") for c in countries_p]
    print(f"  continent distribution among F=1 train: {Counter(continents_p)}")

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 2a. PC1 vs PC2 colour by continent
    ax = axs[0, 0]
    cont_colors = {"Africa": "#e15759", "Asia": "#4e79a7", "Europe": "#59a14f",
                   "Americas": "#f28e2b", "Oceania": "#76b7b2", "Unknown": "#bab0ac"}
    ax.scatter(P_n[:, 0], P_n[:, 1], s=2, alpha=0.15, c="lightgrey",
               label="country=0")
    for cont in ["Africa", "Asia", "Europe", "Americas", "Oceania"]:
        mask = np.array([cn == cont for cn in continents_p])
        if mask.any():
            ax.scatter(P_p[mask, 0], P_p[mask, 1], s=7, alpha=0.55,
                       c=cont_colors[cont], label=cont)
    ax.set_title("h2: PC1 vs PC2 of country=1, coloured by continent")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")

    # 2b. PC1 vs PC3
    ax = axs[0, 1]
    ax.scatter(P_n[:, 0], P_n[:, 2], s=2, alpha=0.15, c="lightgrey")
    for cont in ["Africa", "Asia", "Europe", "Americas", "Oceania"]:
        mask = np.array([cn == cont for cn in continents_p])
        if mask.any():
            ax.scatter(P_p[mask, 0], P_p[mask, 2], s=7, alpha=0.55,
                       c=cont_colors[cont], label=cont)
    ax.set_title("h2: PC1 vs PC3 of country=1, coloured by continent")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC3"); ax.legend(loc="best", fontsize=8)

    # 2c. PC2 vs PC3
    ax = axs[1, 0]
    ax.scatter(P_n[:, 1], P_n[:, 2], s=2, alpha=0.15, c="lightgrey")
    for cont in ["Africa", "Asia", "Europe", "Americas", "Oceania"]:
        mask = np.array([cn == cont for cn in continents_p])
        if mask.any():
            ax.scatter(P_p[mask, 1], P_p[mask, 2], s=7, alpha=0.55,
                       c=cont_colors[cont], label=cont)
    ax.set_title("h2: PC2 vs PC3 of country=1, coloured by continent")
    ax.set_xlabel("PC2"); ax.set_ylabel("PC3"); ax.legend(loc="best", fontsize=8)

    # 2d. Per-country centroids (top 30 countries) on PC1/PC2 with labels
    ax = axs[1, 1]
    top_countries = [c for c, _ in counts.most_common(30)]
    for c in top_countries:
        mask = np.array([cn == c for cn in countries_p])
        if mask.sum() < 5:
            continue
        cx, cy = P_p[mask, 0].mean(), P_p[mask, 1].mean()
        cont = CONTINENT.get(c, "Unknown")
        ax.scatter([cx], [cy], s=40, c=cont_colors[cont])
        ax.annotate(c, (cx, cy), fontsize=8, alpha=0.85)
    ax.set_title("Per-country centroids in PC1/PC2 (top 30 by frequency)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "country_by_continent.png", dpi=130)
    plt.close(fig)
    print(f"  saved -> {FIG_DIR / 'country_by_continent.png'}")

    # ----- 3. Distance-to-nearest-country-centroid as nonlinear probe ----- #
    print("\n[Distance-to-prototype probe]")
    centroids = {}
    for c in counts.keys():
        mask_tr = np.array([cn == c for cn in country_tr])
        if mask_tr.sum() < 5:
            continue
        centroids[c] = X_tr[mask_tr].mean(0)
    proto_matrix = np.stack(list(centroids.values()))  # (K, 64)

    def min_dist(X):
        # ||x - μ_k||^2 for each k; take min.
        d = ((X[:, None, :] - proto_matrix[None, :, :]) ** 2).sum(-1)
        return d.min(axis=1)

    feat_tr = -min_dist(X_tr).reshape(-1, 1)  # negate so larger = more "country-like"
    feat_te = -min_dist(X_te).reshape(-1, 1)
    clf = LogisticRegression(max_iter=2000).fit(feat_tr, y_tr)
    acc = clf.score(feat_te, y_te)
    print(f"   min-distance-to-any-country-centroid linear probe: acc {acc:.4f}")

    # ----- 4. Covariance structure F=1 vs F=0 ------------------------------ #
    print("\n[Covariance structure]")
    cov1 = np.cov(Xp_tr.T)
    cov0 = np.cov(Xn_tr.T)
    print(f"   trace(cov F=1)={cov1.trace():.3f}  trace(cov F=0)={cov0.trace():.3f}")
    print(f"   det rank effective (cov F=1) — top 5 eigvals: "
          f"{np.sort(np.linalg.eigvalsh(cov1))[::-1][:5]}")
    print(f"   det rank effective (cov F=0) — top 5 eigvals: "
          f"{np.sort(np.linalg.eigvalsh(cov0))[::-1][:5]}")
    print(f"   norm mean h2 vector F=1: {np.linalg.norm(Xp_tr.mean(0)):.4f}")
    print(f"   norm mean h2 vector F=0: {np.linalg.norm(Xn_tr.mean(0)):.4f}")
    print(f"   ||mean_F=1 - mean_F=0||: {np.linalg.norm(Xp_tr.mean(0) - Xn_tr.mean(0)):.4f}")


if __name__ == "__main__":
    run()
