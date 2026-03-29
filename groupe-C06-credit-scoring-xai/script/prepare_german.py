import pandas as pd

columns = [
    "statut_compte", "duree_mois", "historique_credit", "objet_credit",
    "montant_credit", "epargne", "anciennete_emploi", "taux_versement",
    "statut_civil_sexe", "autres_debiteurs", "anciennete_residence",
    "propriete", "age", "autres_credits", "logement", "nb_credits",
    "emploi", "nb_personnes_charge", "telephone", "travailleur_etranger",
    "defaut"
]

df = pd.read_csv("data/raw/german_credit/german.data", sep=r"\s+", header=None)
df.columns = columns

df.to_csv("data/raw/german_credit/german_credit.csv", index=False)

print("German Credit converti en CSV propre (FR + defaut)")