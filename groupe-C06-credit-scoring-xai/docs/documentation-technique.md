# Documentation technique : Credit Scoring avec IA Explicable (XAI)
# Groupe C.06 Elsa Bodenan & Shaïli Tuil
## 1. Introduction et contexte
Dans le secteur financier, accorder ou refuser un crédit à un client est une décision lourde de conséquences. Aujourd'hui, cette décision est de plus en plus souvent assistée voire automatisée par des algorithmes de Machine Learning. Ces modèles analysent des dizaines de variables en quelques millisecondes pour produire un score de risque : c'est ce qu'on appelle le scoring de crédit.
Le problème principal est que les modèles les plus performants comme XGBoost ou LightGBM sont des "boîtes noires" : ils produisent un résultat, mais on ne comprend pas toujours pourquoi. Or, le droit européen, et notamment l'article 22 du RGPD, impose que toute décision automatisée significative sur une personne doit pouvoir être expliquée. Si une banque refuse un crédit à cause d'un algorithme, elle doit être capable de dire pourquoi.
 
Ce projet répond à cet enjeu en construisant un système complet de scoring de crédit qui concilie deux objectifs souvent difficiles à atteindre ensemble : être performant et compréhensible. Pour cela, nous avons :
 
*Entraîné et comparé trois modèles de Machine Learning (régression logistique, XGBoost, LightGBM)
* Analysé leurs décisions grâce à SHAP et LIME (méthodes d'IA explicable)
* Généré des explications contrefactuelles ("que faudrait-il changer pour inverser la décision ?")
* Étudié les potentiels biais du modèle (analyse de fairness sur l'âge et le genre)
* Développé un dashboard interactif Streamlit permettant d'explorer tous ces résultats

## 2. Introduction et contexte
Deux datasets ont été utilisés dans ce projet afin de couvrir à la fois un cadre académique et un cas plus réaliste.

### 2.1 German Credit Dataset

Le German Credit Dataset est un dataset classique très utilisé en recherche. Il contient 1 000 individus décrits par 20 variables socio-économiques : âge, durée du crédit, montant emprunté, historique bancaire, situation professionnelle, épargne, etc. La variable cible est binaire : bon payeur (0) ou défaut (1). Le taux de défaut est de 30 %.
 
Pour enrichir les données, nous avons créé trois nouvelles variables :

* charge_mensuelle : montant du crédit / durée, pour estimer la charge financière mensuelle réelle
* jeune_emprunteur : variable binaire, 1 si la personne a moins de 30 ans
* credit_long : variable binaire, 1 si la durée du crédit dépasse 24 mois

https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

### 2.2  Lending Club Dataset
Le Lending Club Dataset, quant à lui,provient d'une vraie plateforme de prêts. Il est beaucoup plus grand (réduit à 50 000 observations) et plus complexe. Il contient des variables financières détaillées : revenu annuel, taux d'intérêt, ratio d'endettement (DTI), grade de risque attribué par la plateforme, etc. La cible est construite à partir du statut du prêt : "Charged Off" = défaut (1), "Fully Paid" = non-défaut (0).
https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv

### 2.3  Préparation des données
Les deux datasets ont été nettoyés et transformés en plusieurs étapes :

* Valeurs manquantes : médiane pour les variables numériques, valeur la plus fréquente pour les variables catégorielles
* Encodage : les variables textuelles ont été converties en nombres avec un OrdinalEncoder
* Normalisation : toutes les variables numériques ont été mises à la même échelle (StandardScaler) pour éviter qu'une variable domine les autres
* Découpage train/test : 80 % pour l'entraînement, 20 % pour l'évaluation (stratifié sur la variable cible)

## 3. Pipeline du projet 
Le projet suit un pipeline classique en Machine Learning, structuré en plusieurs étapes. Les données sont d’abord nettoyées et préparées, notamment par la gestion des valeurs manquantes, l’encodage des variables catégorielles et la normalisation des variables numériques. Cette étape permet de transformer les données brutes en un format exploitable par les modèles.
Ensuite, plusieurs modèles sont entraînés et comparés. Une fois le modèle optimal sélectionné, différentes méthodes d’explicabilité sont appliquées afin de comprendre ses décisions. Enfin, une analyse de fairness est réalisée pour détecter d’éventuels biais, et un dashboard interactif permet de visualiser l’ensemble des résultats.

## 4. Modélisation
Trois modèles ont été étudiés dans ce projet.

* Régression Logistique : modèle simple et directement interprétable. Utilisé comme baseline pour juger si les modèles plus complexes apportent réellement un gain.
* XGBoost : modèle à base d'arbres très performant. Il construit des centaines d'arbres qui se corrigent mutuellement (boosting). C'est notre modèle principal. 
* LightGBM : variante d'XGBoost plus rapide, efficace sur de grands volumes. Il utilise une stratégie de croissance des arbres différente (leaf-wise).

Le choix du meilleur modèle repose sur plusieurs métriques complémentaires, notamment 
*  AUC-ROC : capacité à distinguer bons payeurs et défauts (1 = parfait, 0.5 = aléatoire)
* F1-Score : équilibre entre précision et rappel, adapté aux classes déséquilibrées
* KS Statistic : écart maximal entre les distributions de scores des deux classes
* Accuracy : taux de bonnes prédictions global

## 5. Explicabilité du modèle
L’explicabilité constitue un élément central du projet. La méthode SHAP est utilisée pour analyser les contributions des variables, à la fois au niveau global et au niveau individuel. Basée sur la théorie des jeux, elle permet d’attribuer à chaque variable une contribution cohérente à la prédiction du modèle, garantissant ainsi une interprétation fiable.

En complément, la méthode LIME est utilisée pour produire des explications locales. Contrairement à SHAP, LIME repose sur une approximation locale du modèle et ne dépend pas de sa structure interne. Cela permet de comparer deux approches différentes de l’explicabilité et d’évaluer leur cohérence.

## 6. Résultats:  German Dataset
Voici les résultats obtenus sur le jeu de test (200 observations) :

| Modèle                 | AUC-ROC | F1-Score | Accuracy | KS Stat |
|------------------------|--------:|---------:|---------:|--------:|
| XGBoost                | 0.810   | 0.593    | 0.780    | 0.469   |
| LightGBM               | 0.790   | 0.537    | 0.750    | —       |
| Régression Logistique  | 0.786   | 0.606    | 0.720    | —       |
<small>Figure 1 : Comparaison des performances sur le jeu de test (200 observations).</small>

XGBoost présente la meilleure capacité de discrimination (AUC-ROC = 0.810), ce qui indique qu’il classe efficacement les bons payeurs et les défauts.
Toutefois, la régression logistique obtient un F1-score légèrement supérieur, ce qui signifie qu’elle détecte davantage de défauts au prix d’un nombre plus élevé de faux positifs.

Ce résultat met en évidence un trade-off classique en credit scoring :
maximiser la détection des défauts (réduction du risque)
ou limiter les refus injustifiés (expérience client)
Le choix du modèle dépend donc du coût métier associé aux erreurs, et non uniquement des performances globales.


![fig_model_comparison.png](..%2Fresults%2Ffig_model_comparison.png)
<small>Figure 2 : Courbes ROC, Precision-Recall et comparaison globale des métriques. XGBoost domine sur AUC-ROC.</small>

![fig_confusion_matrices.png](..%2Fresults%2Ffig_confusion_matrices.png)
<small>Figure 3 : Matrices de confusion des trois modèles. XGBoost : 125 vrais bons payeurs et 31 vrais défauts correctement identifiés.</small>


Les matrices de confusion confirment qu'XGBoost est le modèle le plus équilibré : il identifie correctement 125 bons payeurs sur 140, et 31 défauts sur 60. La régression logistique détecte plus de défauts (34) mais génère aussi plus de fausses alarmes.

Dans un contexte bancaire, une erreur de type faux négatif (accorder un crédit à un client risqué) est généralement plus coûteuse qu’un faux positif (refuser un bon client).
Ainsi, même si XGBoost est globalement plus équilibré, un modèle comme la régression logistique pourrait être préféré dans une stratégie plus conservatrice.

![fig_ks_chart.png](..%2Fresults%2Ffig_ks_chart.png)
<small>Figure 4 : KS Chart pour XGBoost. La statistique KS = 0.469 indique une très bonne séparation des deux populations.</small>

Le KS Chart illustre graphiquement la capacité de discrimination du modèle. La courbe rouge (vrais positifs) monte rapidement bien au-dessus de la droite bleue (faux positifs), et l'écart maximal de 0.469 est atteint tôt. Cette valeur est très bonne pour un problème de scoring de crédit.

![fig_feature_importance.png](..%2Fresults%2Ffig_feature_importance.png)
<small>Figure 5 : Importance des variables selon XGBoost (gauche) et LightGBM (droite). Le statut du compte domine dans les deux cas.</small>

L'importance des variables montre que statut_compte est la variable la plus influente pour XGBoost (score ~0.12, très loin devant les autres). LightGBM met davantage en avant montant_credit et charge_mensuelle, mais les deux s'accordent sur l'importance centrale de l'état du compte bancaire.

## 7. Explicabilité du modèle (XAI)
Une fois le modèle entraîné, une question essentielle reste : pourquoi prédit-il ce résultat ? C'est là qu'intervient l'Intelligence Artificielle Explicable. Nous avons utilisé deux méthodes complémentaires pour répondre à cette question.
### 7.1  SHAP
SHAP est une méthode issue de la théorie des jeux coopératifs. L'idée est d'attribuer à chaque variable une contribution précise à la prédiction du modèle. Une contribution positive pousse vers le défaut, une contribution négative pousse vers le bon payeur. L'avantage majeur de SHAP est qu'il est mathématiquement rigoureux : la somme de toutes les contributions SHAP explique exactement la prédiction finale.

#### Importance globale des variables
![shap_bar.png](..%2Fresults%2Fshap_bar.png)
<small>Figure 6 : Importance SHAP moyenne sur le jeu de test. statut_compte domine largement, suivi de duree_mois et montant_credit.</small>

#### Impact détaillé 
Le graphique confirme que statut_compte est la variable la plus influente, avec une contribution moyenne bien supérieure à toutes les autres (~0.68 contre ~0.30 pour la deuxième). Viennent ensuite duree_mois, montant_credit et charge_mensuelle. 

Ces résultats sont cohérents avec la logique financière : l'état du compte bancaire d'un client est le meilleur indicateur de sa capacité à rembourser.
Cette hiérarchie des variables est cohérente avec les pratiques bancaires traditionnelles, ce qui renforce la crédibilité du modèle.
Un modèle performant mais incohérent avec l’intuition métier serait difficilement déployable en production.

![shap_beeswarm.png](..%2Fresults%2Fshap_beeswarm.png)
<small>Figure 7 :  SHAP Beeswarm : chaque point = un individu. Couleur rouge = valeur élevée de la variable, bleu = valeur faible.</small>

Ce graphique apporte une information supplémentaire par rapport au bar chart : non seulement quelles variables comptent, mais aussi dans quel sens. Pour statut_compte : les valeurs élevées (rouge, compte dégradé) poussent fortement vers le défaut (à gauche), et les valeurs basses (bleu, bon compte) protègent. Pour duree_mois : les longues durées augmentent le risque. La forte dispersion horizontale de certaines variables montre que le modèle capte des relations non linéaires complexes.

##### Graphiques de dépendance SHAP
![shap_scatter.png](..%2Fresults%2Fshap_scatter.png)
<small>Figure 8 : Dépendance SHAP pour statut_compte (gauche) et duree_mois (droite). On voit clairement l'effet de chaque variable sur la prédiction.</small>

Pour statut_compte : la relation est très claire et presque discontinue entre les catégories. La catégorie 3 (situation la plus défavorable) est associée à des contributions SHAP très négatives (forte augmentation du risque). Pour duree_mois : les durées courtes (valeurs normalisées négatives) ont des contributions légèrement positives (bon payeur), tandis que les durées longues poussent vers le défaut.
 
##### Explications individuelles
Le graphique waterfall décompose la prédiction pour un client donné, variable par variable. C'est l'outil le plus lisible pour expliquer une décision individuelle.
Ce type d’explication est particulièrement important dans le cadre du RGPD, car il permet de fournir une justification individuelle claire et traçable pour chaque décision.

![shap_waterfall_accepte.png](..%2Fresults%2Fgerman%2Fshap_waterfall_accepte.png)
<small>Figure 9 : Profil accepté (score 0.915). Le statut_compte (+1.02) et l'historique_credit (+0.70) sont les facteurs favorables dominants.</small>

Profil accepté (score : 0.915). Ce client est très clairement prédit comme bon payeur. Quasiment toutes ses contributions SHAP sont positives (en rouge) : un bon statut_compte (+1.02), un historique_credit favorable (+0.70), une propriété rassurante (+0.58). Le modèle est très confiant dans cette décision.

![shap_waterfall_refuse.png](..%2Fresults%2Fgerman%2Fshap_waterfall_refuse.png)
<small>Figure 10 : Profil refusé (score 0.004). La durée du crédit (-1.25) et la charge mensuelle (-0.98) sont les principaux facteurs défavorables.</small>

Profil refusé (score : 0.004). Ce client présente un risque de défaut très élevé. La durée du crédit trop longue (-1.25), la charge mensuelle importante (-0.98) et un mauvais statut de compte (-0.66) le pénalisent massivement. Ce profil illustre parfaitement la transparence que permet SHAP : on sait exactement pourquoi ce crédit serait refusé.

![shap_waterfall_borderline.png](..%2Fresults%2Fgerman%2Fshap_waterfall_borderline.png)
<small>Figure 11 : Profil borderline (score 0.506). Contributions positives et négatives presque équilibrées, décision incertaine.</small>

 Profil borderline (score : 0.506). Ce cas est le plus intéressant : avec un score à la limite, les contributions s'équilibrent. Le bon statut_compte (+0.80) et l'épargne (+0.24) jouent en sa faveur, mais une certaine ancienneté d'emploi et d'autres facteurs négatifs compensent. Ce type de profil nécessite une validation humaine supplémentaire avant toute décision.
 
### 7.2  LIME
LIME fonctionne très différemment de SHAP. Au lieu de calculer des contributions exactes, il crée de nombreuses variations artificielles autour d'un individu, puis entraîne un modèle simple (régression linéaire) sur ces variations pour approximer le comportement local du modèle. L'avantage de LIME est qu'il est indépendant du modèle utilisé,  il peut s'appliquer à n'importe quelle boîte noire. L'inconvénient est qu'il s'agit d'une approximation, pas d'un calcul exact.

![lime_exemple.png](..%2Fresults%2Fgerman%2Flime_exemple.png)
<small>Figure 12 : Explication LIME pour un profil exemple (prob=0.124). Barres vertes = facteurs favorables, barres rouges = facteurs défavorables.</small>

Pour ce profil (probabilité de défaut = 0.124), LIME identifie que historique_credit favorable est le principal facteur protecteur (barre verte dominante), tandis que certains intervalles d'ancienneté_emploi et l'objet_credit jouent défavorablement. LIME exprime ses résultats sous forme de règles lisibles ("si la variable X est dans cet intervalle, alors..."), ce qui les rend très accessibles pour des conseillers bancaires non-techniques.

### 7.3  Comparaison SHAP vs LIME
Nous avons comparé les deux méthodes sur 10 profils en regardant si elles s'accordent sur les 3 variables les plus importantes ("Top-3 concordance") :

![shap_lime_concordance.png](..%2Fresults%2Fgerman%2Fshap_lime_concordance.png)
<small>Figure 13 : Concordance SHAP vs LIME sur 10 profils. La moyenne est de 0.00, largement en dessous du seuil de 67%.</small>

Le résultat est particulièrement marquant : aucune concordance n’est observée entre SHAP et LIME sur les variables les plus importantes.
Cette divergence soulève une question critique : peut-on réellement faire confiance aux explications fournies par certaines méthodes XAI ?
SHAP, basé sur des fondements théoriques solides (valeurs de Shapley), fournit des explications cohérentes et stables.

À l’inverse, LIME repose sur une approximation locale qui peut être instable et sensible aux perturbations des données.
Dans un contexte réglementé comme le crédit, cette instabilité constitue un risque majeur, car elle peut conduire à des explications incohérentes d’un client à l’autre.
Ce résultat met en évidence que toutes les méthodes d’IA explicable ne se valent pas, et que leur choix doit être fait avec prudence.

## 8. Explications contrefactuelles
Les explications contrefactuelles répondent à la question la plus concrète du point de vue client : "Que devrait-il changer pour obtenir le crédit ?" C'est une exigence directe du RGPD : les décisions automatisées doivent être non seulement compréhensibles, mais aussi actionnables.
 
La logique est simple : on part du profil d'un client refusé et on cherche les modifications minimales qui permettraient au modèle de changer sa décision en sa faveur. Ces modifications doivent être réalistes et concrètes pour être utiles.
 
Par exemple, pour un client refusé à cause d'une durée de crédit trop longue et d'une charge mensuelle élevée, les suggestions pourraient être :

* Réduire la durée du crédit de 48 à 24 mois
* Demander un montant moins élevé pour diminuer la charge mensuelle
* Améliorer l'état de son compte courant avant de faire la demande

Ces recommandations sont précieuses car elles donnent au client des leviers d'action concrets pour améliorer son dossier, et à la banque un outil pour justifier ses décisions de manière transparente et documentée. 

Les contrefactuels présentés ici reposent sur une logique d’ajustement des variables les plus influentes identifiées par SHAP. Toutefois, une limite importante est que ces recommandations ne prennent pas explicitement en compte les contraintes réalistes du client (revenu, situation personnelle, faisabilité économique). Des approches plus avancées, comme DiCE, permettent de générer des contrefactuels optimisés sous contraintes, ce qui constitue une piste d’amélioration majeure pour rendre ces recommandations réellement opérationnelles.

Note : sur le German Credit Dataset, les profils testés avaient des scores faibles mais pas extrêmes. Les cas les plus intéressants sont les profils borderline, pour lesquels de petits ajustements suffisent à faire basculer la décision, et où les contrefactuels ont le plus de valeur pratique.

## 9. Analyse de fairness
Un modèle très performant en moyenne peut néanmoins être injuste envers certains groupes. Par exemple, il pourrait systématiquement pénaliser les jeunes ou les femmes, non pas parce qu'ils présentent réellement plus de risque, mais parce que ces caractéristiques sont corrélées à d'autres variables dans les données historiques données elles-mêmes potentiellement biaisées.
 
En France et en Europe, discriminer un client sur la base de l'âge ou du genre pour l'attribution d'un crédit est illégal. L'audit de fairness permet de vérifier que le modèle respecte ces contraintes réglementaires.

#### Variables sensibles étudiées
•       Âge : groupe "jeune" (moins de 30 ans) vs groupe "senior" (30 ans et plus)

•       Genre (statut_civil_sexe) : cette variable encode le sexe et la situation matrimoniale, et permet d'étudier d'éventuelles disparités
#### Métriques de fairness utilisées
•       Demographic Parity : vérifie si le taux de prédictions favorables est identique entre les groupes. Un écart important signifie que le modèle favorise systématiquement un groupe par rapport à un autre.

•       Equalized Odds : vérifie si le taux de vrais positifs (rappel) et le taux de faux positifs sont égaux entre les groupes. C'est une mesure plus fine et plus exigeante que la parité démographique.

 
Cette analyse est essentielle pour valider que le modèle peut être déployé de manière responsable. Elle permet d'identifier les biais, de les quantifier, et de prendre des décisions éclairées sur la manière de les corriger, par exemple en ajustant les seuils de décision par groupe ou en rééquilibrant les données d'entraînement.

## 10. Dashboard interactif
Pour rendre ce projet concret et utilisable en pratique, nous avons développé un dashboard interactif avec Streamlit. C'est une interface web qui permet d'explorer tous les résultats du projet sans écrire une seule ligne de code.

![Interface.jpg](..%2Fresults%2FInterface.jpg)
<small>Figure 14 : Page d'accueil du dashboard : métriques clés du modèle XGBoost sur le dataset German (AUC-ROC : 0.810, F1-Score : 0.593, Accuracy : 0.780, KS : 0.474).</small>

#### Fonctionnalités principales
* Vue d'ensemble : métriques de performance, courbe ROC, sélection du dataset (German ou Lending Club)
* Explicabilité SHAP : graphiques bar, beeswarm, dépendance et waterfall pour les profils individuels 
* LIME & Comparaison : explication LIME pour un profil sélectionné, graphique de concordance SHAP/LIME
* Contrefactuels : suggestions de modifications pour les profils à risque
* Audit Fairness : métriques d'équité par groupe (âge, genre), visualisation des écarts
* Scoring Individuel : simulation en temps réel du score d'un nouveau client via des curseurs interactifs

#### Lancement
streamlit run src/dashboard.py
 
Une fois lancé, l'interface est accessible dans le navigateur. On peut choisir le dataset depuis le menu déroulant en haut à gauche, et le dashboard se met à jour automatiquement. La section Scoring Individuel est particulièrement utile : elle permet à un conseiller bancaire de modifier les caractéristiques d'un client et de voir immédiatement comment cela affecte son score de risque. 
Ce type d’outil permet de réduire l’asymétrie d’information entre le modèle et l’utilisateur, et favorise une prise de décision hybride homme-machine, essentielle dans les systèmes critiques.

## 11. Conformité RGPD
Notre projet répond concrètement à ces trois exigences :
* Explications SHAP et LIME : pour chaque client, une décomposition précise de l'impact de chaque variable est disponible, en termes compréhensibles pour un non-spécialiste
* Contrefactuels : les suggestions actionnables permettent de dire exactement ce que le client doit modifier pour changer la décision — c'est la version opérationnelle du droit à l'explication
* Audit de fairness : l'analyse d'équité vérifie l'absence de discrimination illégale, notamment sur l'âge et le genre
* Dashboard : l'interface permet à un conseiller humain de comprendre, surveiller et si nécessaire corriger les décisions du modèle, répondant à l'exigence d'intervention humaine

## 12. Structure du projet
Le projet est organisé de manière modulaire : chaque fichier Python a une responsabilité unique et bien définie.


| Fichier                 | Rôle                                                                    |
|------------------------|:------------------------------------------------------------------------|
| exploration.py                | Chargement, nettoyage, transformation des données, découpage train/test |
| modelisation.py               | Entraînement des 3 modèles, calcul des métriques, sélection du meilleur                                                                        |
| explicabilite.py  | SHAP, LIME, waterfall, contrefactuels, concordance SHAP/LIME                                                                   |
|     equite.py                   |          Analyse de fairness par groupe sensible (âge, genre)                                                               |
|    dashboard.py                             |                      Interface Streamlit interactive, visualisations et scoring temps réel<br/>                                                   |

<small>Figure 15 : Description des modules du projet.</small>

 
Les données brutes sont stockées dans data/raw/, les données traitées dans data/processed/, les modèles dans models/ et les résultats dans results/. Cette organisation permet de reproduire n'importe quelle étape du pipeline de façon indépendante.
 
Commandes pour exécuter le pipeline complet :
* python src/exploration.py --dataset german
* python src/modelisation.py --dataset german
* python src/explicabilite.py --dataset german
* streamlit run src/dashboard.py

## 13. Perspectives d’amélioration et limites du projet 
Malgré des résultats encourageants, ce projet présente plusieurs limites.

* L’évaluation des modèles repose sur un simple découpage train/test, sans validation croisée, ce qui limite la robustesse des performances observées
* Le dataset German est de taille relativement réduite (1000 observations), ce qui peut limiter la généralisation des résultats
* Certaines méthodes d’explicabilité, comme LIME, peuvent être instables et produire des résultats variables selon les perturbations
* L’analyse de fairness se concentre uniquement sur l’âge et le genre, alors que d’autres variables pourraient introduire des biais indirects

Ces limites ouvrent des perspectives d’amélioration pour renforcer la robustesse, la fiabilité et la validité du système.

De plus à notre échelle plusieurs pistes d'amélioration peuvent être envisagées pour aller plus loin :
* Intégrer des modèles intrinsèquement interprétables comme les Explainable Boosting Machines (EBM)
* Améliorer la fairness via des techniques d'optimisation des seuils de décision par groupe
* Étendre les contrefactuels avec des bibliothèques spécialisées comme DiCE ou Alibi
* Déployer le dashboard en production pour une utilisation réelle par des conseillers bancaires

## 14. Conclusion
Ce projet met en évidence qu’un modèle performant ne suffit pas dans un contexte critique comme le crédit.

La transparence, la robustesse et l’équité sont des exigences tout aussi essentielles que la performance prédictive.

L’analyse conjointe de SHAP, LIME et des métriques de fairness montre que l’explicabilité reste un domaine complexe, où les méthodes peuvent produire des résultats divergents.

Ainsi, l’IA explicable ne doit pas être considérée comme une simple couche d’interprétation, mais comme un élément central de la conception du modèle.

Enfin, ce projet souligne que le déploiement de systèmes d’IA dans des contextes sensibles nécessite une approche globale intégrant performance, explicabilité, conformité réglementaire et supervision humaine.

