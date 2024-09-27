# =============================================================================
# SECTION: Librairies
# =============================================================================
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.stats as stats
from scipy.stats import t

import statsmodels.api as sm # Pour ANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM

import seaborn as sns
import pandas as pd
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import os
import re

# Définir le répertoire de travail au dossier du script
script_dir = os.path.dirname(__file__)  # Obtient le dossier où se trouve le script
os.chdir(script_dir)  # Change le répertoire de travail au dossier du script

bdd = pd.read_csv('BDD_virgules.csv', sep=';', engine='python')
#mesures_vitesses = pd.read_csv('Mesures_vitesses.csv', sep=';', engine='python')
st.set_page_config(page_title="Patinage de vitesse courte piste", page_icon="⛸️") # Set the page title and icon using st.set_page_config

# =============================================================================
# METHODES PAGE ACCUEIL: 
# =============================================================================
def display_fall_statistics(bdd):
        # Filtrer les données pour compter seulement les lignes où une chute est survenue
        fall_data = bdd[bdd['Fall / Out of track'] == 'Fall']

        # Compter le nombre de chutes par pays
        fall_counts = fall_data['Country'].value_counts().reset_index()
        fall_counts.columns = ['Country', 'Number of Falls']

        # Créer un graphique à barres pour le nombre de chutes par pays
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Number of Falls', y='Country', data=fall_counts, palette='viridis')
        plt.title('Nombre de chutes par pays au cours des deux saisons de compétitions internationales 2022/2023 - 2021/2022')
        plt.xlabel('Nombre de chutes')
        plt.ylabel('Pays')
        st.pyplot(plt)

# =============================================================================
# PAGE: Accueil
# =============================================================================
def page_home():
    st.title('Analyse de vitesse de patineurs de vitesse courte piste') # Main title
    image_path = "./PVCP_couverture.jpg"# Main image 
    st.image(image_path) # Display the image with a caption
    st.header("⛸️ Le patinage de vitesse courte piste")
    st.write("""Le patinage de vitesse courte piste est une discipline qui remonte au début du XXème siècle en 
                Amérique du Nord. Les premières compétitions internationales ont eu lieu dans les années 1970, et 
                la discipline a été ajoutée au programme des Jeux Olympiques en 1992. Dans les premières années, 
                le Canada et les États-Unis dominaient le palmarès, mais aujourd’hui des pays comme les Pays-Bas, 
                l’Italie et la Corée du Sud s’imposent dans les classements. Cette discipline se distingue du patinage 
                de vitesse longue piste par des virages plus serrés, des lignes droites plus courtes et un départ groupé
                [4]. 
                Le patinage de vitesse courte piste se déroule sur une piste ovale de 111.12m dessinée sur une 
                patinoire standard de hockey. Les lignes de départ et d’arrivée sont marquées au milieu des lignes 
                droites. L’intérieur de la piste est délimité par des lignes bleues en ligne droite et par 7 cônes en 
                caoutchouc dans les virages, celui du milieu représentant l’apex, c’est-à-dire le centre du virage [4].
                Ces caractéristiques sont représentées dans le schéma en figure 3.
        """)

    st.header("⚡Le patinage de vitesse courte piste : un sport à risque") 
    st.write("Paragraphe sur les risques du sport...")

    st.header("📊 Nombre de chutes par pays")
    display_fall_statistics(bdd)
   
    # Footer
    st.markdown('---')
    st.markdown('**Author:** Marie Salomon - INS Québec (https://www.insquebec.org/)')
    st.markdown('**Date:** mai - août 2024')

# =============================================================================
# METHODES PAGE STATISTIQUES: 
# =============================================================================
def display_impact_speed_distribution(bdd):
        # Créer un histogramme pour montrer la distribution de la vitesse d'impact
        plt.figure(figsize=(10, 8))
        sns.histplot(bdd['Impact speed mean'], kde=True, color='blue', bins=10)
        plt.title('Distribution of Impact Speed Mean')
        plt.xlabel('Impact Speed Mean (km/h)')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(plt)

def display_impact_speed_by_head_impact(bdd):
        # Créer un box plot pour montrer la vitesse d'impact en fonction du type de choc à la tête
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='Head impact', y='Impact speed mean', data=bdd, palette='viridis')
        plt.title('Impact Speed Mean by Head Impact Type')
        plt.xlabel('Head Impact Type')
        plt.ylabel('Impact Speed Mean (km/h)')
        st.pyplot(plt)
# =============================================================================
# PAGE: Statistiques
# =============================================================================
def page_statistiques(df=None):
    st.title("Bienvenue sur la page d'analyse des données recueillies")
    st.write("Voici la base de données sur laquelle nous avons travaillé :")
    st.write(bdd)
    st.write("Les chutes sur lesquelles on a pu calculer une vitesse d'impact :")

    bdd['Impact speed mean'] = bdd['Impact speed mean'].str.replace(',', '.')
    # Convertir les valeurs de 'Impact speed mean' en numériques, non numériques deviennent NaN
    bdd['Impact speed mean'] = pd.to_numeric(bdd['Impact speed mean'], errors='coerce')
    # Filtrer pour garder seulement les rangées avec des valeurs numériques dans 'Impact speed mean'
    bdd_mspeed = bdd[bdd['Impact speed mean'].notna()]
    st.write(bdd_mspeed)

    # def display_impactspeed_impactzone(bdd):

    #     # Fonction pour séparer les combinaisons d'impact
    #     def split_impact(impact):
    #         if '+' in impact:
    #             return impact.split(' + ')
    #         else:
    #             return [impact]

    #     # Appliquer la fonction pour chaque impact et étendre le DataFrame
    #     first_impacts = bdd_mspeed['First impact'].apply(split_impact).explode().reset_index()
    #     second_impacts = bdd_mspeed['Second impact'].apply(split_impact).explode().reset_index()

    #     # Associer chaque impact éclaté avec sa vitesse et son type d'impact à la tête
    #     first_impacts = first_impacts.join(bdd_mspeed[['Impact speed mean', 'Head impact']], on='index')
    #     second_impacts = second_impacts.join(bdd_mspeed[['Impact speed mean', 'Head impact']], on='index')

    #     # Créer un scatter plot pour montrer les relations
    #     plt.figure(figsize=(10, 8))
    #     scatter_first = sns.scatterplot(x='Impact speed mean', y='First impact', data=bdd_mspeed, style='Head impact', hue='Head impact', s=100, palette='Blues', legend = "full") # on désactive la légende automatique
    #     scatter_second = sns.scatterplot(x='Impact speed mean', y='Second impact', data=bdd_mspeed, style='Head impact', hue='Head impact', s=100, palette='Reds', legend = "full")

    #     plt.title('Impact Speed vs. First and Second Impact Points with Head Impact Type')
    #     plt.xlabel('Impact Speed Mean (km/h)')
    #     plt.ylabel('Impact Point')
    #     plt.grid(True)
    #     plt.legend(title='Head Impact')
    #     st.pyplot(plt)

    # display_impactspeed_impactzone(bdd)

    def display_impactspeed_impactzone(bdd):
        def clean_label(label):
            if pd.isna(label):
                return 'No Data'
            else:
                return ' + '.join(word.capitalize() for word in str(label).split(' + '))

        def split_impact(impact):
            if pd.isna(impact):
                return []
            elif '+' in str(impact):
                return impact.split(' + ')
            else:
                return [impact]

        # Appliquer la fonction pour chaque impact et étendre le DataFrame
        bdd['First impact'] = bdd['First impact'].apply(clean_label).apply(split_impact).explode().reset_index(drop=True)
        bdd['Second impact'] = bdd['Second impact'].apply(clean_label).apply(split_impact).explode().reset_index(drop=True)

        # Définir l'ordre des catégories
        ordered_categories = sorted(set(bdd['First impact'].dropna().unique()) | set(bdd['Second impact'].dropna().unique()))
        ordered_categories = [cat for cat in ordered_categories if cat not in ['No Data', 'None']] + ['None', 'No Data']

        # Convertir les catégories en type catégorique avec un ordre spécifié
        bdd['First impact'] = pd.Categorical(bdd['First impact'], categories=ordered_categories, ordered=True)
        bdd['Second impact'] = pd.Categorical(bdd['Second impact'], categories=ordered_categories, ordered=True)

        # Trier le DataFrame selon la nouvelle colonne catégorique pour assurer l'ordre dans le plot
        bdd.sort_values(by=['First impact', 'Second impact'], inplace=True)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Impact speed mean', y='First impact', data=bdd, style='Head impact', hue='Head impact', s=100, palette='Blues', legend="full")
        sns.scatterplot(x='Impact speed mean', y='Second impact', data=bdd, style='Head impact', hue='Head impact', s=100, palette='Reds', legend="full")

        plt.title('Impact Speed vs. First and Second Impact Points with Head Impact Type')
        plt.xlabel('Impact Speed Mean (km/h)')
        plt.ylabel('Impact Point')
        plt.grid(True)
        plt.legend(title='Head Impact')
        st.pyplot(plt)

    display_impactspeed_impactzone(bdd_mspeed)  

    # écris le 1er impact et le 2ème impact pour chaque vitesse d'impact moyenne :
    st.write(bdd_mspeed[['Impact speed mean', 'First impact', 'Second impact']])

    st.write(" * Les signes bleus correspondent aux 1er impacts")
    st.write(" * Les signes rouges correspondent aux 2ème impacts")

    display_impact_speed_distribution(bdd)
    display_impact_speed_by_head_impact(bdd)
    st.write("""Le graphique représente la distribution de la vitesse d'impact moyenne (en km/h) en fonction du type d'impact à la tête lors de chutes en patinage de vitesse courte piste. Trois catégories d'impact à la tête sont observées : aucun impact (None), impact indirect, et impact direct, avec une catégorie supplémentaire pour les cas où aucune donnée n'est disponible (No data). 
             À partir du graphique, on peut observer que la vitesse d'impact moyenne pour les chutes avec un impact direct à la tête est nettement plus élevée comparée à celles avec un impact indirect ou aucun impact. Les chutes avec impact direct montrent aussi une plus grande variabilité dans les vitesses d'impact, comme l'indique la longueur de la boîte et les extrémités des moustaches. En revanche, les chutes sans aucun impact à la tête présentent les vitesses les plus basses et la variabilité la plus faible.
             Ce graphique suggère que les impacts directs à la tête lors des chutes sont généralement associés à des vitesses d'impact plus élevées, ce qui pourrait indiquer un risque accru de blessures graves.
             """)
    
# =============================================================================
# METHODES PAGE MESURES ANALYSES: 
# =============================================================================

def get_t_value(degrees_of_freedom, confidence_level=0.95):
    # Calculer la valeur t pour le niveau de confiance donné et les ddl
    return t.ppf((1 + confidence_level) / 2, degrees_of_freedom)


# Les calculs de reproductibilité et répétabilité :
def calculer_repetabilite(data):
    # Calcul de la moyenne des données
    mean_data = np.mean(data)

    # Calcul de S, écart type des données
    S = np.sqrt(np.sum((data - mean_data)**2) / (len(data) - 1))

    # Calcul de Sbarre, erreur standard de la moyenne
    Sbarre = S / np.sqrt(len(data))

    df = len(data) - 1 # Détermination des degrés de liberté
    t_value = get_t_value(df) # Utilisation de la fonction get_t_value pour obtenir la valeur t
    # st.write(t_value)
    U = t_value * Sbarre # Calcul de U, incertitude avec un facteur de couverture t

    return mean_data, S, Sbarre, U


def calculer_reproductibilite(*datasets):
    p = len(datasets)
    n = len(datasets[0])
    means = np.array([np.mean(data) for data in datasets])
    mean_all = np.mean(means)
    std_devs = np.array([np.std(data, ddof=1) for data in datasets])
    Sr = np.sqrt(np.sum(std_devs**2) / p)
    variance_inter = np.var(means, ddof=1)
    SR = np.sqrt(variance_inter + ((n-1) / n) * Sr**2)
    k = 2.571
    U = k * SR
    return means, mean_all, Sr, SR, variance_inter, U


def plot_density(data1, data2, label1, label2, title):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data1, shade=True, label=label1, color='blue', alpha=0.6)
    sns.kdeplot(data2, shade=True, label=label2, color='red', alpha=0.6)
    plt.title(title)
    plt.xlabel('Vitesse (km/h)')
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def interpreter_densite(vitesses1, vitesses2):
    mean1 = np.mean(vitesses1)
    std1 = np.std(vitesses1)
    mean2 = np.mean(vitesses2)
    std2 = np.std(vitesses2)
    
    interpretation = ""
    
    # Analyse de la moyenne des vitesses
    if abs(mean1 - mean2) > max(std1, std2):
        interpretation += f"Les moyennes des vitesses sont significativement différentes ({mean1:.2f} km/h vs {mean2:.2f} km/h), indiquant une différence notable entre les deux séries de mesures.\n"
    else:
        interpretation += f"Les moyennes des vitesses sont assez proches ({mean1:.2f} km/h vs {mean2:.2f} km/h), indiquant une similarité dans les performances mesurées.\n"
    
    # Analyse de la dispersion des vitesses
    if std1 > std2:
        interpretation += f"La dispersion des vitesses pour la première série (écart-type de {std1:.2f}) est supérieure à celle de la deuxième série (écart-type de {std2:.2f}), suggérant une plus grande variabilité dans les mesures de la première série.\n"
    else:
        interpretation += f"La dispersion des vitesses pour la deuxième série (écart-type de {std2:.2f}) est supérieure à celle de la première série (écart-type de {std1:.2f}), suggérant une plus grande variabilité dans les mesures de la deuxième série.\n"
    
    # Conclusions additionnelles sur la répétabilité et la reproductibilité
    if std1 < 0.5 * mean1 and std2 < 0.5 * mean2:
        interpretation += "Les faibles écarts-types relatifs aux moyennes suggèrent une bonne répétabilité des mesures pour les deux séries."
    else:
        interpretation += "Les écarts-types relativement élevés indiquent des variations significatives dans les mesures, ce qui peut questionner la répétabilité et la précision des mesures."

    return interpretation

# Fonction pour afficher les graphiques basés sur la sélection de l'utilisateur
def afficher_graphiques(vitesses, names, choix_course):
    # Bar Chart for Mean Speeds with Error Bars
    mean_speeds = [np.mean(data) for data in vitesses]
    uncertainties = [calculer_repetabilite(data)[3] for data in vitesses]  # Utilisation de U pour l'erreur

    plt.figure(figsize=(8, 4))
    plt.bar(names, mean_speeds, yerr=uncertainties, color='skyblue', capsize=5)
    plt.ylabel('Vitesse Moyenne (km/h)')
    plt.title(f'Moyennes des Vitesses avec Incertitudes pour {choix_course}')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(plt)

    # Scatter Plot for Individual Speed Measurements
    plt.figure(figsize=(10, 6))
    for i, (speeds, name) in enumerate(zip(vitesses, names)):
        tests = [f'Test {j+1}' for j in range(len(speeds))]
        plt.scatter(tests, speeds, label=f'{name}')
        plt.plot(tests, speeds)  # Adding line to connect points
    plt.title(f'Vitesses Mesurées pour {choix_course}')
    plt.xlabel('Test')
    plt.ylabel('Vitesse (km/h)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(plt)
    

def charger_donnees(fichier):
    try:
        data = pd.read_csv(fichier, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(fichier, delimiter=';', encoding='latin1')

    # Liste des colonnes à exclure de la conversion
    colonnes_exclues = ['Numéro de test', 'Zone', 'Type de test', 'Trajectoire', 'Image K1', 'Image K2', 'Commentaires']

    # Remplacer les virgules par des points et les valeurs invalides par NaN uniquement dans les colonnes numériques
    for col in data.columns:
        if col not in colonnes_exclues:
            data[col] = data[col].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
            data[col] = data[col].apply(lambda x: np.nan if str(x) == '#DIV/0!' else x)
            data[col] = pd.to_numeric(data[col], errors='coerce')

    return data

def nettoyer_donnees(bdd_vitesse):
    # Supprimer les lignes avec des valeurs manquantes dans la colonne 'Trajectoire'
    bdd_vitesse = bdd_vitesse.dropna(subset=['Trajectoire'])
    return bdd_vitesse


# =============================================================================
# PAGE: Mesures et analyses
# =============================================================================

def page_mesures_analyses():
    st.title("Mesures et analyses")
    st.write("## Comment ont été fait les calculs ?")
    st.write("Analyse de vidéos sur Kinovéa")

    image_path = "./trajectoire.jpg"# Main image 
    st.image(image_path) # Display the image with a caption

    def afficher_resultats(choix_competition, choix_course, datasets, names, selected_for_repro, selected_for_repe):
        # Filter datasets for reproducibility analysis
        selected_datasets_repro = [datasets[names.index(name)] for name in selected_for_repro]
        vitesses_repro = [data[choix_course].dropna().astype(float).values for data in selected_datasets_repro]
        
        # Filter datasets for repeatability analysis
        selected_datasets_repe = [datasets[names.index(name)] for name in selected_for_repe]
        vitesses_repe = [data[choix_course].dropna().astype(float).values for data in selected_datasets_repe]
        
        if vitesses_repe:
            # Repeatability calculation for the same person (or unit), different trials
            xbarre, S, Sbarre, Urepetabilite = calculer_repetabilite(vitesses_repe[0])

            # Display results for repeatability
            st.write("### Répétabilité")
            st.write(f"**Vitesses mesurées :** {vitesses_repe[0]}")
            st.write(f"**Moyenne des vitesses :** {xbarre:.2f}")
            st.write(f"**Écart type (S) :** {S:.2f}")
            st.write(f"**Erreur standard de la moyenne (Sbarre) :** {Sbarre:.2f}")
            st.write(f"**U (Incertitude avec facteur de couverture t) :** {Urepetabilite:.2f}")
            st.write(f"**Vitesse moyenne :** {xbarre:.2f} ± {Urepetabilite:.2f}")

        if vitesses_repro:
            # Reproducibility calculation across different units (people or methods)
            means, mean_all, Sr, SR, variance_inter, Ureproductibilite = calculer_reproductibilite(*vitesses_repro)

            # Display results for reproducibility
            st.write("### Reproductibilité")
            for idx, mean in enumerate(means):
                st.write(f"**Moyenne des vitesses (Personne {idx + 1}) :** {mean:.2f}")
            st.write(f"**Moyenne globale :** {mean_all:.2f}")
            st.write(f"**Écart-type de répétabilité (Sr) :** {Sr:.2f}")
            st.write(f"**Écart-type de reproductibilité (SR) :** {SR:.2f}")
            st.write(f"**Variance inter-individuelle :** {variance_inter:.2f}")
            st.write(f"**Incertitude élargie (U) :** {Ureproductibilite:.2f}")

        # Interaction utilisateur pour la sélection des graphiques
        if st.button('Générer Graphiques'):
            afficher_graphiques(vitesses_repro, dataset_names, choix_course)

    # Choix des datasets pour reproductibilité et répétabilité
    st.write("## Calculs d'incertitudes pour les mesures de vitesse")
    datasets = {
        'Montréal': ['Mesures_vitesses_Marie.csv', 'Mesures_vitesses_Laurianne.csv'],
        'Séoul': ['Mesures_vitesses_Seoul_Marie.csv', 'Mesures_vitesses_Seoul_Laurianne.csv', 'Mesures_vitesses_Seoul_Lisa.csv']
    }

    choix_competition = st.selectbox('Choisissez une compétition:', list(datasets.keys()))
    if choix_competition == "Montréal":
        st.write("Les chutes étudiées sont à l'apex")
    elif choix_competition == "Séoul":
        st.write("Les chutes étudiées sont en ligne droite")

    dataset_files = datasets[choix_competition]
    loaded_datasets = [charger_donnees(file) for file in dataset_files]
    dataset_names = ['Marie', 'Laurianne', 'Lisa'][:len(loaded_datasets)]

    # # Initialize selections to avoid issues
    # selected_for_repro = selected_for_repe = dataset_names  # Default to all datasets
    selected_for_repe = st.multiselect('Choisissez les datasets pour la répétabilité:', dataset_names, default=dataset_names)
    selected_for_repro = st.multiselect('Choisissez les datasets pour la reproductibilité:', dataset_names, default=dataset_names)

    if selected_for_repro or selected_for_repe:
        choix_course = st.selectbox('Choisissez une course:', loaded_datasets[0].columns)
        if choix_course:
            afficher_resultats(choix_competition, choix_course, loaded_datasets, dataset_names, selected_for_repro, selected_for_repe)



# =============================================================================
# METHODES PAGE VALIDATION KINOVEA: 
# =============================================================================

def generer_statistiques(bdd_vitesse, erreur_cols, unit):
    if not erreur_cols:
        st.write("Aucune colonne d'erreur sélectionnée.")
        return None

    # Vérifier si toutes les colonnes d'erreur existent dans les données
    erreur_cols_valides = [col for col in erreur_cols if col in bdd_vitesse.columns]
    if not erreur_cols_valides:
        st.write("Les colonnes d'erreur sélectionnées ne sont pas présentes dans les données.")
        return None

    erreurs_stats = bdd_vitesse[erreur_cols_valides].agg(["mean", "std"])
    st.write(erreurs_stats)
    return erreurs_stats

def generer_statistiques_par_zone_selectionnee(bdd_vitesse, erreur_cols_kmh, erreur_cols_pct, unit):
    zone_selectionnee = st.selectbox('Sélectionnez la zone', bdd_vitesse['Zone'].unique())

    if unit == 'km/h':
        erreur_cols = erreur_cols_kmh
    else:
        erreur_cols = erreur_cols_pct

    if len(erreur_cols) == 0:
        st.write("Aucune colonne d'erreur sélectionnée.")
        return

    erreurs_stats = bdd_vitesse[bdd_vitesse['Zone'] == zone_selectionnee][erreur_cols].agg(["mean", "std"])
    st.write(erreurs_stats)
    return erreurs_stats

def generer_boxplots_erreurs(bdd_vitesse, erreur_cols, unit):
    plt.figure(figsize=(12, 8))
    boxprops = dict(facecolor='#FFCC29', color='black')
    medianprops = dict(color='#F58634')
    plt.boxplot([bdd_vitesse[col].dropna() for col in erreur_cols], 
            labels=erreur_cols, 
            patch_artist=True, 
            boxprops=boxprops,
            medianprops=medianprops)
    plt.title(f"Distribution des erreurs par méthode ({unit})")
    plt.ylabel(f"Erreur ({unit})")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

def generer_scatter_plots(bdd_vitesse, vitesse_cols):
    num_plots = len(vitesse_cols)
    cols = 3
    rows = math.ceil(num_plots / cols)

    plt.figure(figsize=(14, 10))
    for i, col in enumerate(vitesse_cols, 1):
        plt.subplot(rows, cols, i)
        valid_data = bdd_vitesse[["Vitesse capteurs", col]].dropna()
        if not valid_data.empty:
            X = valid_data["Vitesse capteurs"].values.reshape(-1, 1)
            y = valid_data[col].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            plt.scatter(X, y, label='Données')
            plt.plot(X, y_pred, color='blue', linewidth=2, label='Régression linéaire')
            plt.plot([X.min(), X.max()], [X.min(), X.max()], 'r--', label='y=x')
            plt.title(col)
            plt.xlabel('Vitesse Capteurs (m/s)')
            plt.ylabel('Vitesse Mesurée (m/s)')
            plt.legend()
            plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def generer_boxplots_par_trajectoire(bdd_vitesse, erreur_cols, unit):
    plt.figure(figsize=(14, 10))
    boxprops = dict(facecolor='#F58634', color='black')
    medianprops = dict(color='#206A5D')
    trajectoires = bdd_vitesse['Trajectoire'].unique()
    for i, col in enumerate(erreur_cols, 1):
        plt.subplot(math.ceil(len(erreur_cols) / 3), 3, i)
        subset = [bdd_vitesse[bdd_vitesse['Trajectoire'] == traj][col].dropna() for traj in trajectoires]
        plt.boxplot(subset, 
                    patch_artist=True, 
                    boxprops=boxprops,
                    medianprops=medianprops,
                    labels=trajectoires)
        plt.title(col)
        plt.ylabel(f'Erreur ({unit})')
        plt.xticks(rotation=45)
        plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def generer_statistiques_par_trajectoire(bdd_vitesse, erreur_cols, unit):
    trajectoires = bdd_vitesse['Trajectoire'].unique()
    stats = pd.DataFrame(index=['mean', 'std'], columns=pd.MultiIndex.from_product([erreur_cols, trajectoires]))
    for col in erreur_cols:
        for traj in trajectoires:
            data = bdd_vitesse[bdd_vitesse['Trajectoire'] == traj][col].dropna()
            if not data.empty:
                stats[col, traj] = [data.mean(), data.std()]
    return stats


def generer_boxplots_par_zone(bdd_vitesse, erreur_cols, unit):
    plt.figure(figsize=(14, 10))
    boxprops = dict(facecolor='#75C0BE', color='black')
    medianprops = dict(color='#FFCC29')
    zones = bdd_vitesse['Zone'].unique()
    for i, col in enumerate(erreur_cols, 1):
        plt.subplot(math.ceil(len(erreur_cols) / 3), 3, i)
        subset = [bdd_vitesse[bdd_vitesse['Zone'] == zone][col].dropna() for zone in zones]
        plt.boxplot(subset, 
                    patch_artist=True, 
                    boxprops=boxprops,
                    medianprops=medianprops,
                    labels=zones)
        plt.title(col)
        plt.ylabel(f'Erreur ({unit})')
        plt.xticks(rotation=45)
        plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def generer_violin_plots(bdd_vitesse, erreur_cols, unit):
    # st.write(f"### Violin plots des erreurs par trajectoire ({unit})")
    plt.figure(figsize=(14, 10))
    palette = {'Courbe': '#F58634', 'Rectiligne': '#F58634'}
    for i, col in enumerate(erreur_cols, 1):
        plt.subplot(math.ceil(len(erreur_cols) / 3), 3, i)
        sns.violinplot(x='Trajectoire', y=col, data=bdd_vitesse, palette=palette)
        plt.title(col)
        plt.ylabel(f'Erreur ({unit})')
        plt.xticks(rotation=45)
        plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def generer_explanation_boxplot_par_contexte(erreurs_stats, erreur_cols, contexte, unit):
    st.write(f"## Explication des boxplots par {contexte} ({unit})")
    for col in erreur_cols:
        try:
            mediane = erreurs_stats[col, 'mean'].values[0]
            std_dev = erreurs_stats[col, 'std'].values[0]
            if pd.isna(mediane) or pd.isna(std_dev):
                st.write(f"Les statistiques pour **{col}** ne sont pas disponibles.")
                continue
        except KeyError:
            st.write(f"Les statistiques pour **{col}** ne sont pas disponibles.")
            continue

        if unit == 'km/h':
            if mediane < 2:
                interpretation = "Médiane très basse avec une boîte très petite, indiquant une très faible variabilité."
                fiabilite = "Cette méthode est très précise."
            elif mediane < 5:
                interpretation = "Médiane basse avec une boîte petite, indiquant une faible variabilité."
                fiabilite = "Cette méthode est précise."
            elif mediane < 10:
                interpretation = "Médiane moyenne avec une boîte modérément large, indiquant une variabilité moyenne."
                fiabilite = "Cette méthode est modérément précise."
            else:
                interpretation = "Médiane élevée avec une boîte large, indiquant une grande variabilité."
                fiabilite = "Cette méthode est peu précise."
        else:  # unit == '%'
            if mediane < 5:
                interpretation = "Médiane très basse avec une boîte très petite, indiquant une très faible variabilité."
                fiabilite = "Cette méthode est très précise."
            elif mediane < 10:
                interpretation = "Médiane basse avec une boîte petite, indiquant une faible variabilité."
                fiabilite = "Cette méthode est précise."
            elif mediane < 15:
                interpretation = "Médiane moyenne avec une boîte modérément large, indiquant une variabilité moyenne."
                fiabilite = "Cette méthode est modérément précise."
            else:
                interpretation = "Médiane élevée avec une boîte large, indiquant une grande variabilité."
                fiabilite = "Cette méthode est peu précise."
        
        st.write(f"* **{col}** : {interpretation} Les moustaches sont courtes, mais il y a un outlier, ce qui signifie qu'il y a quelques valeurs extrêmes. {fiabilite}")

def generer_explanation_scatter(bdd_vitesse, vitesse_cols):
    st.write("## Explication des résultats des scatter plots")
    st.write("""
    * **Ligne y=x :** La ligne rouge pointillée. Si les points se trouvent sur cette ligne, cela signifie une correspondance parfaite entre les vitesses mesurées par les capteurs et celles mesurées par Kinovea.
    * **Ligne de Régression :** La ligne bleue. Elle montre la tendance générale des données et aide à identifier les biais systématiques ou les erreurs.
    """)

    for col in vitesse_cols:
        valid_data = bdd_vitesse[["Vitesse capteurs", col]].dropna()
        if not valid_data.empty:
            X = valid_data["Vitesse capteurs"].values.reshape(-1, 1)
            y = valid_data[col].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            mse = mean_squared_error(y, y_pred)
            r2 = reg.score(X, y)

            observation = "Les points sont dispersés autour de la ligne de régression et la ligne y=x." if r2 < 0.8 else "Les points sont relativement proches de la ligne de régression et la ligne y=x."
            interpretation = "La droite de régression (bleue) est proche de la ligne y=x, indiquant que les mesures de Kinovea sont proportionnelles aux mesures des capteurs mais avec des écarts systématiques." if r2 >= 0.8 else "La droite de régression (bleue) est moins proche de la ligne y=x, indiquant que les mesures de Kinovea sont moins proportionnelles aux mesures des capteurs."
            conclusion = "Il y a une certaine concordance entre les vitesses mesurées par les capteurs et Kinovea, mais des erreurs systématiques existent." if r2 >= 0.8 else "Les mesures de Kinovea présentent une plus grande variabilité et des erreurs systématiques plus importantes."

            st.write(f"""
            * **{col}** :
                - **Observation :** {observation}
                - **Interprétation :** {interpretation}
                - **Conclusion :** {conclusion}
                - **Erreur Quadratique Moyenne (EQM) :** {mse:.2f}
                - **Coefficient de Détermination (R²) :** {r2:.2f}
            """)

def analyser_erreurs_par_zone(erreurs_stats_zone, unit):

    # Définir les seuils en fonction de l'unité
    if unit == 'km/h':
        seuil_basse_variabilite = 2
        seuil_moyenne_variabilite = 5
        seuil_elevee_variabilite = 10
    else:  # unit == '%'
        seuil_basse_variabilite = 5
        seuil_moyenne_variabilite = 10
        seuil_elevee_variabilite = 15

    # # Déduction des méthodes par zone
    # st.write("### Déduction par zone")
    # if st.checkbox('Afficher les détails de la déduction par zone'):
    #     for zone in erreurs_stats_zone.index.levels[0]:
    #         st.write(f"#### Zone : {zone}")
    #         for col in erreurs_stats_zone.columns:
    #             try:
    #                 mediane = erreurs_stats_zone.loc[(zone, 'mean'), col]
    #                 if pd.isna(mediane):
    #                     st.write(f"Les statistiques pour **{col}** ne sont pas disponibles.")
    #                     continue
    #                 if mediane < seuil_basse_variabilite:
    #                     fiabilite = "très précise"
    #                 elif mediane < seuil_moyenne_variabilite:
    #                     fiabilite = "précise"
    #                 elif mediane < seuil_elevee_variabilite:
    #                     fiabilite = "modérément précise"
    #                 else:
    #                     fiabilite = "peu précise"
    #                 st.write(f"* **{col}** : {fiabilite}")
    #             except KeyError:
    #                 st.write(f"Les statistiques pour **{col}** ne sont pas disponibles.")


def generer_explanation_violin(erreur_cols):
    st.write("## Explication des résultats des violin plots")
    for col in erreur_cols:
        st.write(f"""* **{col}** :
        - Observation : Les formes des violons montrent la densité des données et la distribution des erreurs.
        - Interprétation : Une forme plus large indique une plus grande densité de données pour une certaine valeur d'erreur.
        - Conclusion : La forme du violon aide à visualiser la distribution des erreurs et à comprendre la variabilité des données.""")

def calculer_erreur_generale(bdd_vitesse, vitesse_cols):
    mse_erreurs = []
    mae_erreurs = []
    for col in vitesse_cols:
        valid_data = bdd_vitesse[["Vitesse capteurs", col]].dropna()
        if not valid_data.empty:
            y_true = valid_data["Vitesse capteurs"]
            y_pred = valid_data[col]
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse_erreurs.append(mse)
            mae_erreurs.append(mae)
    erreur_generale_mse = np.mean(mse_erreurs)
    erreur_generale_mae = np.mean(mae_erreurs)
    return erreur_generale_mse, erreur_generale_mae

def afficher_protocole():
    st.write()

def generer_statistiques_graphiques(bdd_vitesse):
        partie = st.selectbox('Sélectionnez les mesures à analyser ', ['Premières mesures', 'Deuxièmes mesures', 'Les deux'])

        if partie == 'Premières mesures':
            erreur_cols_kmh = [col for col in bdd_vitesse.columns if 'Erreur1 km/h' in col and 'cam' in col]
            erreur_cols_pct = [col for col in bdd_vitesse.columns if 'Erreur1 %' in col and 'cam' in col]
            vitesse_cols = [col for col in bdd_vitesse.columns if 'Kinovéa1' in col and 'cam' in col]
        elif partie == 'Deuxièmes mesures':
            erreur_cols_kmh = [col for col in bdd_vitesse.columns if 'Erreur2 km/h' in col and 'cam' in col]
            erreur_cols_pct = [col for col in bdd_vitesse.columns if 'Erreur2 %' in col and 'cam' in col]
            vitesse_cols = [col for col in bdd_vitesse.columns if 'Kinovéa2' in col and 'cam' in col]
        else:
            erreur_cols_kmh = [col for col in bdd_vitesse.columns if ('Erreur1 km/h' in col or 'Erreur2 km/h' in col) and 'cam' in col]
            erreur_cols_pct = [col for col in bdd_vitesse.columns if ('Erreur1 %' in col or 'Erreur2 %' in col) and 'cam' in col]
            vitesse_cols = [col for col in bdd_vitesse.columns if ('Kinovéa1' in col or 'Kinovéa2' in col) and 'cam' in col]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## Statistiques sur les erreurs des méthodes")
        with col2:
            unit_stat = st.radio("", ['km/h', '%'], key='stat', horizontal = True)

        erreur_cols_stat = erreur_cols_kmh if unit_stat == 'km/h' else erreur_cols_pct
        erreurs_stats = generer_statistiques(bdd_vitesse, erreur_cols_stat, unit_stat)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## Boxplots des erreurs")
        with col2:
            unit_boxplot = st.radio("", ['km/h', '%'], key='boxplot', horizontal = True)

        erreur_cols_boxplot = erreur_cols_kmh if unit_boxplot == 'km/h' else erreur_cols_pct
        generer_boxplots_erreurs(bdd_vitesse, erreur_cols_boxplot, unit_boxplot)

        with st.expander('Explications des statistiques des erreurs'):
            st.write("## Explication des boxplots")
            st.write("""* **Erreur1 m1 camLD** : Médiane assez élevée avec une boîte relativement large, indiquant une variabilité modérée. Les moustaches sont assez courtes, mais il y a un outlier, ce qui signifie qu'il y a quelques valeurs extrêmes.""")
            st.write("""* **Erreur1 m1 camV** : Médiane très élevée avec une boîte très large, indiquant une grande variabilité. Les moustaches sont longues, et il y a plusieurs outliers, ce qui indique une méthode très peu fiable avec beaucoup de dispersion.""")         
            st.write("""* **Erreur1 m1 camLD ligne** : Médiane basse avec une boîte petite, indiquant une faible variabilité. Les moustaches sont courtes, ce qui signifie que les données sont concentrées et la méthode est très fiable.""")         
            st.write("""* **Erreur1 m1 camV ligne** : Médiane très basse avec une boîte très petite, indiquant une très faible variabilité. Les moustaches sont courtes, ce qui signifie que cette méthode est la plus fiable avec des données très concentrées.""")        
            st.write("""* **Erreur1 m2 camLD** : Médiane moyenne avec une boîte modérément large, indiquant une variabilité moyenne. Les moustaches sont un peu plus longues, mais avec quelques outliers, ce qui indique une certaine dispersion.""")         
            st.write("""* **Erreur1 m2 camV** : Médiane et variabilité similaires à camLD, avec une boîte large et des moustaches relativement longues. Il y a plusieurs outliers, ce qui signifie une certaine dispersion.""")

        st.write("## Scatter plots des vitesses")
        generer_scatter_plots(bdd_vitesse, vitesse_cols)
        # Ajouter une checkbox pour afficher ou non les explications
        with st.expander('Explications des scatter plots des vitesses'):
            generer_explanation_scatter(bdd_vitesse, vitesse_cols)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## Statistiques des erreurs par zone")
        with col2:
            unit_stat_zone = st.radio('', ['km/h', '%'], key='stat_zone', horizontal = True)

        erreurs_stats_zone = generer_statistiques_par_zone_selectionnee(bdd_vitesse, erreur_cols_kmh, erreur_cols_pct, unit_stat_zone)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## Boxplots des erreurs par type de trajectoire")
        with col2:
            unit_boxplot_trajectoire = st.radio('', ['km/h', '%'], key='boxplot_trajectoire', horizontal = True)

        erreur_cols_boxplot_trajectoire = erreur_cols_kmh if unit_boxplot_trajectoire == 'km/h' else erreur_cols_pct
        generer_boxplots_par_trajectoire(bdd_vitesse, erreur_cols_boxplot_trajectoire, unit_boxplot_trajectoire)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("## Boxplots des erreurs par zone")
        with col2:
            unit_boxplot_zone = st.radio('', ['km/h', '%'], key='boxplot_zone', horizontal = True)

        erreur_cols_boxplot_zone = erreur_cols_kmh if unit_boxplot_zone == 'km/h' else erreur_cols_pct
        generer_boxplots_par_zone(bdd_vitesse, erreur_cols_boxplot_zone, unit_boxplot_zone)

        with st.expander('Analyse des boxplots des erreurs par zone'):
            analyser_erreurs_par_zone(erreurs_stats_zone, unit_boxplot_zone)

        st.write("### Synthèse des Recommandations")
        st.write("""
        - **Apex** : Utiliser **m1 camLD** pour les trajectoires courbes. Alternative avec ligne : **m1 camLD ligne**.
        - **Après Apex** : Utiliser **m1 camLD** pour les trajectoires courbes. Alternative avec ligne : **m1 camLD ligne**.
        - **SV** : Utiliser **m1 camLD** pour les trajectoires courbes et éviter **m1 camV** pour les trajectoires rectilignes.
        - **LD** : Utiliser **m2 camLD** pour les trajectoires rectilignes.
        - **Chute en Sortie de Virage** : Utiliser **m1 camLD** pour les trajectoires courbes. Alternative avec ligne : **m1 camLD ligne**. Utiliser **m2 camLD** pour les trajectoires rectilignes.
        """)
                
        # st.write("## Violin plots des erreurs par trajectoire")
        # unit_violin_plot = st.radio('Sélectionnez l\'unité pour les violin plots', ['km/h', '%'], key='violin_plot')
        # erreur_cols_violin_plot = erreur_cols_kmh if unit_violin_plot == 'km/h' else erreur_cols_pct
        # generer_violin_plots(bdd_vitesse, erreur_cols_violin_plot, unit_violin_plot)
        # with st.expander('Explications des violin plots des erreurs par trajectoire'):
        #     generer_explanation_violin(erreur_cols_violin_plot)


def analyse_repetabilite(bdd_vitesse):
    st.write("""
    ### Légende des paramètres calculés :
    - **Moyenne (mean)** : Moyenne des différences entre les erreurs 1 et 2 pour chaque méthode. Plus basse est meilleure.
    - **Écart-type (S)** : Variabilité des différences. Plus bas est meilleur pour la répétabilité.
    - **Erreur standard de la moyenne (Sbarre)** : Précision de la moyenne calculée. Plus basse est meilleure.
    - **Incertitude (U)** : Intervalle de confiance autour de la moyenne. Plus faible est préférable.
    """)

    # Stockage des résultats
    resultats_repetabilite = {}

    # Noms des méthodes pour faciliter l'accès aux colonnes
    methodes = [
        'm1 camLD', 'm1 camV', 'm1 camLD ligne', 'm1 camV ligne', 'm2 camLD', 'm2 camV'
    ]

    # Calculer la répétabilité pour chaque méthode
    for methode in methodes:
        col1 = f'Erreur1 km/h {methode}'
        col2 = f'Erreur2 km/h {methode}'
        data1 = bdd_vitesse[col1].dropna()
        data2 = bdd_vitesse[col2].dropna()

        if len(data1) > 1 and len(data2) > 1:
            combined_data = pd.concat([data1, data2])
            mean_val, S_val, Sbarre_val, U_val = calculer_repetabilite(combined_data)
            resultats_repetabilite[methode] = {
                'mean': mean_val,
                'S': S_val,
                'Sbarre': Sbarre_val,
                'U': U_val
            }
        else:
            st.write(f"Skipping {methode} due to insufficient data")

        # Affichage des résultats avec analyse
    for methode, stats in resultats_repetabilite.items():
        with st.expander(f"Répétabilité de la méthode **{methode}**"):
            st.markdown(f"""
            <div style="background-color: #f7f7f7; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                <ul>
                    <li><strong>Moyenne (mean):</strong> {stats['mean']:.2f} km/h</li>
                    <li><strong>Écart-type (S):</strong> {stats['S']:.2f} km/h</li>
                    <li><strong>Erreur standard de la moyenne (Sbarre):</strong> {stats['Sbarre']:.2f} km/h</li>
                    <li><strong>Incertitude (U):</strong> {stats['U']:.2f} km/h</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #eaf1f8; margin-top: 10px; padding: 10px; border-radius: 5px; border-left: 5px solid #3178c6;">
                <h4>Analyse :</h4>
                <p><strong>Écart-type (S):</strong> {'Excellente répétabilité avec une très faible variabilité.' if stats['S'] < 1 else 'Bonne répétabilité avec une faible variabilité.' if stats['S'] < 2 else 'Variabilité modérée à élevée, répétabilité à améliorer.'}</p>
                <p><strong>Erreur standard de la moyenne (Sbarre):</strong> {'Précision très élevée des mesures.' if stats['Sbarre'] < 0.5 else 'Précision élevée des mesures.' if stats['Sbarre'] < 1 else 'Précision des mesures à améliorer.'}</p>
                <p><strong>Incertitude (U):</strong> {'Très faible incertitude, indicateurs de résultats très fiables.' if stats['U'] < 1 else 'Faible incertitude, indicateurs de bons résultats fiables.' if stats['U'] < 2 else 'Incertitude notable, indicateurs que les résultats peuvent varier.'}</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("\n")

    resultats_repetabilite = {
        "m1 camLD km/h": {"mean": 3.92, "S": 3.59, "Sbarre": 0.68, "U": 1.39},
        "m1 camV km/h": {"mean": 9.82, "S": 9.39, "Sbarre": 1.77, "U": 3.64},
        "m1 camLD ligne km/h": {"mean": 2.08, "S": 1.36, "Sbarre": 0.34, "U": 0.73},
        "m1 camV ligne km/h": {"mean": 1.77, "S": 2.07, "Sbarre": 0.60, "U": 1.32},
        "m2 camLD km/h": {"mean": 2.99, "S": 2.61, "Sbarre": 0.49, "U": 1.01},
        "m2 camV km/h": {"mean": 3.79, "S": 3.80, "Sbarre": 0.72, "U": 1.48}
    }

    df = pd.DataFrame(resultats_repetabilite).T  # Transpose pour ajuster le DataFrame

    # Ajout de la colonne "Répétabilité" selon le critère de l'écart-type S
    df['Répétabilité'] = df['S'].apply(lambda x: 'Élevée' if x < 2 else ('Moyenne' if x < 5 else 'Faible'))

    # Fonction pour appliquer un code couleur selon les valeurs de S, Sbarre, U
    def apply_color(val, col):
        if col in ['S', 'Sbarre', 'U']:  # Check if the column should have color applied
            if col == 'S':
                color = 'background-color: green' if val < 2 else 'background-color: orange' if val < 5 else 'background-color: red'
            elif col == 'Sbarre':
                color = 'background-color: green' if val < 0.5 else 'background-color: orange' if val < 1 else 'background-color: red'
            elif col == 'U':
                color = 'background-color: green' if val < 1 else 'background-color: orange' if val < 2 else 'background-color: red'
            return color
        return ''

    # Apply the style for specific columns using the applymap method correctly
    styled_df = df.style.apply(lambda x: x.map(lambda v: apply_color(v, x.name)), axis=0)
    st.dataframe(styled_df)


def preparer_donnees_anova_moyennes(bdd_vitesse):
    anova_data = pd.DataFrame()

    # Inclure toutes les méthodes, y compris les méthodes lignes
    methodes = ['m1 camLD', 'm1 camV', 'm1 camLD ligne', 'm1 camV ligne', 'm2 camLD', 'm2 camV']

    # Calculer les moyennes pour chaque méthode
    for methode in methodes:
        if f'Kinovéa1 {methode}' in bdd_vitesse.columns and f'Kinovéa2 {methode}' in bdd_vitesse.columns:
            moyenne_vitesses = (bdd_vitesse[f'Kinovéa1 {methode}'] + bdd_vitesse[f'Kinovéa2 {methode}']) / 2
            temp_df = pd.DataFrame({
                'Vitesse': moyenne_vitesses,
                'Méthode': methode,
                'Sujet': bdd_vitesse.index
            })
            anova_data = pd.concat([anova_data, temp_df])

    # Ajouter les valeurs de référence (vitesses capteurs)
    reference_data = pd.DataFrame({
        'Vitesse': bdd_vitesse['Vitesse capteurs'],
        'Méthode': 'Référence',
        'Sujet': bdd_vitesse.index
    })
    anova_data = pd.concat([anova_data, reference_data])

    return anova_data.dropna()


# Étape 2 : Réalisation de l'ANOVA
def effectuer_anova_vitesse(anova_data):
    model = ols('Vitesse ~ C(Méthode)', data=anova_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# Étape 3 : Tests post-hoc
def tests_post_hoc_vitesse(anova_data):
    tukey_results = pairwise_tukeyhsd(endog=anova_data['Vitesse'], groups=anova_data['Méthode'], alpha=0.05)
    return tukey_results


def preparer_donnees_anova_repetee(bdd_vitesse):
    anova_data = pd.DataFrame()

    methodes = ['m1 camLD', 'm1 camV', 'm2 camLD', 'm2 camV']
    
    # Ajouter les valeurs de référence
    ref_data = pd.DataFrame({
        'Vitesse': bdd_vitesse['Vitesse capteurs'],
        'Méthode': 'Référence',
        'Sujet': bdd_vitesse.index
    })
    anova_data = pd.concat([anova_data, ref_data])

    # Ajouter les valeurs des autres méthodes
    for methode in methodes:
        for i in range(1, 3):  # Pour inclure Kinovea1 et Kinovea2
            temp_df = pd.DataFrame({
                'Vitesse': bdd_vitesse[f'Kinovéa{i} {methode}'],
                'Méthode': methode,
                'Sujet': bdd_vitesse.index
            }).dropna()
            anova_data = pd.concat([anova_data, temp_df])

    return anova_data.dropna()

def effectuer_anova_repetee(anova_data):
    anova_rm = AnovaRM(anova_data, 'Vitesse', 'Sujet', within=['Méthode'])
    anova_results = anova_rm.fit()
    return anova_results

def preparer_donnees_anova_erreur(bdd_vitesse):
    # Créer un DataFrame pour l'ANOVA avec une colonne pour les méthodes et une pour les erreurs
    anova_data = pd.DataFrame()

    methodes = [
        'm1 camLD', 'm1 camV', 'm1 camLD ligne', 'm1 camV ligne', 'm2 camLD', 'm2 camV'
    ]

    # Concaténer les données de chaque méthode
    for methode in methodes:
        # On suppose que les erreurs pour chaque méthode sont stockées dans des colonnes 'Erreur1 km/h {methode}'
        # et 'Erreur2 km/h {methode}'
        temp_df = pd.DataFrame({
            'Erreur': pd.concat([bdd_vitesse[f'Erreur1 km/h {methode}'], bdd_vitesse[f'Erreur2 km/h {methode}']]),
            'Méthode': methode
        })
        anova_data = pd.concat([anova_data, temp_df])

    return anova_data.dropna()  # Supprimer les valeurs manquantes pour l'analyse


def effectuer_anova_erreur(anova_data):
    # Formuler le modèle : 'Erreur' est la variable dépendante, 'Méthode' est la variable indépendante
    model = ols('Erreur ~ C(Méthode)', data=anova_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    return anova_table

def afficher_explications_anova_erreur():
    st.subheader("Explications des résultats ANOVA")

    with st.expander("Somme des Carrés (sum_sq)"):
        st.markdown("""
        - **C(Méthode) - 1,063.2919**: Représente la somme des carrés due aux différences entre les moyennes des groupes (méthodes). Une valeur élevée indique une variation significative des moyennes entre les groupes par rapport à la moyenne globale, suggérant des différences potentiellement importantes entre les méthodes.
        - **Résiduelle - 3,377.2504**: Correspond à la somme des carrés des résidus, indiquant la variation à l'intérieur des groupes. Cette valeur mesure les fluctuations non expliquées par le modèle ANOVA, reflétant la variabilité naturelle ou l'erreur de mesure.
        """)

    with st.expander("Degrés de Liberté (df)"):
        st.markdown("""
        - **C(Méthode) - 5**: Le nombre de groupes (méthodes) moins un. Indique le nombre de catégories indépendantes comparées.
        - **Résiduelle - 34**: Le total des observations moins le nombre de groupes, utilisé pour estimer la variabilité au sein de chaque groupe.
        """)

    with st.expander("Statistique F et P-value"):
        st.markdown("""
        - **F - 8.4377**: Ratio de la variance moyenne entre les groupes sur la variance moyenne à l'intérieur des groupes. Une valeur élevée suggère que les moyennes de groupe diffèrent plus qu'on ne pourrait s'y attendre par hasard. Une valeur F supérieure à 1 signifie qu'il y a plus de variance entre les groupes qu'à l'intérieur des groupes. Plus cette valeur est élevée, plus il est probable que les différences entre les groupes soient réelles et non le résultat du hasard. 
        - **P-value (PR(>F)) - 0.0000**: Probabilité d'observer une valeur de F aussi extrême si l'hypothèse nulle était vraie. Une valeur proche de 0 indique qu'il est très improbable que les variations observées soient dues au hasard, permettant de rejeter l'hypothèse nulle que toutes les moyennes sont égales.
        """)

def afficher_explications_anova_vitesse():
    st.subheader("Explications des résultats ANOVA avec interprétation")

    with st.expander("Somme des Carrés (sum_sq)"):
        st.markdown("""
        - **C(Méthode) - 1,305.7491**: Cette valeur représente la somme des carrés due aux différences entre les moyennes des groupes (méthodes). Une somme élevée indique qu'il existe des variations notables entre les différentes méthodes de calcul des vitesses, y compris la méthode de référence. Cela signifie que les différentes méthodes ne produisent probablement pas les mêmes résultats, et certaines méthodes peuvent être plus précises ou fiables que d'autres.
        - **Résiduelle - 5,874.9224**: Cette somme des carrés résiduelle reflète la variabilité non expliquée par les différences entre les méthodes, c'est-à-dire la variabilité à l'intérieur des groupes. Cela pourrait indiquer des erreurs de mesure ou une variabilité naturelle dans les données.
        """)

    with st.expander("Degrés de Liberté (df)"):
        st.markdown("""
        - **C(Méthode) - 5**: Il s'agit du nombre de groupes comparés moins un, ce qui correspond aux 6 groupes comparés dans cette ANOVA (les 4 méthodes, les méthodes lignes, et la référence).
        - **Résiduelle - 72**: Ce nombre correspond aux degrés de liberté associés à la variabilité résiduelle. Il s'agit du total des observations moins le nombre de groupes, utilisé pour estimer la variabilité au sein de chaque groupe.
        """)

    with st.expander("Statistique F et P-value"):
        st.markdown("""
        - **F - 3.2005**: La statistique F est un rapport entre la variance moyenne entre les groupes et la variance moyenne à l'intérieur des groupes. Dans ce contexte, une valeur de F de 3.2005 indique que la variation entre les méthodes est plus importante que la variation à l'intérieur des méthodes. Cela suggère que certaines méthodes sont potentiellement meilleures que d'autres, mais pas de manière extrême.
        - **P-value (PR(>F)) - 0.0115**: Cette p-value est inférieure à 0.05, ce qui signifie que les différences observées entre les méthodes ne sont probablement pas dues au hasard. En d'autres termes, il est très probable qu'au moins une des méthodes de calcul des vitesses soit significativement différente des autres, y compris la méthode de référence.
        """)

    st.subheader("Interprétation des résultats")
    st.markdown("""
    Les résultats de cette ANOVA montrent qu'il existe des différences significatives entre les méthodes de calcul des vitesses, y compris par rapport à la méthode de référence (les capteurs). La statistique F indique qu'il y a plus de variabilité entre les différentes méthodes qu'au sein de chaque méthode. Cela signifie que certaines méthodes peuvent produire des résultats systématiquement différents des autres, ce qui pourrait affecter la précision et la fiabilité des mesures de vitesse.

    Plus spécifiquement, la p-value de 0.0115 indique que les différences observées sont statistiquement significatives. Vous pouvez donc conclure que toutes les méthodes ne sont pas équivalentes en termes de précision, et certaines pourraient être plus adaptées pour des applications spécifiques par rapport à la méthode de référence.
    """)

def tests_post_hoc(anova_data):
    tukey_results = pairwise_tukeyhsd(endog=anova_data['Erreur'], groups=anova_data['Méthode'], alpha=0.05)
    return tukey_results

def conclusion_test_post_hoc_vitesse():
    st.header("Méthodes Recommandées après les tests ANOVA et post-hoc")
    
    # Méthode 1 : Référence
    st.subheader("Comparaison des Méthodes avec la Référence")
    st.markdown("""
    **Performance :**
    - Aucune des méthodes testées n'a montré de différence significative avec la référence, ce qui indique que les méthodes m1 camLD, m1 camLD ligne, m2 camLD, et m2 camV ont des performances similaires à la référence en termes de vitesse mesurée.
    - Seule la méthode m1 camV montre une tendance à différer de la référence (p-value = 0.0697), mais cette différence n'est pas suffisante pour être considérée comme significative selon le seuil habituel de 0.05.
    """)

    st.markdown("---")
    
    # Méthode 2 : Comparaisons entre les Méthodes
    st.subheader("Comparaisons entre les Méthodes")
    st.markdown("""
    **Méthode m1 camLD vs m1 camV:**
    - Il y a une différence statistiquement significative entre m1 camLD et m1 camV (p-value = 0.0243), ce qui suggère que ces deux méthodes produisent des vitesses mesurées différentes.

    **Méthode m1 camLD ligne vs m1 camV:**
    - Il y a également une différence significative entre m1 camLD ligne et m1 camV (p-value = 0.0387), ce qui renforce l'idée que m1 camV diffère notablement des autres méthodes.

    **Méthode m1 camV vs m2 camLD:**
    - Une différence significative a aussi été observée entre m1 camV et m2 camLD (p-value = 0.0393), ce qui pourrait indiquer que m1 camV a tendance à produire des résultats qui diffèrent des autres méthodes testées.
    """)

    st.markdown("---")
    
    # Conclusion
    st.subheader("Conclusion Générale")
    st.markdown("""
    **Recommandations :**
    - **m1 camLD et m2 camLD** : Ces méthodes sont recommandées en raison de leur cohérence avec la référence et l'absence de différences significatives avec les autres méthodes, à l'exception de m1 camV.
    - **m1 camV** : Cette méthode montre des différences avec d'autres méthodes, notamment m1 camLD, m1 camLD ligne, et m2 camLD. Elle pourrait donc ne pas être la méthode la plus fiable pour ce type de mesure.
    - **Méthodes avec Ligne** : Les méthodes basées sur des lignes (m1 camLD ligne et m1 camV ligne) semblent avoir des performances variables, mais elles ne présentent pas de différence significative avec la référence, ce qui indique qu'elles peuvent être fiables dans des conditions spécifiques.
    """)


def conclusion_test_post_hoc():
    st.header("Méthodes Recommandées après les tests ANOVA et post-hoc")
    # Méthode 1 : m1 camLD
    st.subheader("m1 camLD")
    st.markdown("""
    **Performance :**
    - Cette méthode n'a pas de différences significatives avec plusieurs autres méthodes, sauf avec m1 camV, ce qui suggère une bonne cohérence et une meilleure fiabilité.

    **Reproductibilité :**
    - La méthode m1 camLD présente une reproductibilité moyenne, avec une variabilité modérée. Cela en fait une méthode fiable pour les mesures.
    """)

    st.markdown("---")

    # Méthode 2 : m2 camLD
    st.subheader("m2 camLD")
    st.markdown("""
    **Performance :**
    - Comme m1 camLD, cette méthode montre une bonne cohérence avec les autres méthodes (à l'exception de m1 camV), ce qui indique une bonne fiabilité.

    **Reproductibilité :**
    - La méthode m2 camLD montre également une reproductibilité moyenne, avec une variabilité modérée. Elle est donc également recommandée.
    """)

    # Séparation visuelle
    st.markdown("---")

    # Méthodes avec Ligne
    st.subheader("Méthodes avec Ligne (m1 camLD ligne, m1 camV ligne)")
    st.markdown("""
    **Performance :**
    - Ces méthodes n'ont pas montré de différences significatives avec la plupart des autres méthodes (sauf m1 camV), ce qui indique qu'elles sont également cohérentes.

    **Reproductibilité :**
    - Bien que les méthodes avec ligne aient montré de bonnes performances, leur applicabilité peut être limitée dans certaines situations (par exemple, si la ligne n'est pas bien visible).
    """)

def conclusion_repetabilite():
    st.header("Méthodes recommandées après l'étude de la répétabilité")
    st.subheader("Meilleure Méthode Théorique")

    st.markdown("""
    - **m1 camLD ligne km/h** : Cette méthode présente une reproductibilité élevée avec un faible écart-type (S = 1.36), une faible erreur standard de la moyenne (Sbarre = 0.34), et une incertitude (U = 0.73), ce qui en fait la méthode la plus fiable en termes de performance pure. Cependant, il est important de noter que les méthodes avec lignes ne sont pas toujours applicables dans toutes les situations (par exemple, si la trajectoire du patineur n'est pas exactement parallèle à la caméra), ce qui limite leur recommandation pour une utilisation généralisée.
    """)
    st.markdown("---")
    st.subheader("Méthodes Recommandées")

    st.markdown("""
    - **m1 camLD km/h** et **m2 camLD km/h** : Ces méthodes montrent une reproductibilité moyenne avec des valeurs modérées pour S, Sbarre, et U, ce qui les rend fiables et plus applicables dans une variété de contextes, contrairement aux méthodes avec lignes. Elles sont donc recommandées pour des mesures régulières. Cela montre de plus que la caméra en ligne droite est généralement efficace.

    - **m2 camV km/h** : Cette méthode présente également une reproductibilité moyenne avec un écart-type légèrement plus élevé (S = 3.80) et une incertitude modérée (U = 1.48). Bien qu'elle soit moins performante que **m2 camLD**, elle reste une option valable et applicable dans la plupart des cas.
    """)
    st.markdown("---")
    st.subheader("Méthode à Éviter")

    st.markdown("""
    - **m1 camV km/h** : Cette méthode présente une reproductibilité faible, avec un écart-type élevé (S = 9.39) et une incertitude importante (U = 3.64), ce qui indique une grande variabilité dans les résultats. Elle est donc moins fiable et non recommandée pour des mesures précises.
    """)
    st.markdown("---")
    st.subheader("Conclusion")

    st.markdown("""
    Ainsi, bien que **m1 camLD ligne km/h** soit la méthode la plus performante, ses limitations en termes d'applicabilité nous poussent à recommander les méthodes **m1 camLD km/h**, **m2 camLD km/h** et **m2 camV km/h** pour des mesures régulières et fiables. Dans le cadre d'une étude lors de prochaine compétition en patinoire on va se concentrer sur les méthodes **m2 camLD km/h** et **m2 camV km/h** car elles ne nécessite pas d'éléments extérieur pour les mesures, seulement les dimensions des patinoires.
    """)
    
def conclusion_generale():
    st.header("Conclusion sur la performance des méthodes")
    st.write("""
    Les analyses ANOVA et les tests post-hoc ont permis d'évaluer l'efficacité des différentes méthodes de mesure. Voici les principales découvertes et recommandations basées sur ces analyses.
    """)
    st.markdown("---")
    st.subheader("Méthodes les plus fiables")
    st.markdown("""
    **Méthode m2 camLD:**
    - **Fiabilité Moyenne**: Reproductibilité satisfaisante avec une variabilité modérée.
    - **Applicable**: Recommandée pour des mesures régulières.

    **Méthode m2 camV:**
    - **Fiabilité Moyenne**: Légèrement plus de variabilité que m2 camLD, mais reste une option robuste.
    - **Applicable**: Convient à la majorité des situations sans nécessiter d'éléments extérieurs.
    """)
    st.markdown("---")
    st.subheader("Méthodes avec besoins d'amélioration")
    st.markdown("""
    **Méthode m1 camV:**
    - **Variabilité Élevée**: Faible reproductibilité avec une grande dispersion des erreurs.
    - **Moins Recommandée**: Non fiable pour des mesures précises, et a montré des différences significatives avec d'autres méthodes dans les tests post-hoc.

    **Méthodes avec Ligne (m1 camLD ligne, m1 camV ligne):**
    - **Limitations d'Applicabilité**: Bien qu'elles soient performantes, elles sont limitées par des contraintes d'application pratique, mais n'ont pas montré de différences significatives avec la référence.
    """)
    st.markdown("---")
    st.subheader("Recommandations finales")
    st.markdown("""
    - **Privilégier `m2 camLD` et `m2 camV`** pour leur précision et leur applicabilité dans les mesures futures.
    - **Éviter `m1 camV`** en raison de sa faible fiabilité et des différences significatives observées avec d'autres méthodes.
    - **Utiliser avec précaution les méthodes avec ligne** uniquement dans des contextes appropriés, car elles peuvent être limitées par des contraintes spécifiques.
    """)

# =============================================================================
# PAGE: Validation Kinovéa
# =============================================================================
def page_validation_kinovea():
    st.title("Validation des valeurs de Kinovéa")

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Protocole", "Données et statistiques", "ANOVA", "Répétabilité", "Conclusion" ])

    bdd_vitesse = charger_donnees('Mesure_vitesse_patineur_v5.csv') # Pour rendre la bdd accessible depuis tous les onglets
    bdd_vitesse = nettoyer_donnees(bdd_vitesse)  # Nettoyer les données

    with tab0:
        st.header("Protocole des tests en patinoire")
        st.write("Partie  a compléter")
        st.write("Explication des différentes méthodes de calcul")

    with tab1:
        st.header("Récolte des données en patinoire")
        st.write("""Explication des détails ...""")
        st.write(bdd_vitesse.head())

        st.header("Statistiques sur les erreurs")
        generer_statistiques_graphiques(bdd_vitesse)

    with tab2:
        st.header("ANOVA sur les vitesses des 6 méthodes") #ANOVA simple en faisant la moyennne des 2 valeurs des mesures 1 et 2 pour chaque méthodes
        anova_data = preparer_donnees_anova_moyennes(bdd_vitesse)
        anova_table = effectuer_anova_vitesse(anova_data)

        st.subheader("But de l'analyse ANOVA")
        st.markdown("""
        L'intérêt de réaliser une ANOVA dans ce contexte est de déterminer si les différentes méthodes de mesure produisent des résultats qui diffèrent de manière significative les uns des autres. En identifiant des différences significatives, vous pouvez conclure avec certitude que certaines méthodes sont peut-être plus fiables ou précises que d'autres.
        """)
        # Expliquer les colonnes du tableau ANOVA
        st.markdown("""
        - **C(Méthode)**: La variance entre les groupes, c'est-à-dire la variance due aux différences entre les différentes méthodes de mesure comparées.
        - **Residual**: La variance résiduelle ou variance à l'intérieur des groupes. Cela correspond à la somme des carrés des résidus, c'est-à-dire à la variabilité non expliquée par les différences entre les méthodes.
        - **DF**: Degrés de liberté associés à la source de variation.
        - **sum_sq**: Somme des carrés due à chaque source de variation.
        - **mean_sq**: Moyenne des carrés, obtenue en divisant la somme des carrés par les degrés de liberté.
        - **F**: Statistique F, calculée en divisant la moyenne des carrés entre les groupes par la moyenne des carrés à l'intérieur des groupes.
        - **PR(>F)**: P-value associée au test statistique F, indiquant la probabilité de voir de telles données si les moyennes de toutes les méthodes étaient identiques.
        """)

        st.write('Données utilisées pour le tableau ANOVA:')
        st.write(anova_data)

        st.subheader("Tableau ANOVA")
        st.write(anova_table.style.format("{:.4f}"))
        # st.write(anova_table)

        # Interprétation des résultats
        if anova_table['PR(>F)'][0] < 0.05:
            st.success("Les résultats indiquent une différence statistiquement significative entre les méthodes, suggérant que certaines méthodes peuvent être plus fiables ou précises que d'autres.")
        else:
            st.error("Aucune différence significative n'a été trouvée entre les méthodes.")
        afficher_explications_anova_vitesse()

        st.header("Tests post-hoc (Tukey)")
        post_hoc_results = tests_post_hoc_vitesse(anova_data)

        # Pour visualiser le tableau Tukey de manière plus lisible
        tukey_table = pd.DataFrame(data=post_hoc_results._results_table.data[1:], columns=post_hoc_results._results_table.data[0])
        st.write("Tableau complet des comparaisons post-hoc Tukey :")
        st.dataframe(tukey_table.style.applymap(lambda x: 'background-color: yellow' if isinstance(x, str) and 'True' in x else ''))

        # Affichage détaillé des résultats post-hoc
        st.subheader("Détails des résultats test post-hoc")
        for _, row in tukey_table.iterrows():
            with st.expander(f"Comparaison entre {row['group1']} et {row['group2']}"):
                st.markdown("""
                **Différence moyenne (meandiff):** {:.4f} km/h  
                **P-value ajustée (p-adj):** {:.4f}  
                **Intervalle de confiance:** de {:.4f} à {:.4f} km/h  
                **Différence significative:** {}
                """.format(row['meandiff'], row['p-adj'], row['lower'], row['upper'], "Oui" if row['reject'] else "Non"))

                if row['reject']:
                    st.success("Il y a une différence statistiquement significative entre les groupes.")
                else:
                    st.info("Aucune différence significative détectée entre les groupes.")

        conclusion_test_post_hoc_vitesse()


        # Ce que j'avais fait avant avec que les erreurs et pas les vitesses donc j'avais pas de groupe référence pour comparer 
        # Car la comparaison avec le groupe ref s'était faite plus tôt pendant le calcul de l'erreur sur Excel
        if st.checkbox('Afficher ANOVA et post-hoc sur les **erreurs** des 6 méthodes'):
            st.header("ANOVA sur les erreurs des 6 méthodes")
            anova_data2 = preparer_donnees_anova_erreur(bdd_vitesse)
            anova_table2 = effectuer_anova_erreur(anova_data2)

            st.subheader("Tableau ANOVA")
            st.write(anova_table2.style.format("{:.4f}"))

            # Interprétation des résultats
            if anova_table2['PR(>F)'][0] < 0.05:
                st.success("Les résultats indiquent une différence statistiquement significative entre les méthodes, suggérant que certaines méthodes peuvent être plus fiables ou précises que d'autres.")
            else:
                st.error("Aucune différence significative n'a été trouvée entre les méthodes.")

            afficher_explications_anova_erreur()


            st.header("Tests post-hoc (Tukey)")
            post_hoc_results2 = tests_post_hoc(anova_data2)

            # Pour visualiser le tableau Tukey de manière plus lisible
            tukey_table2 = pd.DataFrame(data=post_hoc_results2._results_table.data[1:], columns=post_hoc_results2._results_table.data[0])
            st.dataframe(tukey_table2.style.applymap(lambda x: 'background-color: yellow' if isinstance(x, str) and 'True' in x else ''))

            st.subheader("Détails des résultats test post-hoc")
            for _, row in tukey_table2.iterrows():
                with st.expander(f"Comparaison entre {row['group1']} et {row['group2']}"):
                    st.markdown("""
                    **Différence moyenne (meandiff):** {:.4f} km/h  
                    **P-value ajustée (p-adj):** {:.4f}  
                    **Intervalle de confiance:** de {:.4f} à {:.4f} km/h  
                    **Différence significative:** {}
                    """.format(row['meandiff'], row['p-adj'], row['lower'], row['upper'], "Oui" if row['reject'] else "Non"))

                    if row['reject']:
                        st.success("Il y a une différence statistiquement significative entre les groupes.")
                    else:
                        st.info("Aucune différence significative détectée entre les groupes.")
            
            conclusion_test_post_hoc()


    with tab3:
        st.header("Analyse de la répétabilité des méthodes")
        analyse_repetabilite(bdd_vitesse)
        conclusion_repetabilite()

    with tab4:
        conclusion_generale()



# =============================================================================
# METHODES PAGE MESURES ET ANALYSES APRES ETUDE : 
# =============================================================================


def get_t_value(degrees_of_freedom, confidence_level=0.95):
    return t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

def calculer_repetabilite_mesures_analyses(data):
    if len(data) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan  # Retourne des NaN pour toutes les valeurs
    
    mean_speed = np.nanmean(data)
    S = np.nanstd(data, ddof=1)
    Sbarre = S / np.sqrt(len(data) - np.isnan(data).sum())
    df = len(data) - np.isnan(data).sum() - 1

    if df > 0:
        t_value = get_t_value(df)
        U = t_value * Sbarre
    else:
        t_value, U = np.nan, np.nan

    return mean_speed, S, Sbarre, U, df, t_value

# Afficher un graphique de répétabilité
def plot_repeatability(data_dict):
    fig, ax = plt.subplots()
    for observer, data in data_dict.items():
        mean_speed, S, Sbarre, U, df, t_value = calculer_repetabilite_mesures_analyses(data)
        color = 'green' if U < 5 else 'red'  # Seuil d'incertitude pour la couleur
        ax.bar(observer, mean_speed, yerr=S, color=color, label=f'U={U:.2f}')
    plt.ylabel('Moyenne des vitesses (km/h)')
    plt.title('Répétabilité par observateur et course')
    plt.legend()
    st.pyplot(fig)



def calculer_reproductibilite_approximee(*datasets):
    valid_data = [data for data in datasets if len(data) > 1]
    single_measure_datasets = [data for data in datasets if len(data) == 1]

    all_means = [np.nanmean(data) for data in valid_data] + [data[0] for data in single_measure_datasets]
    mean_all = np.nanmean(all_means)

    if valid_data:
        std_devs = [np.nanstd(data, ddof=1) for data in valid_data]
        Sr = np.sqrt(np.nanmean([s**2 for s in std_devs]))
        mean_std_dev = np.nanmean(std_devs) if std_devs else np.nan
    else:
        mean_std_dev, Sr = np.nan, np.nan

    estimated_std_devs = [mean_std_dev] * len(single_measure_datasets)
    variance_inter = np.var(all_means, ddof=1)
    n_min = min([len(data) for data in datasets])
    SR = np.sqrt(variance_inter + ((n_min - 1) / n_min) * Sr**2)
    U = 2.571 * SR

    return mean_all, Sr, SR, variance_inter, U

def charger_et_nettoyer_donnees_mesures_analyses(fichier):
    try:
        data = pd.read_csv(fichier, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(fichier, delimiter=';', encoding='latin1')
    data.replace(to_replace=r',', value='.', regex=True, inplace=True)
    cols_vitesse = [col for col in data.columns if 'Impact speed' in col]
    for col in cols_vitesse:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=cols_vitesse, how='all', inplace=True)
    return data


def filtrer_et_afficher_donnees(data):
    # Filtrer les données pour inclure les lignes où au moins une colonne 'Impact speed' est non nulle
    cols_a_verifier = [col for col in data.columns if 'Impact speed' in col]
    data_filtrée = data.dropna(subset=cols_a_verifier, how='all')
    
    # Colonnes à afficher
    colonnes_affichage = [
        'Competition Name', 'City', 'Race', 'Year', 'Head area',
        *cols_a_verifier
    ]
    # Afficher le tableau filtré
    return data_filtrée[colonnes_affichage]


def evaluer_ecarts_significatifs(data, seuil=10):
    # Résultats pour les courses avec écarts significatifs
    resultats_significatifs = {}

    # Itérer sur chaque course et observer
    for index, row in data.iterrows():
        course = row['Race']
        marie_vitesse = row['Impact speed Marie m1'] if 'Impact speed Marie m1' in row and pd.notna(row['Impact speed Marie m1']) else None
        if marie_vitesse is None:
            continue  # Passer si Marie n'a pas de mesure

        # Calculer la moyenne des autres observateurs présents
        autres_vitesses = []
        for obs in ['Laurianne', 'Lisa']:
            for i in range(1, 4):  # Supposer jusqu'à trois mesures
                key = f'Impact speed {obs} m{i}'
                if key in row and pd.notna(row[key]):
                    autres_vitesses.append(row[key])

        if not autres_vitesses:
            continue  # Passer si aucun autre observateur n'a de mesure

        moyenne_autres = np.mean(autres_vitesses)
        ecart = abs(marie_vitesse - moyenne_autres)

        # Vérifier si l'écart est supérieur au seuil
        if ecart > seuil:
            resultats_significatifs[course] = {
                'Vitesse Unique Marie': marie_vitesse,
                'Moyenne des Autres': moyenne_autres,
                'Ecart': ecart,
                'Est Proche': ecart < moyenne_autres * 0.1
            }

    return resultats_significatifs

# Vous pouvez utiliser cette fonction après avoir chargé et préparé vos données.

def afficher_resultats_evaluation(resultats):
    # Assumons que `resultats` est un DataFrame avec les colonnes nécessaires
    courses = resultats['Race'].unique()
    fig, axs = plt.subplots(len(courses), 1, figsize=(10, 5 * len(courses)))
    
    if len(courses) == 1:
        axs = [axs]  # Assurez-vous que axs est toujours une liste pour la cohérence

    for ax, course in zip(axs, courses):
        data_course = resultats[resultats['Race'] == course]
        distances = data_course['Distance à la Moyenne']
        est_proche = data_course['Est Proche']
        labels = [f"{obs}" for obs in data_course['Observateur']]
        
        # Utiliser une couleur différente en fonction de la proximité
        colors = ['green' if proche else 'red' for proche in est_proche]
        
        ax.bar(labels, distances, color=colors)
        ax.set_title(f'Race: {course}')
        ax.set_ylabel('Distance à la moyenne (km/h)')
        ax.set_xlabel('Observateur')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def page_mesures_analyses_apres_etude():
    st.title("Mesures et analyses")
    st.write("## Comment ont été fait les calculs ?")
    st.write("Analyse de vidéos sur Kinovéa")

    data = charger_et_nettoyer_donnees_mesures_analyses('Falls data and analysis v2.csv')
    data['Competition_Combined'] = data['Competition Name'] + ' ' + data['City'] + ' ' + data['Year'].astype(str)
    competitions = data['Competition_Combined'].unique()


    choix_competition = st.selectbox('Choisissez une compétition:', competitions)
    filtered_data = data[data['Competition_Combined'] == choix_competition]
    races = filtered_data['Race'].unique()
    choix_course = st.selectbox('Choisissez une course:', races)
    selected_data = filtered_data[filtered_data['Race'] == choix_course]
    observateurs = ['Marie', 'Laurianne', 'Lisa']

    selected_for_repe = st.multiselect('Choisissez les observateurs pour la répétabilité:', observateurs, default=observateurs)
    if selected_for_repe:
        results_repe = afficher_resultats_repe(selected_data, selected_for_repe)
        st.write("### Résultats de répétabilité")
        for observer, res in results_repe.items():
            st.write(f"**{observer}:**")
            st.write(f"Moyenne des vitesses: {res['Moyenne des vitesses']:.2f} km/h")
            st.write(f"Écart type: {res['Écart type']:.2f}")
            st.write(f"Erreur standard de la moyenne: {res['Erreur standard de la moyenne']:.2f}")
            st.write(f"U (Incertitude avec facteur de couverture t): {res['U (Incertitude avec facteur de couverture t)']:.2f} km/h")

    tableau_filtre = filtrer_et_afficher_donnees(data)
    st.write(tableau_filtre)
    st.write("Nombre de vitesses d'impact mesurées :")
    st.write(len(tableau_filtre))

    # Regrouper les données par observateur et par course
    grouped_data = {}
    for observer in ['Marie', 'Laurianne', 'Lisa']:
        observer_data = data[[col for col in data.columns if observer in col]].dropna()
        grouped_data[observer] = observer_data.values.flatten()  # Créer un array plat de toutes les mesures

    
    # Préparation des données pour le graphique de répétabilité et pour le tableau
    results = {obs: {'Mean Speed': [], 'Uncertainty': []} for obs in observateurs}
    course_labels = []  # Liste pour les noms de course
    mauvaises_repe = []  # Pour stocker les mauvaises répétabilités

    for race in races:
        course_data = filtered_data[filtered_data['Race'] == race]
        course_labels.append(race)  # Ajouter le nom de la course à la liste
        for observer in observateurs:
            # Chercher toutes les colonnes qui correspondent à cet observateur
            obs_data = [course_data[f'Impact speed {observer} m{i}'] for i in range(1, 4) if f'Impact speed {observer} m{i}' in course_data.columns]
            # Combiner toutes les valeurs non nulles
            speeds = pd.concat(obs_data).dropna().values
            if len(speeds) > 0:
                mean_speed, S, Sbarre, U, df, t_value = calculer_repetabilite_mesures_analyses(speeds)
                results[observer]['Mean Speed'].append(mean_speed)
                results[observer]['Uncertainty'].append(U)

                # Vérification des mauvaises répétabilités (si l'incertitude est trop élevée)
                if U > 10:  # Seuil arbitraire pour l'exemple, à ajuster selon le besoin
                    mauvaises_repe.append({
                        'Race': race,
                        'Observer': observer,
                        'Mean Speed': mean_speed,
                        'Uncertainty': U
                    })
            else:
                results[observer]['Mean Speed'].append(np.nan)  # Ajouter un NaN si pas de données
                results[observer]['Uncertainty'].append(np.nan)

    # Création du DataFrame pour les résultats (bonnes et mauvaises répétabilités incluses)
    df_results = pd.DataFrame({
        'Race': course_labels,
        **{f'Mean Speed {obs}': results[obs]['Mean Speed'] for obs in observateurs},
        **{f'Uncertainty {obs}': results[obs]['Uncertainty'] for obs in observateurs}
    })

    # Afficher le tableau des résultats
    st.write("### Tableau des résultats (toutes les répétabilités)")
    st.write(df_results)

    # Création du graphique des répétabilités
    bar_width = 0.25
    index = np.arange(len(course_labels))

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, observer in enumerate(observateurs):
        ax.bar(index + i * bar_width, df_results[f'Mean Speed {observer}'], bar_width, label=f'Mean Speed {observer}', yerr=df_results[f'Uncertainty {observer}'], capsize=5)

    ax.set_xlabel('Course')
    ax.set_ylabel('Mean Speed (km/h)')
    ax.set_title('Mean Speed and Uncertainty by Observer and Course')
    ax.set_xticks(index + bar_width / len(observateurs) * (len(observateurs) - 1))
    ax.set_xticklabels(course_labels)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Afficher le tableau des mauvaises répétabilités
    if mauvaises_repe:
        st.write("### Tableau des mauvaises répétabilités")
        df_mauvaises_repe = pd.DataFrame(mauvaises_repe)
        st.write(df_mauvaises_repe)


# Function to color cells based on uncertainty
def highlight_cells(val):
    if pd.isna(val):
        color = ''
    elif val <= 10:
        color = 'background-color: green'
    elif 10 < val <= 30:
        color = 'background-color: yellow'
    else:
        color = 'background-color: red'
    return color


def page_mesures_analyses_apres_etude():
    st.title("Mesures et analyses - Répétabilité et Reproductibilité des vitesses")
    data = charger_et_nettoyer_donnees_mesures_analyses('Falls data and analysis v2.csv')

    data['Competition_Combined'] = data['Competition Name'] + ' ' + data['City'] + ' ' + data['Year'].astype(str)
    competitions = data['Competition_Combined'].unique()
    observateurs = ['Marie', 'Laurianne', 'Lisa']

    rows_for_table = []

    # Parcourir chaque compétition et course
    for competition in competitions:
        filtered_data = data[data['Competition_Combined'] == competition]
        races = filtered_data['Race'].unique()

        for race in races:
            # Filtrer les données pour la course actuelle
            course_data = filtered_data[filtered_data['Race'] == race]

            # Calculer les résultats de répétabilité pour chaque observateur
            results_repe = afficher_resultats_repe(course_data, observateurs)
            
            # Construire la ligne de tableau pour chaque course
            row = {
                'Competition': competition,
                'Race': race
            }

            for obs in observateurs:
                row[f"Mean Speed {obs}"] = results_repe[obs]["Moyenne des vitesses"]
                row[f"Uncertainty {obs}"] = results_repe[obs]["U (Incertitude avec facteur de couverture t)"]

            rows_for_table.append(row)

    # Créer le DataFrame récapitulatif
    df_results = pd.DataFrame(rows_for_table)

    # Highlight cells based on uncertainty
    def highlight_cells(value):
        if pd.isna(value):
            return 'background-color: gray'
        elif value <= 10:
            return 'background-color: green'
        elif 10 < value <= 30:
            return 'background-color: yellow'
        else:
            return 'background-color: red'

    # Apply the color styling to the uncertainty columns
    styled_df = df_results.style.applymap(highlight_cells, subset=[f'Uncertainty {obs}' for obs in observateurs])

    st.write("### Tableau des résultats de répétabilité")
    st.write(styled_df)

    # Section pour le choix de la compétition, course et observateurs
    choix_competition = st.selectbox('Choisissez une compétition:', competitions)
    filtered_data = data[data['Competition_Combined'] == choix_competition]
    races = filtered_data['Race'].unique()
    choix_course = st.selectbox('Choisissez une course:', races)
    selected_data = filtered_data[filtered_data['Race'] == choix_course]

    selected_for_repe = st.multiselect('Choisissez les observateurs pour la répétabilité:', observateurs, default=observateurs)
    if selected_for_repe:
        results_repe = afficher_resultats_repe(selected_data, selected_for_repe)
        st.write("### Résultats de répétabilité")
        for observer, res in results_repe.items():
            st.write(f"**{observer}:**")
            st.write(f"Moyenne des vitesses: {res['Moyenne des vitesses']:.2f} km/h")
            st.write(f"Écart type: {res['Écart type']:.2f}")
            st.write(f"Erreur standard de la moyenne: {res['Erreur standard de la moyenne']:.2f}")
            st.write(f"U (Incertitude avec facteur de couverture t): {res['U (Incertitude avec facteur de couverture t)']:.2f} km/h")

    # Filtrer les données et afficher le nombre de mesures
    tableau_filtre = filtrer_et_afficher_donnees(data)
    st.write(tableau_filtre)
    st.write("Nombre de vitesses d'impact mesurées :")
    st.write(len(tableau_filtre))

    # Section des valeurs uniques
    unique_values = []
    non_unique_data = data.copy()

    for index, row in data.iterrows():
        # Identifying unique measure for Marie
        if pd.notna(row.get('Impact speed Marie m1')) and pd.isna(row.get('Impact speed Marie m2')) and pd.isna(row.get('Impact speed Marie m3')):
            unique_value = row['Impact speed Marie m1']
            mean_lisa = row.get('Impact speed mean Lisa')
            mean_laurianne = row.get('Impact speed mean Laurianne')

            if pd.notna(mean_lisa) and pd.notna(mean_laurianne):
                diff_lisa = abs(unique_value - mean_lisa)
                diff_laurianne = abs(unique_value - mean_laurianne)
                unique_values.append({
                    "Competition": row['Competition_Combined'],
                    "Race": row['Race'],
                    "Unique Marie": unique_value,
                    "Mean Lisa": mean_lisa,
                    "Mean Laurianne": mean_laurianne,
                    "Diff Lisa": diff_lisa,
                    "Diff Laurianne": diff_laurianne
                })
            
            # Remove unique values from data used for other reproducibility calculations
            non_unique_data.drop(index, inplace=True)

    # Display the table with unique values and differences
    if unique_values:
        df_unique = pd.DataFrame(unique_values)
        st.write("### Tableau des mesures uniques de Marie et écarts avec les moyennes")
        st.write(df_unique)

    # Calcul de la reproductibilité pour chaque course
    rows_for_repro_table = []
    for competition in competitions:
        filtered_data = data[data['Competition_Combined'] == competition]
        races = filtered_data['Race'].unique()

        for race in races:
            course_data = filtered_data[filtered_data['Race'] == race]
            
            # Vérifier les mesures uniques pour Marie
            unique_mar_data = course_data[pd.notna(course_data['Impact speed Marie m1']) & 
                                          pd.isna(course_data['Impact speed Marie m2']) & 
                                          pd.isna(course_data['Impact speed Marie m3'])]

            non_unique_data = course_data.drop(unique_mar_data.index)

            row_repro = {
                'Competition': competition,
                'Race': race
            }

            # Calcul de la reproductibilité pour les valeurs non uniques
            repro_non_unique = calculer_reproductibilite_non_uniques(non_unique_data, observateurs)
            if repro_non_unique:
                for obs, res in repro_non_unique.items():
                    row_repro[f"Repro Mean {obs}"] = res["Global Mean Speed"]
                    row_repro[f"Repro Std Dev {obs}"] = res["Standard Deviation"]
                    row_repro[f"Repro U {obs}"] = res["Uncertainty"]

            # Calcul de la reproductibilité pour les valeurs uniques de Marie
            if not unique_mar_data.empty:
                repro_unique = calculer_reproductibilite_uniques(unique_mar_data)
                row_repro["Repro Unique Marie"] = repro_unique["Global Mean Diff"]

            rows_for_repro_table.append(row_repro)

    # Créer le DataFrame récapitulatif de la reproductibilité
    df_repro_results = pd.DataFrame(rows_for_repro_table)


    # Appel de la méthode de calcul
    df_global_repro = calculer_reproductibilite_globale(data, observateurs)

    # Affichage du tableau des reproductibilités globales
    st.write("### Tableau récapitulatif des reproductibilités globales")
    st.write(df_global_repro)



def afficher_resultats_repe(data, observateurs):
    results = {}
    for obs in observateurs:
        cols = [col for col in data.columns if f'Impact speed {obs}' in col]
        speed_measurements = data[cols].values.flatten()
        mean_speed, S, Sbarre, U, df, t_value = calculer_repetabilite_mesures_analyses(speed_measurements)
        results[obs] = {
            "Moyenne des vitesses": mean_speed,
            "Écart type": S,
            "Erreur standard de la moyenne": Sbarre,
            "U (Incertitude avec facteur de couverture t)": U,
            "Degrés de liberté": df,
            "Valeur t": t_value
        }
    return results

def calculer_reproductibilite_non_uniques(data, observateurs):
    vitesses = {}
    for obs in observateurs:
        obs_data = []
        for i in range(1, 4):  # Assuming up to 3 measurements
            col_name = f"Impact speed {obs} m{i}"
            if col_name in data.columns:
                obs_data.extend(data[col_name].dropna().values)
        
        if obs_data:
            vitesses[obs] = np.array(obs_data)

    repro_results = {}
    if len(vitesses) >= 2:
        # Calculer la reproductibilité pour chaque observateur
        for obs, speeds in vitesses.items():
            mean_speed = np.mean(speeds)
            std_dev = np.std(speeds)
            uncertainty = 2 * std_dev  # Exemple pour U (ajuster si nécessaire)

            repro_results[obs] = {
                "Global Mean Speed": mean_speed,
                "Standard Deviation": std_dev,
                "Uncertainty": uncertainty
            }
    return repro_results

def calculer_reproductibilite_uniques(unique_mar_data):
    # Calculer la reproductibilité des valeurs uniques de Marie
    diffs = unique_mar_data['Impact speed Marie m1'].values
    mean_diff = np.mean(diffs)
    return {
        "Global Mean Diff": mean_diff
    }

def calculer_reproductibilite_globale(data, observateurs):
    # Regrouper toutes les mesures pour chaque course
    rows_for_global_repro = []
    
    for competition in data['Competition_Combined'].unique():
        filtered_data = data[data['Competition_Combined'] == competition]
        races = filtered_data['Race'].unique()
        
        for race in races:
            # Combiner toutes les mesures des observateurs pour la course
            course_data = filtered_data[filtered_data['Race'] == race]
            all_speeds = []
            unique_marie_speed = None

            for obs in observateurs:
                for i in range(1, 4):  # Supposant qu'il y a jusqu'à 3 mesures par observateur
                    col_name = f'Impact speed {obs} m{i}'
                    if col_name in course_data.columns:
                        speeds = course_data[col_name].dropna().values
                        all_speeds.extend(speeds)
                        
                        # Vérifier si Marie a une seule mesure unique
                        if obs == 'Marie' and len(speeds) == 1:
                            unique_marie_speed = speeds[0]
            
            # Calculer la moyenne et l'écart-type globaux
            if len(all_speeds) > 1:  # Assurez-vous d'avoir au moins deux mesures
                mean_global = np.mean(all_speeds)
                std_dev_global = np.std(all_speeds, ddof=1)  # ddof=1 pour un échantillon
                
                # Calcul de l'incertitude élargie
                U_global = std_dev_global * 2  # Facteur de couverture (2 pour 95%)
                
                # Si une valeur unique de Marie a été trouvée, estimer un écart type spécifique
                if unique_marie_speed is not None:
                    # Estimer l'écart-type en utilisant la différence entre la valeur unique de Marie et la moyenne globale
                    estimated_std_dev_marie = abs(unique_marie_speed - mean_global)
                else:
                    estimated_std_dev_marie = np.nan
                
                # Ajouter les résultats au tableau
                rows_for_global_repro.append({
                    'Competition': competition,
                    'Race': race,
                    'Global Mean Speed': mean_global,
                    'Global Std Dev': std_dev_global,
                    'Global U': U_global,
                    'Estimated Std Dev Marie': estimated_std_dev_marie
                })
    
    # Créer un DataFrame des résultats globaux
    df_global_repro = pd.DataFrame(rows_for_global_repro)
    
    return df_global_repro

def calculer_reproductibilite_globale(data, observateurs):
    # Regrouper toutes les mesures pour chaque course
    rows_for_global_repro = []
    
    for competition in data['Competition_Combined'].unique():
        filtered_data = data[data['Competition_Combined'] == competition]
        races = filtered_data['Race'].unique()
        
        for race in races:
            # Combiner toutes les mesures des observateurs pour la course
            course_data = filtered_data[filtered_data['Race'] == race]
            all_speeds = []
            speeds_by_observer = {}
            unique_marie_speed = None

            for obs in observateurs:
                speeds_by_observer[obs] = []  # Initialize the list for each observer
                for i in range(1, 4):  # Supposant qu'il y a jusqu'à 3 mesures par observateur
                    col_name = f'Impact speed {obs} m{i}'
                    if col_name in course_data.columns:
                        speeds = course_data[col_name].dropna().values
                        all_speeds.extend(speeds)
                        speeds_by_observer[obs].extend(speeds)
                        
                        # Vérifier si Marie a une seule mesure unique
                        if obs == 'Marie' and len(speeds) == 1:
                            unique_marie_speed = speeds[0]
            
            # Calculer la moyenne et l'écart-type globaux
            if len(all_speeds) > 1:  # Assurez-vous d'avoir au moins deux mesures
                mean_global = np.mean(all_speeds)
                std_dev_global = np.std(all_speeds, ddof=1)  # ddof=1 pour un échantillon
                
                # Calcul de l'écart-type de répétabilité (Sr) et reproductibilité (SR)
                std_devs = [np.std(speeds, ddof=1) for speeds in speeds_by_observer.values() if len(speeds) > 1]
                Sr = np.mean(std_devs) if std_devs else np.nan
                SR = std_dev_global
                
                # Calcul de l'incertitude élargie
                U_global = SR * 2  # Facteur de couverture (2 pour 95%)
                
                # Si une valeur unique de Marie a été trouvée, estimer un écart type spécifique
                if unique_marie_speed is not None:
                    # Estimer l'écart-type en utilisant la différence entre la valeur unique de Marie et la moyenne globale
                    estimated_std_dev_marie = abs(unique_marie_speed - mean_global)
                else:
                    estimated_std_dev_marie = np.nan
                
                # Ajouter les résultats au tableau
                rows_for_global_repro.append({
                    'Competition': competition,
                    'Race': race,
                    'Global Mean Speed': mean_global,
                    'Global Std Dev': std_dev_global,
                    'Global U': U_global,
                    'Estimated Std Dev Marie': estimated_std_dev_marie,
                    'Repeatability Std Dev (Sr)': Sr,
                    'Reproducibility Std Dev (SR)': SR
                })
    
    # Créer un DataFrame des résultats globaux
    df_global_repro = pd.DataFrame(rows_for_global_repro)
    
    return df_global_repro


def extraire_vitesses_maximales_et_moyennes(data):
    # Création de deux séries pour les vitesses maximales et les moyennes maximales
    max_speeds = pd.Series(dtype=float)
    mean_max_speeds = pd.Series(dtype=float)
    
    for col in data.columns:
        if 'Impact speed' in col and 'mean' not in col:
            max_speed = data.groupby('Head area')[col].max()
            max_speeds = pd.concat([max_speeds, max_speed.rename(col)], axis=1)
        elif 'Impact speed mean' in col:
            mean_speed = data.groupby('Head area')[col].max()
            mean_max_speeds = pd.concat([mean_max_speeds, mean_speed.rename(col)], axis=1)
    
    # Combinaison des deux séries en un seul DataFrame et exclusion des zones non pertinentes
    combined_speeds = pd.DataFrame({
        'Max Speed': max_speeds.max(axis=1),
        'Max Mean Speed': mean_max_speeds.max(axis=1)
    }).drop(index=['None', 'No data'])  # Exclure 'None' et 'No data'

    return combined_speeds

def plot_max_speeds(combined_speeds, title, ylabel):
    ax = combined_speeds.plot(kind='bar', figsize=(12, 6))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Head Area')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
# =============================================================================
# MENU DU STREAMLIT
# ============================================================================= 
st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#D6EAF8,#D6EAF8);
            color: white;
        }
        .css-1v3fvcr, .css-1aumxhk {
            background-color: #1A5276;
            color: white;
            border: none;
        }
        .css-1aumxhk:hover {
            background-color: #148F77;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.title("Menu")  # Layout the sidebar
page = st.sidebar.radio(
    "Choisissez une page",
    ["Accueil", "Statistiques", "Mesures avant l'étude de la méthode de mesure", "Validation des valeurs de Kinovéa", "Mesures après l'étude de la méthode de mesure"],
    index=0,
    key='page_select'
)

# Display the selected page using a dictionary lookup to map between page name and function
pages = {
    "Accueil": page_home,
    "Statistiques": page_statistiques,
    "Mesures avant l'étude de la méthode de mesure": page_mesures_analyses,
    "Validation des valeurs de Kinovéa": page_validation_kinovea,
    "Mesures après l'étude de la méthode de mesure": page_mesures_analyses_apres_etude
}

# Call the app function associated with the selected page
if page in pages:
    pages[page]()

# home.py
def show():
    st.title("Accueil Page")
    #st.write("Welcome to the multi-page application.")

def show():
    st.title("Statistiques")

def show():
    st.title("Mesures avant l'étude de la méthode de mesure")

def show():
    st.title("Validation des valeurs de Kinovéa")

def show():
    st.title("Mesures après l'étude de la méthode de mesure")