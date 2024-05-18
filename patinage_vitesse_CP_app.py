# Importing the necessary libraries :
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from io import StringIO
import os
import re

# Définir le répertoire de travail au dossier du script
script_dir = os.path.dirname(__file__)  # Obtient le dossier où se trouve le script
os.chdir(script_dir)  # Change le répertoire de travail au dossier du script

bdd = pd.read_csv('BDD_virgules.csv', sep=';', engine='python')

st.set_page_config(page_title="Patinage de vitesse courte piste", page_icon="⛸️") # Set the page title and icon using st.set_page_config

# Define a function for each page
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

    def display_fall_statistics(bdd):
        # Filtrer les données pour compter seulement les lignes où une chute est survenue
        fall_data = bdd[bdd['Fall / Out of track'] == 'Fall']

        # Compter le nombre de chutes par pays
        fall_counts = fall_data['Country'].value_counts().reset_index()
        fall_counts.columns = ['Country', 'Number of Falls']

        # Créer un graphique à barres pour le nombre de chutes par pays
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Number of Falls', y='Country', data=fall_counts, palette='viridis')
        plt.title('Nombre de Chutes par Pays')
        plt.xlabel('Nombre de Chutes')
        plt.ylabel('Pays')
        st.pyplot(plt)
    
    display_fall_statistics(bdd)
   
    # Footer
    st.markdown('---')
    st.markdown('**Author:** INS Québec (https://www.insquebec.org/)')
    st.markdown('**Date:** mai 2024')


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

    def display_impact_speed_distribution(bdd):
        # Créer un histogramme pour montrer la distribution de la vitesse d'impact
        plt.figure(figsize=(10, 8))
        sns.histplot(bdd['Impact speed mean'], kde=True, color='blue', bins=10)
        plt.title('Distribution of Impact Speed Mean')
        plt.xlabel('Impact Speed Mean (km/h)')
        plt.ylabel('Frequency')
        plt.grid(True)
        st.pyplot(plt)
    
    display_impact_speed_distribution(bdd)

    def display_impact_speed_by_head_impact(bdd):
        # Créer un box plot pour montrer la vitesse d'impact en fonction du type de choc à la tête
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='Head impact', y='Impact speed mean', data=bdd, palette='viridis')
        plt.title('Impact Speed Mean by Head Impact Type')
        plt.xlabel('Head Impact Type')
        plt.ylabel('Impact Speed Mean (km/h)')
        st.pyplot(plt)

    display_impact_speed_by_head_impact(bdd)

    st.write("""Le graphique représente la distribution de la vitesse d'impact moyenne (en km/h) en fonction du type d'impact à la tête lors de chutes en patinage de vitesse courte piste. Trois catégories d'impact à la tête sont observées : aucun impact (None), impact indirect, et impact direct, avec une catégorie supplémentaire pour les cas où aucune donnée n'est disponible (No data). 
             À partir du graphique, on peut observer que la vitesse d'impact moyenne pour les chutes avec un impact direct à la tête est nettement plus élevée comparée à celles avec un impact indirect ou aucun impact. Les chutes avec impact direct montrent aussi une plus grande variabilité dans les vitesses d'impact, comme l'indique la longueur de la boîte et les extrémités des moustaches. En revanche, les chutes sans aucun impact à la tête présentent les vitesses les plus basses et la variabilité la plus faible.
             Ce graphique suggère que les impacts directs à la tête lors des chutes sont généralement associés à des vitesses d'impact plus élevées, ce qui pourrait indiquer un risque accru de blessures graves.
             """)

def page_mesures():
    st.title("Comment ont été fait les calculs ?")
    st.write("Analyse de vidéos sur Kinovéa")

# Menu
st.sidebar.title("Menu") # Layout the sidebar
page = st.sidebar.radio("Choisissez une page", ["Accueil", "Statistiques", "Mesures"])

# Display the selected page using a dictionary lookup to map between page name and function
pages = {
    "Accueil": page_home,
    "Statistiques": page_statistiques,
    "Mesures": page_mesures
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
    #st.write("Here you can try your dress on")