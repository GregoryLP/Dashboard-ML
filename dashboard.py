import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import zipfile

with zipfile.ZipFile("opendata-vitesse-2021-01-01-2021-12-31.zip", "r") as zip_ref:
    zip_ref.extractall()
df_vitesse = pd.read_csv("opendata-vitesse-2021-01-01-2021-12-31.csv", sep=';')

nbrexces = df_vitesse['mesure'] - df_vitesse['limite']
nbrexces = nbrexces[nbrexces >= 0]

mesure_0_10 = nbrexces[(nbrexces >= 0) & (nbrexces < 10)]
mesure_10_20 = nbrexces[(nbrexces >= 10) & (nbrexces < 20)]
mesure_20_30 = nbrexces[(nbrexces >= 20) & (nbrexces < 30)]
mesure_30_40 = nbrexces[(nbrexces >= 30) & (nbrexces < 40)]
mesure_supp_40 = nbrexces[nbrexces >= 40]

mesures = [mesure_0_10, mesure_10_20, mesure_20_30, mesure_30_40, mesure_supp_40]
plages = ['0-10', '10-20', '20-30', '30-40', 'Sup à 40']

st.title('Dashboard d\'analyse des excès de vitesse')

# histogramme des vitesses mesurées
st.subheader('Histogramme des vitesses mesurées')
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_vitesse['mesure'], bins=200, color='skyblue')
ax.set_xlabel('Vitesse mesurée')
ax.set_ylabel('Fréquence')
ax.set_title('Histogramme des vitesses mesurées')
st.pyplot(fig)

# Histogramme nombre des excès de vitesse par plage
st.subheader('Histogramme des différents excès de vitesse par plage de vitesse')
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(plages, [len(mesure) for mesure in mesures], color='skyblue')
ax.set_xlabel('Plage de différence de vitesse')
ax.set_ylabel('Fréquence')
ax.set_title('Histogramme des excès de vitesse')
st.pyplot(fig)

# Histogramme longitude et latitude
st.subheader('Histogramme des excès de vitesse par position')
st.text("Entrez la première latitude et longitude :")
optionslatitude = ["44.00000", "45.00000", "46.00000", "47.00000", "48.00000", "49.00000", "50.00000", "51.00000", "52.00000"]
selected_option_latitude = st.selectbox("Sélectionnez une latitude", optionslatitude)
optionslongitude = ["-0.00000", "-1.00000", "-2.00000", "-3.00000", "-4.00000", "-5.00000", "-6.00000", "-7.00000", "-8.00000", "-9.00000"]
selected_option_longitude = st.selectbox("Sélectionnez une longitude", optionslongitude)
latitudeLongitude1 =  selected_option_latitude + selected_option_longitude

st.text("Entrez la deuxième latitude et longitude :")
options2latitude = ["44.00000", "45.00000", "46.00000", "47.00000", "48.00000", "49.00000", "50.00000", "51.00000", "52.00000"]
selected_option_latitude2 = st.selectbox("Sélectionnez une deuxieme latitude", options2latitude)
options2longitude = ["-0.00000", "-1.00000", "-2.00000", "-3.00000", "-4.00000", "-5.00000", "-6.00000", "-7.00000", "-8.00000", "-9.00000"]
selected_option_longitude2 = st.selectbox("Sélectionnez une deuxieme longitude", options2longitude)
latitudelongitude2 = selected_option_latitude2 + selected_option_longitude2

if latitudeLongitude1 and latitudelongitude2:

    filtered_data = df_vitesse[(df_vitesse['position'] >= latitudeLongitude1) & (df_vitesse['position'] <= latitudelongitude2)]

    mesures_50_60 = filtered_data[(filtered_data['mesure'] >= 50) & (filtered_data['mesure'] < 60)]
    mesures_60_70 = filtered_data[(filtered_data['mesure'] >= 60) & (filtered_data['mesure'] < 70)]
    mesures_70_80 = filtered_data[(filtered_data['mesure'] >= 70) & (filtered_data['mesure'] < 80)]
    mesures_80_90 = filtered_data[(filtered_data['mesure'] >= 80) & (filtered_data['mesure'] < 90)]
    mesures_90_100 = filtered_data[(filtered_data['mesure'] >= 90) & (filtered_data['mesure'] < 100)]
    mesures_sup_100 = filtered_data[filtered_data['mesure'] >= 100]

    moyenne_50_60 = mesures_50_60['mesure'].mean()
    moyenne_60_70 = mesures_60_70['mesure'].mean()
    moyenne_70_80 = mesures_70_80['mesure'].mean()
    moyenne_80_90 = mesures_80_90['mesure'].mean()
    moyenne_90_100 = mesures_90_100['mesure'].mean()
    moyenne_sup_100 = mesures_sup_100['mesure'].mean()

    st.bar_chart({'50-60': moyenne_50_60, '60-70': moyenne_60_70, '70-80': moyenne_70_80, '80-90': moyenne_80_90, '90-100': moyenne_90_100, 'Sup à 100': moyenne_sup_100})


#Histogramme par sélection de deux dates
date_min = datetime.date(2021, 1, 1)
date_max = datetime.date(2021, 12, 31)
date1 = st.date_input("Sélectionnez la première date :", date_max, date_min)
date2 = st.date_input("Sélectionnez la deuxième date :", date_max, date_min)

if date1 and date2:

    date_string1 = date1.strftime("%Y-%m-%d %H:%M")
    date_string2 = date2.strftime("%Y-%m-%d %H:%M")
    filtered_data = df_vitesse[(df_vitesse['date'] >= date_string1) & (df_vitesse['date'] <= date_string2)]

    mesures_50_60 = filtered_data[(filtered_data['mesure'] >= 50) & (filtered_data['mesure'] < 60)]
    mesures_60_70 = filtered_data[(filtered_data['mesure'] >= 60) & (filtered_data['mesure'] < 70)]
    mesures_70_80 = filtered_data[(filtered_data['mesure'] >= 70) & (filtered_data['mesure'] < 80)]
    mesures_80_90 = filtered_data[(filtered_data['mesure'] >= 80) & (filtered_data['mesure'] < 90)]
    mesures_90_100 = filtered_data[(filtered_data['mesure'] >= 90) & (filtered_data['mesure'] < 100)]
    mesures_sup_100 = filtered_data[filtered_data['mesure'] >= 100]

    moyenne_50_60 = mesures_50_60['mesure'].mean()
    moyenne_60_70 = mesures_60_70['mesure'].mean()
    moyenne_70_80 = mesures_70_80['mesure'].mean()
    moyenne_80_90 = mesures_80_90['mesure'].mean()
    moyenne_90_100 = mesures_90_100['mesure'].mean()
    moyenne_sup_100 = mesures_sup_100['mesure'].mean()

    st.bar_chart({'50-60': moyenne_50_60, '60-70': moyenne_60_70, '70-80': moyenne_70_80, '80-90': moyenne_80_90, '90-100': moyenne_90_100, 'Sup à 100': moyenne_sup_100})
    st.write(f'Moyenne du nombre de mesures entre {date1} et {date2}')


# Statistiques des excès de vitesse
st.subheader('Statistiques des excès de vitesse')
st.write(f"Nombre total d'excès de vitesse : {len(nbrexces)}")
st.write(f"Nombre d'excès de vitesse entre 0 km/h et 10 km/h: {len(mesure_0_10)}")
st.write(f"Nombre d'excès de vitesse entre 10 km/h et 20 km/h : {len(mesure_10_20)}")
st.write(f"Nombre d'excès de vitesse entre 20 km/h et 30 km/h : {len(mesure_20_30)}")
st.write(f"Nombre d'excès de vitesse entre 30 km/h et 40 km/h : {len(mesure_30_40)}")
st.write(f"Nombre d'excès de vitesse supérieurs à 40 km/h : {len(mesure_supp_40)}")
