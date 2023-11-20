import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import zipfile
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import altair as alt
from pdb import set_trace
import random
from scipy.stats import beta as beta_dist

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

st.title('Machine Learning')
# Sélection d'un nombre entre 0 et 500 000 pour éviter d'être au dessus des 200mb
newData = df_vitesse.iloc[0:500000]
# Afficher le graphiques avec Altair
st.subheader('Afficher le graphique avec Altair')
chart = alt.Chart(newData).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='variable:N'
).properties(
    width=150,
    height=150
).repeat(
    row=['mesure'],
    column=['limite']
).interactive()
st.altair_chart(chart, use_container_width=True)

st.subheader('Afficher la linear regression et le clustering')

position = newData['position'].str.strip()
newData['position'].replace('null', pd.NA, inplace=True)
newData[['latitude', 'longitude']] = position.str.split(expand=True)
newData['latitude'] = pd.to_numeric(newData['latitude'], errors='coerce')
newData['longitude'] = pd.to_numeric(newData['longitude'], errors='coerce')
newData = newData.dropna()

X = newData[['latitude', 'longitude']]
y = newData['limite']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
linear_reg_score = linear_reg_model.score(X_test, y_test)
st.write(f"Score de la régression linéaire : {linear_reg_score}")
st.write("Coefficient (pente) :", linear_reg_model.coef_)
st.write("Intercept :", linear_reg_model.intercept_)


kmeans_model = KMeans(n_clusters=1) # 1 clusters
newData['cluster'] = kmeans_model.fit_predict(X)

chart = alt.Chart(newData).mark_circle().encode(
    x='latitude',
    y='longitude',
    color='cluster:N'
).properties(
    width=600,
    height=400
)
st.altair_chart(chart, use_container_width=True)


#Renforcement learning

st.title('Reinforcement Learning - Thompson Sampling')

class ThompsonSamplingAgent:
    def __init__(self, actions):
        self.actions = actions
        self.alpha_beta = {action: (1, 1) for action in actions}

    def get_action(self):
        samples = {action: beta_dist.rvs(alpha, beta, size=1)[0] for action, (alpha, beta) in self.alpha_beta.items()}
        return max(samples, key=samples.get)

    def update_parameters(self, action, reward):
        alpha, beta = self.alpha_beta[action]
        self.alpha_beta[action] = (alpha + reward, beta + (1 - reward))

actions = [0, 1]
agent_ts = ThompsonSamplingAgent(actions)

success_rate_history = {action: [] for action in actions}

for epoch in range(50):
    chosen_action_ts = agent_ts.get_action()
    reward_ts = len(mesures[actions.index(chosen_action_ts)]) > 0
    agent_ts.update_parameters(chosen_action_ts, reward_ts)
    for action, history in success_rate_history.items():
        history.append(agent_ts.alpha_beta[action][0] / sum(agent_ts.alpha_beta[action]))

epoch_values = list(range(1, 51))
success_rate_df = pd.DataFrame({action: history for action, history in success_rate_history.items()})
success_rate_df['Epoch'] = epoch_values

chart_ts = alt.Chart(success_rate_df.melt('Epoch')).mark_line().encode(
    x='Epoch:O',
    y='value:Q',
    color='variable:N'
).properties(
    width=600,
    height=400
)

st.altair_chart(chart_ts, use_container_width=True)
