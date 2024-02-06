import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carico il dataset
df = pd.read_csv("dataset_climatico.csv")

# Pulizia dei dati
df.dropna(inplace=True)  # Rimuove le righe con valori mancanti

# Converti la colonna 'data_osservazione' in tipo datetime
df['data_osservazione'] = pd.to_datetime(df['data_osservazione'])

# Imposta la data come indice
df.set_index('data_osservazione', inplace=True)

# Normalizzazione Z-score
cols_da_normalizzare = ['temperatura_media', 'precipitazioni', 'umidita', 'velocita_vento']
df[cols_da_normalizzare] = (df[cols_da_normalizzare] - df[cols_da_normalizzare].mean()) / df[cols_da_normalizzare].std()

# Analisi Esplorativa dei Dati
desc_stats = df.describe().round(2)
print(desc_stats)

# Calcoli di correlazione
correlazione_temperatura_umidita = df['temperatura_media'].corr(df['umidita'])
correlazione_temperatura_precipitazioni = df['temperatura_media'].corr(df['precipitazioni'])
correlazione_temperatura_velocita_vento = df['temperatura_media'].corr(df['velocita_vento'])
correlazione_precipitazioni_umidita = df['precipitazioni'].corr(df['umidita'])
correlazione_velocita_vento_umidita = df['velocita_vento'].corr(df['umidita'])
correlazione_velocita_vento_precipitazioni = df['velocita_vento'].corr(df['precipitazioni'])

# Stampa dei risultati
print(f'Correlazione tra Temperatura e Umidità: {correlazione_temperatura_umidita:.2f}')
print(f'Correlazione tra Temperatura e Precipitazioni: {correlazione_temperatura_precipitazioni:.2f}')
print(f'Correlazione tra Temperatura e Velocità del Vento: {correlazione_temperatura_velocita_vento:.2f}')
print(f'Correlazione tra Precipitazioni e Umidità: {correlazione_precipitazioni_umidita:.2f}')
print(f'Correlazione tra Velocità del Vento e Umidità: {correlazione_velocita_vento_umidita:.2f}')
print(f'Correlazione tra Velocità del Vento e Precipitazioni: {correlazione_velocita_vento_precipitazioni:.2f}')


# Creazione di grafici
plt.figure(figsize=(12, 8))

# Istogrammi
for col in cols_da_normalizzare:
    plt.subplot(2, 2, cols_da_normalizzare.index(col) + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Istogramma di {col}')

plt.tight_layout()
plt.show()

# Box plots
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[cols_da_normalizzare])
plt.title('Box plots delle variabili normalizzate')
plt.show()

# Analisi di Correlazione
correlation_matrix = df[cols_da_normalizzare].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Matrice di correlazione tra variabili meteorologiche')
plt.show()


# Analisi di Correlazione per ogni stazione meteorologica
stazioni_meteorologiche = df['stazione_meteorologica'].unique()

for stazione in stazioni_meteorologiche:
    df_stazione = df[df['stazione_meteorologica'] == stazione]

    # Calcola la matrice di correlazione
    correlation_matrix = df_stazione[cols_da_normalizzare].corr()

    # Visualizza la heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f'Matrice di correlazione - {stazione}')
    plt.show()


# Traccia l'andamento della temperatura media nel tempo per ogni stazione meteorologica
for stazione in df['stazione_meteorologica'].unique():
    df_stazione = df[df['stazione_meteorologica'] == stazione]

    # Reimposta l'indice
    df_stazione.reset_index(inplace=True)

    plt.plot(df_stazione['data_osservazione'], df_stazione['temperatura_media'], label=stazione)

plt.title('Andamento della Temperatura Media nel Tempo')
plt.xlabel('Data')
plt.ylabel('Temperatura Media')
plt.legend()
plt.show()

#Andamento del trend temporale
plt.plot(df.index, df['temperatura_media'])
plt.title('Andamento della Temperatura Media nel Tempo')
plt.xlabel('Data')
plt.ylabel('Temperatura Media')
plt.show()

#Analisi stagionale
df['Mese'] = df.index.month
sns.boxplot(x='Mese', y='temperatura_media', data=df)
plt.title('Distribuzione della Temperatura Media per Mese')
plt.xlabel('Mese')
plt.ylabel('Temperatura Media')
plt.show()

#Calcolo la media mobile per avere una visione più chiara del trend nel tempo.
df['Media Mobile'] = df['temperatura_media'].rolling(window=30).mean()
plt.plot(df.index, df['temperatura_media'], label='Temperatura Media')
plt.plot(df.index, df['Media Mobile'], label='Media Mobile (30 giorni)')
plt.title('Andamento della Temperatura Media nel Tempo con Media Mobile')
plt.xlabel('Data')
plt.ylabel('Temperatura Media')
plt.legend()
plt.show()

#Frequenza delle osservazioni nel tempo
df['Anno'] = df.index.year
df.groupby('Anno').size().plot(kind='bar')
plt.title('Frequenza delle Osservazioni per Anno')
plt.xlabel('Anno')
plt.ylabel('Numero di Osservazioni')
plt.show()

# Trend Temporali delle Precipitazioni
plt.plot(df.index, df['precipitazioni'])
plt.title('Andamento delle Precipitazioni nel Tempo')
plt.xlabel('Data')
plt.ylabel('Precipitazioni')
plt.show()

# Analisi Mensile delle Precipitazioni
df['Mese'] = df.index.month
sns.boxplot(x='Mese', y='precipitazioni', data=df)
plt.title('Distribuzione delle Precipitazioni per Mese')
plt.xlabel('Mese')
plt.ylabel('Precipitazioni')
plt.show()

# Trend Temporali dell'Umidità
plt.plot(df.index, df['umidita'])
plt.title('Andamento dell\'Umidità nel Tempo')
plt.xlabel('Data')
plt.ylabel('Umidità')
plt.show()

# Analisi Mensile dell'Umidità
sns.boxplot(x='Mese', y='umidita', data=df)
plt.title('Distribuzione dell\'Umidità per Mese')
plt.xlabel('Mese')
plt.ylabel('Umidità')
plt.show()

# Trend Temporali della Velocità del Vento
plt.plot(df.index, df['velocita_vento'])
plt.title('Andamento della Velocità del Vento nel Tempo')
plt.xlabel('Data')
plt.ylabel('Velocità del Vento')
plt.show()

# Analisi Mensile della Velocità del Vento
sns.boxplot(x='Mese', y='velocita_vento', data=df)
plt.title('Distribuzione della Velocità del Vento per Mese')
plt.xlabel('Mese')
plt.ylabel('Velocità del Vento')
plt.show()
