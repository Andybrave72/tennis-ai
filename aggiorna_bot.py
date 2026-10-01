import pandas as pd
import numpy as np
import os
import pickle
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURAZIONE TELEGRAM ---
TELEGRAM_TOKEN = '8936192832:AAHR_SBXs77iDFmudt-cO0qeLrJ9FEiReSE'
CHAT_ID = '6959156465'

def invia_telegram(testo):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': testo,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Report inviato con successo su Telegram!")
        else:
            print(f"Errore Telegram: {response.text}")
    except Exception as e:
        print(f"Errore di connessione a Telegram: {e}")

print("--- AVVIO AGGIORNAMENTO AUTOMATICO BOT TENNIS ---")

# 1. Caricamento e unione dei dati dalla cartella dati_storici
path_cartella = 'dati_storici'
anni = range(2020, 2027)
chunks = []

for anno in anni:
    file_path = os.path.join(path_cartella, f"{anno}.csv")
    if os.path.exists(file_path):
        df_anno = pd.read_csv(file_path, low_memory=False)
        chunks.append(df_anno)

df = pd.concat(chunks, ignore_index=True)
df = df[df['tourney_level'] == 'M'].copy()
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['tourney_date', 'winner_name', 'loser_name']).sort_values('tourney_date')

# Dataset simmetrico per l'addestramento
np.random.seed(42)
scambia = np.random.rand(len(df)) < 0.5

df_model = pd.DataFrame()
df_model['tourney_date'] = df['tourney_date']
df_model['surface'] = df['surface']
df_model['giocatore_1'] = np.where(scambia, df['loser_name'], df['winner_name'])
df_model['giocatore_2'] = np.where(scambia, df['winner_name'], df['loser_name'])
df_model['rank_1'] = np.where(scambia, df['loser_rank'], df['winner_rank'])
df_model['rank_2'] = np.where(scambia, df['winner_rank'], df['loser_rank'])

hand_w = df['winner_hand'].fillna('R')
hand_l = df['loser_hand'].fillna('R')
df_model['g1_hand'] = np.where(scambia, hand_l, hand_w)
df_model['g2_hand'] = np.where(scambia, hand_w, hand_l)

df_model['target_vittoria_g1'] = np.where(scambia, 0, 1)
df_model['rank_1'] = df_model['rank_1'].fillna(9999)
df_model['rank_2'] = df_model['rank_2'].fillna(9999)
df_model['diff_rank'] = df_model['rank_2'] - df_model['rank_1']

# 2. Simulazione cronologica per ricostruire Elo, forme e stanchezza aggiornate a oggi
def calcola_elo(r1, r2, win1, k=32):
    exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    exp2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
    return r1 + k * (win1 - exp1), r2 + k * ((1 - win1) - exp2)

elo_superficie = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
storico_vittorie_generale = {}
storico_vittorie_superficie = {'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {}}
scontri_diretti = {}
ultima_data_giocatore = {}
player_hands = {}

g1_elo_surf, g2_elo_surf = [], []
g1_forma_gen, g2_forma_gen = [], []
g1_forma_surf, g2_forma_surf = [], []
h2h_vantaggio = []
g1_rest_days, g2_rest_days = [], []
g1_is_lefty, g2_is_lefty = [], []

for index, row in df_model.iterrows():
    g1 = row['giocatore_1']
    g2 = row['giocatore_2']
    surf = row['surface']
    if surf not in elo_superficie: surf = 'Hard'
    current_date = row['tourney_date']
    
    player_hands[g1] = row['g1_hand']
    player_hands[g2] = row['g2_hand']
    g1_is_lefty.append(1 if row['g1_hand'] == 'L' else 0)
    g2_is_lefty.append(1 if row['g2_hand'] == 'L' else 0)
    
    e1 = elo_superficie[surf].get(g1, 1500.0)
    e2 = elo_superficie[surf].get(g2, 1500.0)
    g1_elo_surf.append(e1)
    g2_elo_surf.append(e2)
    
    g1_forma_gen.append(np.mean(storico_vittorie_generale[g1][-10:]) if g1 in storico_vittorie_generale and len(storico_vittorie_generale[g1]) > 0 else 0.5)
    g2_forma_gen.append(np.mean(storico_vittorie_generale[g2][-10:]) if g2 in storico_vittorie_generale and len(storico_vittorie_generale[g2]) > 0 else 0.5)
    
    dict_surf_g1 = storico_vittorie_superficie[surf]
    g1_forma_surf.append(np.mean(dict_surf_g1[g1][-5:]) if g1 in dict_surf_g1 and len(dict_surf_g1[g1]) > 0 else 0.5)
    dict_surf_g2 = storico_vittorie_superficie[surf]
    g2_forma_surf.append(np.mean(dict_surf_g2[g2][-5:]) if g2 in dict_surf_g2 and len(dict_surf_g2[g2]) > 0 else 0.5)
    
    coppia = tuple(sorted([g1, g2]))
    if coppia in scontri_diretti:
        vittorie_g1_h2h = scontri_diretti[coppia]['vittorie'][g1]
        totale_h2h = scontri_diretti[coppia]['totale']
        h2h_vantaggio.append(vittorie_g1_h2h / totale_h2h if totale_h2h > 0 else 0.5)
    else:
        h2h_vantaggio.append(0.5)
        
    rest_g1 = (current_date - ultima_data_giocatore[g1]).days if g1 in ultima_data_giocatore else 14
    rest_g2 = (current_date - ultima_data_giocatore[g2]).days if g2 in ultima_data_giocatore else 14
    g1_rest_days.append(min(rest_g1, 30))
    g2_rest_days.append(min(rest_g2, 30))
    
    win_val = row['target_vittoria_g1']
    
    ne1, ne2 = calcola_elo(e1, e2, win_val)
    elo_superficie[surf][g1] = ne1
    elo_superficie[surf][g2] = ne2
    
    if g1 not in storico_vittorie_generale: storico_vittorie_generale[g1] = []
    if g2 not in storico_vittorie_generale: storico_vittorie_generale[g2] = []
    storico_vittorie_generale[g1].append(win_val)
    storico_vittorie_generale[g2].append(1 - win_val)
    
    if g1 not in storico_vittorie_superficie[surf]: storico_vittorie_superficie[surf][g1] = []
    if g2 not in storico_vittorie_superficie[surf]: storico_vittorie_superficie[surf][g2] = []
    storico_vittorie_superficie[surf][g1].append(win_val)
    storico_vittorie_superficie[surf][g2].append(1 - win_val)
    
    if coppia not in scontri_diretti:
        scontri_diretti[coppia] = {'totale': 0, 'vittorie': {g1: 0, g2: 0}}
    scontri_diretti[coppia]['totale'] += 1
    vincitore_reale = g1 if win_val == 1 else g2
    scontri_diretti[coppia]['vittorie'][vincitore_reale] += 1
    
    ultima_data_giocatore[g1] = current_date
    ultima_data_giocatore[g2] = current_date

# Salvataggio dello stato aggiornato in pickle
state_to_save = {
    'elo': elo_superficie,
    'storico_gen': storico_vittorie_generale,
    'storico_surf': storico_vittorie_superficie,
    'h2h': scontri_diretti,
    'last_date': ultima_data_giocatore,
    'hands': player_hands
}
with open('bot_state.pkl', 'wb') as f:
    pickle.dump(state_to_save, f)

# 3. Addestramento del modello definitivo sulle stagioni passate
df_model['diff_elo_surf'] = np.array(g1_elo_surf) - np.array(g2_elo_surf)
df_model['diff_forma_gen'] = np.array(g1_forma_gen) - np.array(g2_forma_gen)
df_model['diff_forma_surf'] = np.array(g1_forma_surf) - np.array(g2_forma_surf)
df_model['h2h_score'] = h2h_vantaggio
df_model['diff_rest'] = np.array(g1_rest_days) - np.array(g2_rest_days)
df_model['g1_lefty'] = g1_is_lefty
df_model['g2_lefty'] = g2_is_lefty

df_train = df_model[df_model['tourney_date'] < '2024-01-01']
features = ['diff_rank', 'diff_elo_surf', 'diff_forma_gen', 'diff_forma_surf', 'h2h_score', 'diff_rest', 'g1_lefty', 'g2_lefty']

X_train = df_train[features]
y_train = df_train['target_vittoria_g1']

modello_finale = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_split=10, random_state=42)
modello_finale.fit(X_train, y_train)

# 4. Generazione del report e invio su Telegram
ultime_partite = df_model[df_model['tourney_date'] >= df_model['tourney_date'].max() - pd.Timedelta(days=3)].copy()

if len(ultime_partite) > 0:
    X_ultime = ultime_partite[features]
    probabilita = modello_finale.predict_proba(X_ultime)[:, 1]
    
    report_lines = [f"🎾 <b>REPORT PRONOSTICI TENNIS</b>\nAggiornato al: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    
    for i, row in ultime_partite.reset_index(drop=True).iterrows():
        g1 = row['giocatore_1']
        g2 = row['giocatore_2']
        p_g1 = probabilita[i] * 100
        p_g2 = (1 - probabilita[i]) * 100
        favorito = g1 if p_g1 > p_g2 else g2
        perc_fav = max(p_g1, p_g2)
        
        linea = f"• <b>{g1}</b> vs <b>{g2}</b>\n👉 <i>Favorito:</i> <b>{favorito}</b> ({perc_fav:.1f}%)\n"
        report_lines.append(linea)
    
    messaggio_finale = "\n".join(report_lines)
    invia_telegram(messaggio_finale)
else:
    invia_telegram("🎾 <b>BOT TENNIS:</b> Nessuna partita recente trovata per il report odierno.")