import pandas as pd
from tqdm import tqdm
import numpy as np
from .constants import stats, types_dict
from .utils import crit_rate, get_p2_team

def _stage_multiplier(n: int) -> float:
    """
    Calcola il moltiplicatore corrispondente a un certo numero di "stadi" di boost o riduzione.

    Ogni statistica può essere aumentata o diminuita da -6 a +6 stadi.
    La formula del moltiplicatore è:
        - se n >= 0 → (2 + n) / 2
        - se n < 0  → 2 / (2 - n)

    """
    try:
        n = int(n)  # converte in int in caso arrivi come float/stringa
    except Exception:
        n = 0  # default di sicurezza

    if n >= 0:
        return (2.0 + n) / 2.0
    else:
        return 2.0 / (2.0 - n)


def _effective_speed(base_spe: float, boosts: dict | None, status: str | None) -> float:
    """
    Calcola la "Speed effettiva" del Pokémon per il turno corrente.

    Passaggi:
      1. Si parte dalla Speed base (base_spe) del Pokémon.
      2. Si applica un eventuale boost di Speed presente nel dizionario 'boosts' (es. dopo Agility).
         - lo stadio è preso da boosts['spe'], e convertito tramite _stage_multiplier()
      3. Se il Pokémon è paralizzato ('par'), si applica il malus Gen1: la Speed diventa il 25% dell'originale.

    Restituisce la Speed effettiva tenendo conto di boost e status.
    """
    eff = float(base_spe or 0.0)  # velocità base
    b = (boosts or {})             # eventuali modifiche di stadio
    stage = b.get('spe', 0) or 0   # estrae lo stadio di Speed (default = 0)

    # applica moltiplicatore degli stadi
    eff *= _stage_multiplier(stage)

    # applica malus se il Pokémon è paralizzato (in Gen 1 diventa 1/4 della speed)
    if (status or 'nostatus') == 'par':
        eff *= 0.25

    return eff


def _base_spe_from_name(name: str | None, p1_team: list, p2_team: list, pokedex: dict) -> float:
    """
    Ricava la Speed base di un Pokémon a partire dal suo nome.

    Logica di ricerca (priorità):
      1. cerca nel team di P1 (se il Pokémon è uno dei suoi)
      2. se non trovato, cerca nel team osservato di P2
      3. come fallback finale, cerca nel Pokédex globale (se fornito)

    Questo serve per poter calcolare la Speed effettiva anche quando il Pokémon
    in campo non è nel team completo disponibile (es. Pokémon visti solo nel log).
    """
    if not name:
        return 0.0

    # cerca nel team P1
    for mon in (p1_team or []):
        if mon and mon.get('name') == name:
            return float(mon.get('base_spe', 0) or 0.0)

    # cerca nel team osservato di P2
    for mon in (p2_team or []):
        if mon and mon.get('name') == name:
            return float(mon.get('base_spe', 0) or 0.0)

    # fallback nel Pokédex completo
    if pokedex and name in pokedex:
        return float(pokedex[name].get('base_spe', 0) or 0.0)

    return 0.0  # se non trovato da nessuna parte

def speed_advantage_features(battle, pokedex):
    """
    Calcola quante volte durante la battaglia P1 ha avuto un vantaggio di Speed
    rispetto al Pokémon avversario in campo.

    Definizione:
        - "vantaggio di Speed" = Speed_effettiva(P1) > Speed_effettiva(P2)
        - tiene conto di boost/drops di Speed e dello status (paralysis)

    Restituisce due feature:
        p1_speed_adv_turns : numero di turni in cui P1 è stato più veloce
        p1_speed_adv_rate  : percentuale sul totale dei turni
    """
    tl = battle.get('battle_timeline', []) or []
    if not tl:
        return {'p1_speed_adv_turns': 0, 'p1_speed_adv_rate': 0.0}

    # informazioni sui team (per recuperare le base stats)
    p1_team = battle.get('p1_team_details', []) or []
    p2_team = get_p2_team(battle, pokedex) or []

    adv = 0     # conteggio di turni in vantaggio di Speed
    total = 0   # numero totale di turni

    # --- Loop su tutti i turni della battaglia ---
    for turn in tl:
        total += 1

        # stati attuali dei Pokémon in campo
        p1s = (turn.get('p1_pokemon_state', {}) or {})
        p2s = (turn.get('p2_pokemon_state', {}) or {})

        # nomi e condizioni dei Pokémon attivi
        p1_name = p1s.get('name')
        p2_name = p2s.get('name')
        p1_status = p1s.get('status') or 'nostatus'
        p2_status = p2s.get('status') or 'nostatus'

        # boost di stat (es. dopo Agility)
        p1_boosts = p1s.get('boosts') or {}
        p2_boosts = p2s.get('boosts') or {}

        # ricava le Speed base (dal team o pokedex)
        p1_base_spe = _base_spe_from_name(p1_name, p1_team, p2_team, pokedex)
        p2_base_spe = _base_spe_from_name(p2_name, p1_team, p2_team, pokedex)

        # calcola Speed effettive dei due Pokémon
        p1_eff = _effective_speed(p1_base_spe, p1_boosts, p1_status)
        p2_eff = _effective_speed(p2_base_spe, p2_boosts, p2_status)

        # se P1 è più veloce → incrementa il contatore
        if p1_eff > p2_eff:
            adv += 1

    # calcola la percentuale di turni con vantaggio di Speed
    rate = adv / total if total > 0 else 0.0

    return {
        'p1_speed_adv_turns': adv,  # conteggio assoluto
        'p1_speed_adv_rate': rate   # valore normalizzato (0–1)
    }


def switch_dynamics_features(battle):
    """
    Conta e classifica i cambi (switch) di Pokémon per Player 1 e Player 2.
    
    Classifica ogni cambio come:
      - voluntary: il giocatore ha scelto volontariamente di cambiare Pokémon
                   (nessuna mossa fatta, pX_move_details = None)
      - forced_faint: il cambio è forzato perché il Pokémon precedente è andato KO ('fnt')
    
    Ritorna un dizionario con i conteggi totali per ciascuna categoria.
    """
    

    # Timeline dei turni della battaglia
    tl = battle.get('battle_timeline', []) or []

    # Se la battaglia ha meno di 2 turni, non può esserci stato alcun cambio
    if len(tl) < 2:
        return {
            'p1_switch_count': 0, 'p2_switch_count': 0,
            'switch_diff': 0,
            'p1_voluntary_switches': 0, 'p2_voluntary_switches': 0,
            'p1_forced_faint_switches': 0, 'p2_forced_faint_switches': 0,
            'p1_switch_rate': 0.0, 'p2_switch_rate': 0.0,
        }

    # --- Inizializza stato del turno precedente ---
    # Nome e stato (status) dei Pokémon iniziali
    p1_prev_name = tl[0].get('p1_pokemon_state', {}).get('name')
    p2_prev_name = tl[0].get('p2_pokemon_state', {}).get('name')
    p1_prev_status = (tl[0].get('p1_pokemon_state', {}) or {}).get('status') or 'nostatus'
    p2_prev_status = (tl[0].get('p2_pokemon_state', {}) or {}).get('status') or 'nostatus'

    # --- Contatori per ogni tipo di cambio ---
    p1_sw = p2_sw = 0   # cambi totali
    p1_vol = p2_vol = 0 # cambi volontari
    p1_fnt = p2_fnt = 0 # cambi forzati da KO
    total_turns = len(tl)

    # --- Loop sui turni successivi ---
    for t in range(1, total_turns):
        # Stato corrente del turno t
        turn = tl[t]
        p1s = turn.get('p1_pokemon_state', {}) or {}
        p2s = turn.get('p2_pokemon_state', {}) or {}

        # Pokémon attuali e stati
        p1_name = p1s.get('name')
        p2_name = p2s.get('name')
        p1_status = p1s.get('status') or 'nostatus'
        p2_status = p2s.get('status') or 'nostatus'

        # Mossa usata in questo turno (può essere None se ha solo cambiato)
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')

        # Verifica se è avvenuto un cambio (nome Pokémon diverso da quello precedente)
        p1_switch = (p1_prev_name is not None and p1_name and p1_name != p1_prev_name)
        p2_switch = (p2_prev_name is not None and p2_name and p2_name != p2_prev_name)

        # --- Classificazione cambio P1 ---
        if p1_switch:
            p1_sw += 1  # incrementa i cambi totali
            if p1_prev_status == 'fnt':
                # il Pokémon precedente è morto → cambio forzato
                p1_fnt += 1
            elif p1_move is None:
                # non ha usato mosse → cambio volontario
                p1_vol += 1
            

        # --- Classificazione cambio P2 ---
        if p2_switch:
            p2_sw += 1
            if p2_prev_status == 'fnt':
                p2_fnt += 1
            elif p2_move is None:
                p2_vol += 1
            

        # Aggiorna "stato precedente" per il prossimo turno
        p1_prev_name = p1_name or p1_prev_name
        p2_prev_name = p2_name or p2_prev_name
        p1_prev_status = p1_status
        p2_prev_status = p2_status

    # --- Output finale ---
    # Restituisce tutti i conteggi raccolti
    return {
        'p1_switch_count': p1_sw,
        'p2_switch_count': p2_sw,
        'p1_voluntary_switches': p1_vol,
        'p2_voluntary_switches': p2_vol,
        'p1_forced_faint_switches': p1_fnt,
        'p2_forced_faint_switches': p2_fnt,
    }

def status_timing_features(battle):
    """
    Estrae feature 'mirate' sugli status cruciali in Gen1, in modo simmetrico per P1/P2
    e con differenze già pronte per la Logistic.

    Feature calcolate:
      - p1_freeze_turns, p2_freeze_turns:
          Numero di turni in cui il lato è in stato 'frz' (congelato).
          Il freeze in RBY è estremamente impattante: più turni -> peggiore per quel lato.

      - freeze_turns_diff:
          p2_frz - p1_frz (positivo se P2 è cong. più a lungo → vantaggio P1)

      - first_par_inv_diff:
          (1 / primo_turno_par_P1) - (1 / primo_turno_par_P2)
          Valori maggiori indicano che P1 ha paralizzato prima (early paralysis),
          che in RBY spesso determina il controllo della Speed e del momentum.

    """
    # Timeline dei turni (lista di dict); se vuota, ritorna zeri coerenti col tipo.
    tl = battle.get('battle_timeline', []) or []
    if not tl:
        return {
            'p1_freeze_turns': 0,
            'p2_freeze_turns': 0,
            'p1_first_par_inv': 0.0,   # lasciata per compatibilità, ma non usata nel return finale
            'p2_first_par_inv': 0.0,   # idem
            'freeze_turns_diff': 0,
            'first_par_inv_diff': 0.0,
        }

    def scan_side(side):
        """
        Scansiona l'intera timeline per un lato ('p1' o 'p2') e ritorna:
          - freeze_turns: conteggio dei turni con status 'frz'
          - first_par_inv: inversa del primo turno in cui appare 'par'
                * se la paralisi compare al turno 3 -> 1/3 ≈ 0.333
                * se non compare mai -> 0.0
        """
        freeze_turns = 0
        first_par_turn = None  # memorizzeremo il primo turno in cui vediamo 'par'

        for idx, turn in enumerate(tl):
            # Alcuni log potrebbero non avere 'turn' esplicito: usiamo indice+1 come fallback.
            t = turn.get('turn') or (idx + 1)

            # Leggi lo stato del lato per questo turno; default 'nostatus' se mancante.
            status = (turn.get(f'{side}_pokemon_state', {}) or {}).get('status') or 'nostatus'

            # Conta i turni in cui il lato è congelato
            if status == 'frz':
                freeze_turns += 1

            # Se è la prima volta che vediamo 'par', registriamo il turno (una sola volta)
            if status == 'par' and first_par_turn is None:
                first_par_turn = int(t)

        # Converte il primo turno in inversa: 1/t (0 se mai paralizzato)
        first_par_inv = (1.0 / first_par_turn) if (first_par_turn and first_par_turn > 0) else 0.0
        return freeze_turns, first_par_inv

    # Scansiona entrambi i lati
    p1_frz, p1_par_inv = scan_side('p1')
    p2_frz, p2_par_inv = scan_side('p2')

    # Costruisci il dizionario di output
    return {
        'p1_freeze_turns': p1_frz,                # turni di freeze su P1
        'p2_freeze_turns': p2_frz,                # turni di freeze su P2

        # Le inverse individuali erano peggiorative nel tuo testing:
        # 'p1_first_par_inv': p1_par_inv,
        # 'p2_first_par_inv': p2_par_inv,

        'freeze_turns_diff': p2_frz - p1_frz,     # + se P2 congelato più di P1 (vantaggio P1)
        'first_par_inv_diff': p1_par_inv - p2_par_inv,  # + se P1 paralizza prima (vantaggio P1)
    }


def offensive_momentum_features(battle):
    """
    Calcola la "spinta offensiva" (offensive momentum) di ciascun giocatore
    durante la battaglia.

    L’idea è misurare quante volte P1 e P2 usano mosse offensive
    (quelle con base_power > 0) rispetto al numero totale di turni.

    Output:
        p1_offensive_rate : frequenza di mosse offensive di P1
        p2_offensive_rate : frequenza di mosse offensive di P2
        offensive_diff    : differenza (P1 - P2), positiva → P1 più aggressivo
    """

    # Dizionario dove salveremo le feature calcolate
    features = {}

    # Recupera la sequenza temporale della battaglia
    # (lista di "turni", ciascuno con info su mosse, stati, HP, ecc.)
    battle_timeline = battle.get('battle_timeline', [])

    # Se la battaglia non ha turni, non si possono calcolare statistiche
    if not battle_timeline:
        return {'offensive_diff': 0.0}

    # --- Inizializzazione contatori ---
    p1_off_turns = 0  # Numero di turni in cui P1 ha attaccato
    p2_off_turns = 0  # Numero di turni in cui P2 ha attaccato
    total_turns = 0   # Numero totale di turni nella battaglia

    # --- Analisi turno per turno ---
    for turn in battle_timeline:
        total_turns += 1  # Conta un turno in più

        # Dettagli delle mosse del turno (possono essere None se ha solo cambiato Pokémon)
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')

        # Una mossa è "offensiva" se:
        # - esiste (non è None)
        # - ha un campo "base_power" positivo (cioè infligge danni)
        if p1_move and p1_move.get('base_power', 0) > 0:
            p1_off_turns += 1
        if p2_move and p2_move.get('base_power', 0) > 0:
            p2_off_turns += 1

    # --- Calcolo delle frequenze (percentuali di turni offensivi) ---
    # Divide il numero di mosse offensive per il numero totale di turni
    p1_rate = p1_off_turns / total_turns
    p2_rate = p2_off_turns / total_turns

    # --- Costruisce le feature da restituire ---
    features['p1_offensive_rate'] = p1_rate  # aggressività P1
    features['p2_offensive_rate'] = p2_rate  # aggressività P2
    features['offensive_diff'] = p1_rate - p2_rate  # differenza netta

    return features



def static_features(battle: dict, pokedex) -> dict: 

    features = {}

    # --- Player 1 Team Features ---
    p1_team = battle.get('p1_team_details', [])
    if p1_team:
        features['p1_avg_crit_rate'] = np.mean([crit_rate(p.get('base_spe', 0)) for p in p1_team])
        
        # Average stats for p1 team
        for stat in stats:
            features[f'p1_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p1_team])

        
    # --- Player 2 Observed Team Features ---
    p2_lead = battle.get('p2_lead_details')
    p2_team = get_p2_team(battle, pokedex)
    
    if p2_team:
        features['p2_avg_crit_rate'] = np.mean([crit_rate(p.get('base_spe', 0)) for p in p2_team])
        
        # Average stats for observed p2_team
        for stat in stats:
            features[f'p2_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p2_team])

        # Team coverage
        features["p2_team_coverage"] = min(len(p2_team) / 6.0, 1.0)


    # --- Average stats differences
    #for stat in stats:
    #    p1_mean = features.get(f'p1_mean_{stat}', 0)
    #    p2_mean = features.get(f'p2_mean_{stat}', 0)
    #    features[f'mean_{stat}_diff'] = p1_mean - p2_mean


    # --- First turn matchup ---
    battle_timeline = battle.get('battle_timeline', [])
    if p1_team and p2_lead and battle_timeline:
        first_turn = battle_timeline[0]
        p1_pokemon_name = first_turn.get('p1_pokemon_state', {}).get('name')

        # Find matching Pokemon in p1_team
        p1_lead = next((p for p in p1_team if p.get('name') == p1_pokemon_name), None)

        if p1_lead: 
            p1_spe = p1_lead.get('base_spe', 0)
            p2_spe = p2_lead.get('base_spe', 0)
            features['spe_diff'] = p1_spe - p2_spe
        else: 
            features['spe_diff'] = 0.0

    return features

# status info with weights 
def extract_status_features(battle):
    status_weights = {"slp": 3, "frz": 4, "par": 2,"tox": 1.5, "psn": 1,"brn": 0.5}

    features = {}

    battle_timeline = battle.get('battle_timeline', [])
    p1_score = 0.0
    p2_score = 0.0

    for turn in battle_timeline:
        p1_status = turn.get('p1_pokemon_state', {}).get('status')
        p2_status = turn.get('p2_pokemon_state', {}).get('status')

        if p1_status in status_weights:
            p1_score += status_weights[p1_status]
        if p2_status in status_weights:
            p2_score += status_weights[p2_status]

    features['status_diff'] = p1_score - p2_score
    return features

def first_move_rate(battle, pokedex):
    pass
    

# TODO - Risistemare
def dynamic_features(battle: dict) -> dict:

    features = {
        'p1_bad_status': 0, 'p2_bad_status': 0,
        'p1_ko_count': 0, 'p2_ko_count': 0
    }

    p1_hp_loss = 0.0
    p2_hp_loss = 0.0
    prev_p1_hp = None
    prev_p2_hp = None
    
    battle_timeline = battle.get('battle_timeline', [])

    for turn in battle_timeline:
        p1_pokemon_state = turn.get('p1_pokemon_state', {})
        p2_pokemon_state = turn.get('p2_pokemon_state', {})
        
        p1_status = p1_pokemon_state.get('status', {})
        p2_status = p2_pokemon_state.get('status', {})

        p1_hp = p1_pokemon_state.get('hp_pct', 1.0)
        p2_hp = p2_pokemon_state.get('hp_pct', 1.0)

        # HP loss 
        if prev_p1_hp is not None:
            d = p1_hp - prev_p1_hp
            if d < 0:
                p1_hp_loss += -d
        if prev_p2_hp is not None:
            d = p2_hp - prev_p2_hp
            if d < 0:
                p2_hp_loss += -d

        prev_p1_hp = p1_hp
        prev_p2_hp = p2_hp

        features['p1_hp_loss'] = round(p1_hp_loss * 100, 2) 
        features['p2_hp_loss'] = round(p2_hp_loss * 100, 2)

        # Number of turns with altered status
        if p1_status not in ['nostatus', 'fnt']:
            features['p1_bad_status'] += 1

        if p2_status not in ['nostatus', 'fnt']:
            features['p2_bad_status'] += 1

        # Number of fainted pokemons
        if p1_status == 'fnt': 
            features['p1_ko_count'] += 1
            
        if p2_status == 'fnt': 
            features['p2_ko_count'] += 1

    return features
    

def create_features(data: list[dict], pokedex) -> pd.DataFrame:
    """
    A very basic feature extraction function.
    It only uses the aggregated base stats of the player's team and opponent's lead.
    """
    feature_list = []
    for battle in tqdm(data, desc="Extracting features"):
        #if battle.get('battle_id') == 4877: continue
        
        features = {}

        features.update(extract_status_features(battle))
        features.update(static_features(battle, pokedex))
        features.update(dynamic_features(battle))
        features.update(offensive_momentum_features(battle)) # <-- NEW
        features.update(switch_dynamics_features(battle)) # <-- NEW
        features.update(status_timing_features(battle))   # <= NEW, simmetrica e continua
        features.update(speed_advantage_features(battle, pokedex))  # <-- NEW


        
        # We also need the ID and the target variable (if it exists)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)