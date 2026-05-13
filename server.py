from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, json, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TODAY = datetime.now()

# ═══════════════════════════════════════════════════════════
# 1. BASE FACTUELLE : dates irréfutables (ne pas toucher)
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# 1. BASE FACTUELLE : dates au format MM-JJ (mois-jour)
# ═══════════════════════════════════════════════════════════
FETES_FIXES = {
    "Fête de l'Indépendance": "03-20",   # 20 mars
    "Fête du Travail": "05-01",          # 1er mai
    "Fête de la République": "07-25",    # 25 juillet
    "Fête de la Femme": "08-13",         # 13 août
    "Fête de l'Arbre": "11-15",          # 15 novembre
    "Révolution": "01-14",               # 14 janvier
    "Fête du Jeune": "11-16",            # 16 novembre
    "Fête de l'Evacuation": "10-15",     # 15 octobre
    "Fête Nationale": "11-07",           # 7 novembre
}

FETES_LUNAIRES_2026 = {
    "Ramadan": {"debut": "2026-02-17", "fin": "2026-03-18", "type": "religion"},
    "Aïd El Fitr": {"date": "2026-03-18", "type": "religion"},
    "Aïd El Adha": {"date": "2026-05-25", "type": "religion", "duree": 4},
    "Mouled": {"date": "2026-08-26", "type": "religion"},
}

ARTICLE_CONTEXT = {
    "BOGA CIDRE VER.RET 100CL": {"famille": "cidre", "boost": 15},
    "COCA-COLA PET 100CL": {"famille": "cola", "boost": 10},
    "FANTA PET 100CL": {"famille": "fanta", "boost": 10},
    "JUS": {"famille": "jus", "boost": 5},
    "SCHWEPPES": {"famille": "tonic", "boost": 8},
}

# ═══════════════════════════════════════════════════════════
# 2. PROMPT LLM : plus de dates factuelles écrites dedans
# ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Tu es un expert commercial et industriel senior pour la SFBT (Societe des Boissons de Tunisie).

MISSION : Analyser les evenements qui influencent les VENTES et la PRODUCTION d'un article SFBT sur une periode donnee en Tunisie.

=== CATEGORIES D'EVENEMENTS ===

FETES ET JOURS FERIES : Fete du Travail, Fete de la Republique, Fete de la Femme, Fete de l'Independance, Aïd El Adha, Aïd El Fitr, Ramadan, Mouled...
SPORT : matchs Ligue 1 Tunisienne, selection nationale, finale Coupe de Tunisie, grands matchs TV...
METEO : canicule >35 degres, vague de froid, saison estivale, pluies...
MARKETING : promotions GMS (Carrefour, Monoprix), campagnes TV SFBT, rentree scolaire, vacances estivales...
PRODUCTION : matieres premieres (sucre, aluminium, PET, CO2, eau), logistique, greves transporteurs, arrets techniques...
CONCURRENCE : lancements concurrents, promotions aggressives, nouvelles marques...
ECONOMIE : inflation, devaluation dinar, restrictions importation...

=== CONSIGNES STRICTES ===
- Identifie les evenements pertinents UNIQUEMENT dans la periode demandee.
- Pour les fetes nationales et religieuses, indique le NOM exact dans la description. Tu peux te tromper sur la date : un correcteur la rectifiera automatiquement.
- Pour les evenements meteo, sportifs spontanes ou economiques, raisonne sur la saison et le contexte realiste tunisien.
- Minimum 10 evenements, melanger VENTES et PRODUCTION (au moins 3 production).
- Descriptions precises avec impact chiffre quand possible (+20%, -15%, etc.).
- Tendance = hausse | stable | baisse
- Impact = Eleve | Moyen | Faible
- type = sport | meteo | fete | marketing | religion | production | concurrence | economie

=== FORMAT JSON OBLIGATOIRE ===
Retourne UNIQUEMENT un JSON valide, sans texte avant/apres, sans balises markdown :

{
  "article": "nom exact",
  "jours": 30,
  "date_debut": "JJ/MM/AAAA",
  "date_fin": "JJ/MM/AAAA",
  "resume": "Analyse en 2 phrases",
  "tendance": "hausse",
  "variation_pct": 15,
  "evenements": [
    {
      "date": "JJ/MM/2026",
      "description": "Description precise avec impact chiffre",
      "impact": "Eleve",
      "type": "sport"
    }
  ]
}"""


# ═══════════════════════════════════════════════════════════
# 3. CORRECTEUR FACTUEL (post-traitement)
# ═══════════════════════════════════════════════════════════
def normalize_fete(description: str, annee: int):
    texte = description.lower()
    
    for fete, mmjj in FETES_FIXES.items():
        if fete.lower() in texte or fete.lower().replace("é", "e") in texte:
            # mmjj est MM-JJ, donc f"{annee}-{mmjj}" est parseable par %Y-%m-%d
            return f"{annee}-{mmjj}", "fete"
    ...
    
    # Fetes lunaires
    for fete, data in FETES_LUNAIRES_2026.items():
        if fete.lower() in texte:
            if "date" in data:
                return data["date"], data["type"]
            elif "debut" in data and ("debut" in texte or "ramadan" in texte):
                return data["debut"], data["type"]
            elif "fin" in data and ("fin" in texte or "aid el fitr" in texte):
                return data["fin"], data["type"]
    
    return None, None


def post_process_events(raw_events, today, horizon):
    """Corrige les dates des fetes connues, filtre hors periode, normalise l'impact."""
    annee = today.year
    results = []
    
    for ev in raw_events:
        desc = ev.get("description", "")
        date_llm = ev.get("date", "")
        type_ev = ev.get("type", "autre").lower().strip()
        
        # --- Tentative de correction factuelle ---
        date_correcte, type_detecte = normalize_fete(desc, annee)
        
        if date_correcte:
            # C'est une fete connue : on force la date officielle
            date_finale = date_correcte
            type_ev = type_detecte
        else:
            # Evenement dynamique (meteo, sport, eco...) : on garde la date du LLM
            try:
                dt = datetime.strptime(date_llm, "%d/%m/%Y")
                date_finale = dt.strftime("%Y-%m-%d")
            except:
                # Si le LLM donne une periode vague, on tente d'extraire un mois
                mois_map = {
                    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4,
                    "mai": 5, "juin": 6, "juillet": 7, "aout": 8,
                    "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12
                }
                mois_trouve = None
                for m_fr, m_num in mois_map.items():
                    if m_fr in date_llm.lower():
                        mois_trouve = m_num
                        break
                if mois_trouve:
                    date_finale = f"{annee}-{mois_trouve:02d}-15"
                else:
                    continue  # date indeterminee -> ignore
        
        # --- Verification periode ---
        try:
            dt_evt = datetime.strptime(date_finale, "%Y-%m-%d")
        except:
            continue
        
        if not (today <= dt_evt <= horizon):
            continue  # hors de la fenetre demandee
        
        # --- Normalisation impact ---
        impact = ev.get("impact", "Moyen").strip().upper()
        if "ELEV" in impact or "HAUT" in impact or "FORT" in impact:
            impact_norm = "Élevé"
        elif "MOY" in impact or "MED" in impact:
            impact_norm = "Moyen"
        elif "FAIB" in impact or "BAS" in impact:
            impact_norm = "Faible"
        else:
            impact_norm = "Moyen"
        
        results.append({
            "date": dt_evt.strftime("%d/%m/%Y"),
            "type": type_ev.upper(),
            "description": desc,
            "impact": impact_norm
        })
    
    return results


def inject_missing_events(events, today, horizon, article):
    annee = today.year
    
    # Fetes fixes
    for nom_fete, mmjj in FETES_FIXES.items():
        # mmjj est maintenant au format MM-JJ, donc f"{annee}-{mmjj}" = AAAA-MM-JJ
        dt = datetime.strptime(f"{annee}-{mmjj}", "%Y-%m-%d")
        if today <= dt <= horizon:
            if not any(nom_fete.lower() in e["description"].lower() for e in events):
                events.append({
                    "date": dt.strftime("%d/%m/%Y"),
                    "type": "FETE",
                    "description": f"{nom_fete} : pic de consommation nationale",
                    "impact": "Élevé"
                })
    
    # Fetes lunaires (deja au format AAAA-MM-JJ, donc pas de probleme)
    for nom_fete, data in FETES_LUNAIRES_2026.items():
        if "date" in data:
            dt = datetime.strptime(data["date"], "%Y-%m-%d")
            if today <= dt <= horizon:
                if not any(nom_fete.lower() in e["description"].lower() for e in events):
                    events.append({
                        "date": dt.strftime("%d/%m/%Y"),
                        "type": "RELIGION",
                        "description": f"{nom_fete} : consommation familiale et nocturne",
                        "impact": "Élevé"
                    })
        elif "debut" in data:
            dt = datetime.strptime(data["debut"], "%Y-%m-%d")
            if today <= dt <= horizon:
                if not any(nom_fete.lower() in e["description"].lower() for e in events):
                    events.append({
                        "date": dt.strftime("%d/%m/%Y"),
                        "type": "RELIGION",
                        "description": f"{nom_fete} : debut de la periode de consommation nocturne",
                        "impact": "Élevé"
                    })
    
    # Tri chronologique final
    events.sort(key=lambda x: datetime.strptime(x["date"], "%d/%m/%Y"))
    return events


# ═══════════════════════════════════════════════════════════
# 4. APPEL GROQ (inchangé en surface, mais post-traité)
# ═══════════════════════════════════════════════════════════
def call_groq(article, jours):
    date_fin = TODAY + timedelta(days=jours)
    
    user_msg = f"""Genere une analyse complete de prevision des ventes et production pour :
- Article SFBT : {article}
- Duree : {jours} jours
- Periode : du {TODAY.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}

Retourne uniquement le JSON demande avec evenements ventes ET production."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.6,
        "max_tokens": 2048
    }

    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError("Aucun JSON trouve dans la reponse")

    data = json.loads(match.group())
    
    # ── POST-TRAITEMENT FACTUEL ──
    raw_events = data.get("evenements", [])
    events = post_process_events(raw_events, TODAY, date_fin)
    events = inject_missing_events(events, TODAY, date_fin, article)
    
    data["evenements"] = events
    data["date_debut"] = TODAY.strftime("%d/%m/%Y")
    data["date_fin"] = date_fin.strftime("%d/%m/%Y")
    
    return data


# ═══════════════════════════════════════════════════════════
# 5. ROUTES FLASK (conservées telles quelles)
# ═══════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "SFBT AI Backend operationnel — Groq LLaMA3 + Correcteur factuel"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON manquant"}), 400

        article = data.get('article', '').strip()
        jours = int(data.get('jours', 30))

        if not article:
            return jsonify({"error": "article manquant"}), 400

        result = call_groq(article, jours)
        return jsonify(result)

    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Erreur API Groq : {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "Message manquant"}), 400

        m = re.search(r'(\d+)\s*jours?', message, re.IGNORECASE)
        jours = int(m.group(1)) if m else 30

        article = re.sub(r'\d+\s*jours?', '', message, flags=re.IGNORECASE)
        article = re.sub(r'(prevision|prevision|analyse|pour|sur)\s*', '', article, flags=re.IGNORECASE)
        article = article.strip().upper() or message.upper()

        result = call_groq(article, jours)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
