from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, json, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

TODAY = datetime(2026, 4, 22)

SYSTEM_PROMPT = """Tu es un expert commercial et industriel senior pour la SFBT (Societe des Boissons de Tunisie).

MISSION : Analyser tous les evenements qui influencent les VENTES et la PRODUCTION d'un article SFBT sur une periode donnee en Tunisie.

=== EVENEMENTS QUI INFLUENCENT LES VENTES ===

FETES ET JOURS FERIES TUNISIENS :
- 1er mai : Fete du Travail - forte demande GMS
- 25 juillet : Fete de la Republique - pic consommation
- 13 aout : Fete de la Femme - promotions distributeurs
- 15 octobre : Fete de l'Evacuation
- 7 novembre : Fete Nationale - tres forte demande
- Aid El Adha 2026 : 26-27 mai - pic familial ++
- Ramadan 2026 : 18 fevrier au 19 mars - consommation nocturne

SPORT ET GRANDS EVENEMENTS :
- Ligue 1 Tunisienne : matchs samedis/dimanches - hausse ventes avant-match
- Matchs selection nationale tunisienne : tres forte affluence stades
- Finale Coupe de Tunisie : evenement national majeur
- Matchs a forte audience TV : hausse ventes dans les cafes et GMS

METEO ET SAISONS :
- Canicule >35 degres (juin-aout) : pic de consommation boissons +40 a +60%
- Saison estivale (juin-aout) : periode la plus forte de l'annee
- Pluies et froid (decembre-janvier) : baisse consommation -20 a -30%
- Printemps (mars-mai) : reprise progressive de la demande

MARKETING ET DISTRIBUTION :
- Promotions GMS (Carrefour, Monoprix, MG) : impact immediat ventes
- Campagnes publicitaires TV SFBT : hausse notoriete
- Rentree scolaire septembre : relance consommation familles
- Vacances estivales juillet-aout : pics touristiques zones cotieres

=== EVENEMENTS QUI INFLUENCENT LA PRODUCTION ===

MATIERES PREMIERES :
- Hausse prix sucre mondial : impact cout de production +15 a +25%
- Hausse prix aluminium : impact cout boites metalliques
- Variation prix CO2 industriel : impact production boissons gazeuses
- Hausse prix plastique PET : impact bouteilles plastique
- Fluctuation prix eau industrielle : impact direct production

CONCURRENCE :
- Lancement nouveau produit concurrent (Boga, Coca-Cola, Pepsi) : pression sur parts de marche
- Promotion agressive concurrent : risque perte clients distributeurs
- Entree nouvelle marque etrangere sur le marche tunisien

LOGISTIQUE ET SUPPLY CHAIN :
- Greves portuaires ou transporteurs : retard approvisionnement matieres
- Penurie d'emballages sur le marche : ralentissement production
- Hausse prix carburant : impact cout livraison et distribution
- Arret technique programme usine SFBT : baisse temporaire production

CONTEXTE ECONOMIQUE TUNISIEN :
- Inflation generale : impact pouvoir d'achat consommateurs
- Devaluation dinar tunisien : hausse cout matieres importees
- Restrictions importation matieres premieres : tension approvisionnement

=== FORMAT DE REPONSE OBLIGATOIRE ===
Retourne UNIQUEMENT un JSON valide, sans texte avant ni apres, sans balises markdown :

{
  "article": "nom exact de l'article",
  "jours": 30,
  "date_debut": "JJ/MM/AAAA",
  "date_fin": "JJ/MM/AAAA",
  "resume": "Analyse commerciale et industrielle en 2 phrases",
  "tendance": "hausse",
  "variation_pct": 15,
  "evenements": [
    {
      "date": "JJ/MM/2026",
      "description": "Description precise de l'evenement et son impact chiffre",
      "impact": "Eleve",
      "type": "sport"
    }
  ]
}

REGLES STRICTES :
- tendance = hausse | stable | baisse
- impact = Eleve | Moyen | Faible
- type = sport | meteo | fete | marketing | religion | production | concurrence | economie
- Minimum 10 evenements, maximum 14
- Melanger evenements ventes ET production (au moins 3 evenements production)
- Dates strictement dans la plage demandee
- Descriptions precises avec impact chiffre quand possible (+20%, -15%, etc.)
- Evenements realistes et contextualises pour la Tunisie"""


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
        "temperature": 0.7,
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

    return json.loads(match.group())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "SFBT AI Backend operationnel — Groq LLaMA3"})


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

