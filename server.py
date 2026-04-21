"""
SFBT — Backend IA Prévision
Stack : Flask + Google Gemini Flash (gratuit) + Google Search intégré
Déploiement : Render.com (gratuit)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # autorise les appels depuis le HTML (GitHub Pages / Power BI)

# Clé API Gemini — obtenez-la gratuitement sur https://aistudio.google.com/
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "VOTRE_CLE_ICI")
genai.configure(api_key=GEMINI_API_KEY)

TODAY = datetime(2026, 4, 17)

SYSTEM_PROMPT = """Tu es un expert commercial senior pour la SFBT (Société des Boissons de Tunisie).

MISSION : Analyser les événements réels qui vont influencer les ventes d'un article SFBT 
sur une période donnée en Tunisie.

INSTRUCTIONS :
1. Utilise Google Search pour trouver les événements RÉELS et DATÉS pour la période :
   - Calendrier Ligue 1 tunisienne 2025-2026 (dates exactes des journées)
   - Fêtes nationales tunisiennes officielles (1er mai, 25 juillet, 13 août, 15 oct, 7 nov)
   - Dates exactes Aïd El Fitr et Aïd El Adha 2026 selon calendrier islamique
   - Matchs de la sélection nationale tunisienne
   - Saison estivale (pic de consommation boissons juin-août)
   - Événements promotionnels GMS (grandes surfaces) tunisiennes

2. Retourne UNIQUEMENT un objet JSON valide, sans texte avant ni après, ni balises markdown :

{
  "article": "nom exact de l'article",
  "jours": 30,
  "date_debut": "JJ/MM/AAAA",
  "date_fin": "JJ/MM/AAAA",
  "resume": "Analyse commerciale en 2 phrases maximum",
  "tendance": "hausse",
  "variation_pct": 15,
  "evenements": [
    {
      "date": "JJ/MM/2026",
      "description": "Description précise et réaliste de l'événement",
      "impact": "Élevé",
      "type": "sport"
    }
  ]
}

RÈGLES :
- tendance = hausse | stable | baisse
- impact = Élevé | Moyen | Faible
- type = sport | meteo | fete | marketing | religion
- Minimum 8 événements, maximum 12
- Dates strictement dans la plage demandée
- Événements réalistes pour le marché tunisien des boissons
"""


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "SFBT AI Backend opérationnel"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Corps de requête JSON manquant"}), 400

        article = data.get('article', '').strip()
        jours = int(data.get('jours', 30))

        if not article:
            return jsonify({"error": "Champ 'article' manquant"}), 400

        date_debut = TODAY
        date_fin = TODAY + timedelta(days=jours)

        user_message = f"""Génère une analyse de prévision pour :
- Article : {article}
- Durée : {jours} jours
- Du : {date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}

Recherche les événements réels tunisiens sur cette période et retourne le JSON demandé."""

        # Appel Gemini Flash avec Google Search intégré
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
            tools="google_search_retrieval"  # Google Search natif — gratuit
        )

        response = model.generate_content(user_message)
        raw_text = response.text

        # Nettoyage du JSON
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*', '', raw_text)
        raw_text = raw_text.strip()

        # Extraction JSON robuste
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("Aucun JSON trouvé dans la réponse")

        result = json.loads(json_match.group())
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Erreur parsing JSON : {str(e)}", "raw": raw_text[:300]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint conversation libre avec le chatbot SFBT"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "Message manquant"}), 400

        # Détection article + jours dans le message
        jours_match = re.search(r'(\d+)\s*jours?', message, re.IGNORECASE)
        jours = int(jours_match.group(1)) if jours_match else 30

        # Suppression du nombre de jours pour extraire l'article
        article_raw = re.sub(r'\d+\s*jours?', '', message, flags=re.IGNORECASE)
        article_raw = re.sub(r'(prévision|prevision|analyse|pour|sur)\s*', '', article_raw, flags=re.IGNORECASE)
        article = article_raw.strip().upper() or message.upper()

        # Appel endpoint predict
        with app.test_request_context(
            '/predict', method='POST',
            json={'article': article, 'jours': jours},
            content_type='application/json'
        ):
            resp = predict()
            return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
