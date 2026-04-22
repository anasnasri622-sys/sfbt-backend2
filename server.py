from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, json, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

TODAY = datetime(2026, 4, 22)

SYSTEM_PROMPT = """Tu es un expert commercial senior pour la SFBT (Société des Boissons de Tunisie).

ÉVÉNEMENTS TUNISIENS RÉELS À UTILISER :
- Fêtes nationales : 1er mai (Fête du travail), 25 juillet (Fête de la République), 13 août (Fête de la Femme), 15 octobre (Fête de l'Evacuation), 7 novembre (Fête nationale)
- Aïd El Adha 2026 : 26-27 mai 2026
- Saison estivale juin-août : pic de consommation boissons +30 à +50%
- Ligue 1 Tunisienne : matchs les samedis et dimanches
- Ramadan 2026 : 18 février au 19 mars 2026
- Fêtes scolaires : rentrée septembre, vacances décembre

Retourne UNIQUEMENT un JSON valide, sans texte avant ni après, sans balises markdown.
Format exact :
{
  "article": "nom de l'article",
  "jours": 30,
  "date_debut": "JJ/MM/AAAA",
  "date_fin": "JJ/MM/AAAA",
  "resume": "Analyse commerciale en 2 phrases",
  "tendance": "hausse",
  "variation_pct": 15,
  "evenements": [
    {
      "date": "JJ/MM/2026",
      "description": "Description précise de l'événement",
      "impact": "Élevé",
      "type": "sport"
    }
  ]
}
Règles strictes :
- tendance = hausse | stable | baisse
- impact = Élevé | Moyen | Faible  
- type = sport | meteo | fete | marketing | religion
- Minimum 8 événements, maximum 12
- Dates strictement dans la plage demandée
- Événements réalistes pour le marché tunisien des boissons"""


def call_groq(article, jours):
    date_fin = TODAY + timedelta(days=jours)

    user_msg = f"""Génère une analyse de prévision des ventes pour :
- Article SFBT : {article}
- Durée : {jours} jours
- Période : du {TODAY.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}

Retourne uniquement le JSON demandé."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
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
        raise ValueError("Aucun JSON trouvé dans la réponse")

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
        article = re.sub(r'(prévision|prevision|analyse|pour|sur)\s*', '', article, flags=re.IGNORECASE)
        article = article.strip().upper() or message.upper()

        result = call_groq(article, jours)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

