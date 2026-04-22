from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, json, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

TODAY = datetime(2026, 4, 22)

SYSTEM_PROMPT = """Tu es un expert commercial senior pour la SFBT (Société des Boissons de Tunisie).

ÉVÉNEMENTS TUNISIENS RÉELS :
- Fêtes nationales : 1er mai, 25 juillet, 13 août, 15 octobre, 7 novembre
- Aïd El Adha 2026 : 26-27 mai 2026
- Saison estivale juin-août : pic de consommation boissons +30 à +50%
- Ligue 1 Tunisienne : matchs les samedis et dimanches
- Ramadan 2026 : 18 février au 19 mars 2026

Retourne UNIQUEMENT un JSON valide, sans texte avant ni après, sans markdown :
{
  "article": "nom de l'article",
  "jours": 30,
  "date_debut": "JJ/MM/AAAA",
  "date_fin": "JJ/MM/AAAA",
  "resume": "Analyse en 2 phrases",
  "tendance": "hausse",
  "variation_pct": 15,
  "evenements": [
    {"date": "JJ/MM/2026", "description": "...", "impact": "Élevé", "type": "sport"}
  ]
}
Règles : tendance = hausse|stable|baisse, impact = Élevé|Moyen|Faible, type = sport|meteo|fete|marketing|religion, minimum 8 événements."""


def call_gemini(article, jours):
    date_fin = TODAY + timedelta(days=jours)
    prompt = f"""{SYSTEM_PROMPT}

Article : {article}
Durée : {jours} jours
Du : {TODAY.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')}"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
    }

    resp = requests.post(GEMINI_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError("Aucun JSON trouvé")
    return json.loads(match.group())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "SFBT AI Backend operationnel"})


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
        result = call_gemini(article, jours)
        return jsonify(result)
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

        result = call_gemini(article, jours)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

