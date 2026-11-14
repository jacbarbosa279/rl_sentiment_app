import os
from pathlib import Path
import webbrowser
import folium
import matplotlib
matplotlib.use("Agg")  # Safe for Docker/headless environments.
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import time
import random
import json
from openai import APIError, OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer

OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "VeiculosSubtraidos_2024.xlsx"
MESSAGES_PATH = BASE_DIR / "data" / "processed" / "mensagens_geradas.txt"
OUTPUT_DIR = BASE_DIR / "output"
CHART_PATH = OUTPUT_DIR / "grafico_sentimentos.png"
MAP_PATH = OUTPUT_DIR / "mapa_crimes.html"
Q_TABLE_PATH = OUTPUT_DIR / "q_table.json"
LEGACY_Q_TABLE_PATH = BASE_DIR / "q_table.json"

# Download the VADER lexicon only when it is not already available.
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    if not nltk.download("vader_lexicon", quiet=True):
        raise RuntimeError(
            "Could not download the NLTK VADER lexicon. Check your internet connection "
            "or install the lexicon manually."
        )

# Initialize the NLTK sentiment analyzer.
sia = SentimentIntensityAnalyzer()

# ==========================================
# CONTEXTUAL-BANDIT CONFIGURATION
# ==========================================
ALPHA = 0.1      # Incremental action-value update rate
EPSILON = 0.2    # Exploration rate (randomly test an action 20% of the time)


def load_action_values():
    """Load previous action-value estimates or initialize a new table.

    The old q_table.json filename is still read for backward compatibility.
    The current update rule is an incremental contextual-bandit estimate,
    not temporal-difference Q-learning (there is no next-state bootstrap).
    """
    for path in (Q_TABLE_PATH, LEGACY_Q_TABLE_PATH):
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
    # Contexts: "0" (very negative), "1" (slightly negative), "2" (neutral/positive)
    # Actions: "0", "1", "2"
    return {
        state: {action: 0.0 for action in ("0", "1", "2")}
        for state in ("0", "1", "2")
    }


def save_action_values(action_values):
    """Persist action-value estimates for subsequent runs."""
    Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Q_TABLE_PATH, "w", encoding="utf-8") as file:
        json.dump(action_values, file, indent=4)


def get_state(robbery_score):
    """Map the robbery sentiment score to a discrete state."""
    if robbery_score < -0.5:
        return "0"  # Very negative
    elif robbery_score < 0:
        return "1"  # Slightly negative
    return "2"      # Neutral/positive


def choose_action(state, action_values):
    """Choose which prompt to use (exploration vs. exploitation)."""
    if random.uniform(0, 1) < EPSILON:
        return str(random.choice([0, 1, 2]))  # Explore
    else:
        actions = action_values[state]
        return max(actions, key=actions.get)  # Exploit the best option


# ==========================================
# LLM CLIENT & BILINGUAL GENERATION
# ==========================================

def create_llm_client():
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENROUTER_API_KEY is not set. "
            "Messages will be generated locally without using the API."
        )
        return None

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def generate_message(client, prompt, local_fallback, index, message_type):
    if client is None:
        return local_fallback

    try:
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        message = completion.choices[0].message.content
        if message and message.strip():
            return message.strip()
        raise ValueError("The API returned an empty message.")
    except (APIError, IndexError, ValueError) as error:
        print(
            f"API error while generating {message_type} on row {index}: {error}. "
            "A local message will be used."
        )
        return local_fallback


def generate_bilingual_message(client, prompt, fallback_en, fallback_pt, index, message_type):
    """Generate both English (for VADER) and Portuguese (for display/logs) versions."""
    raw_response = generate_message(client, prompt, "", index, message_type)
    if not raw_response:
        return fallback_en, fallback_pt

    try:
        cleaned = (
            raw_response.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
        text_en = data.get("en", "").strip() or fallback_en
        text_pt = data.get("pt", "").strip() or fallback_pt
        return text_en, text_pt
    except (json.JSONDecodeError, AttributeError):
        print(f"JSON parse warning on row {index} ({message_type}). Using fallback.")
        return fallback_en, fallback_pt


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def main():
    client = create_llm_client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MESSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    robbery_data = pd.read_excel(DATA_PATH, sheet_name='AMOSTRAGEM')
    crime_map = folium.Map(location=[-23.550520, -46.633308], zoom_start=12)

    robbery_sentiments = []
    policing_sentiments = []

    # Load contextual-bandit action-value estimates.
    action_values = load_action_values()

    json_instruction = (
        '\nReturn ONLY a valid JSON object with two keys: '
        '"en" (the message in English) and "pt" (the exact same message in Brazilian Portuguese).'
    )

    with open(MESSAGES_PATH, "w", encoding="utf-8", buffering=1) as messages_file:
        for index, row in robbery_data.iterrows():
            crime_type = "vehicle theft"
            neighborhood = row['BAIRRO']
            incident_datetime = row['DATA_OCORRENCIA_BO']
            hour = row['HORA_OCORRENCIA']
            vehicle_type = row['DESCR_TIPO_VEICULO']

            try:
                hour = pd.to_datetime(hour, format='%H:%M:%S')
                formatted_hour = hour.strftime('%H:%M:%S')
            except Exception:
                formatted_hour = "Not available"

            content = (
                f"Create a social media post warning about a real {crime_type} involving a vehicle "
                f"(in Portuguese: '{vehicle_type}', translate this vehicle type to English in the 'en' field) "
                f"in the neighborhood of {neighborhood} on {incident_datetime.strftime('%d/%m/%Y')} at {formatted_hour}. "
                f"Include relevant hashtags.{json_instruction}"
            )

            robbery_post_pt = ""

            try:
                robbery_fallback_en = (
                    f"Alert: a {crime_type} ({vehicle_type}) was reported "
                    f"in the neighborhood of {neighborhood} on "
                    f"{incident_datetime.strftime('%d/%m/%Y')} at {formatted_hour}. "
                    "Please stay alert. #Safety"
                )
                robbery_fallback_pt = (
                    f"Alerta: um roubo de veículo ({vehicle_type}) foi registrado "
                    f"no bairro {neighborhood} em {incident_datetime.strftime('%d/%m/%Y')} às {formatted_hour}. "
                    "Fique atento. #Segurança"
                )

                robbery_post_en, robbery_post_pt = generate_bilingual_message(
                    client, content, robbery_fallback_en, robbery_fallback_pt, index, "robbery post"
                )

                print(f"\n--- Robbery Post (Row {index}) ---")
                print(f"[PT]: {robbery_post_pt}")
                print(f"[EN]: {robbery_post_en}")

                robbery_record = (
                    f"Robbery post (PT):\n{robbery_post_pt}\n"
                    f"Robbery post (EN):\n{robbery_post_en}\n---\n"
                )
                messages_file.write(robbery_record)

                # VADER evaluates the English text to match its lexicon.
                robbery_sentiment = sia.polarity_scores(robbery_post_en)
                robbery_sentiments.append(robbery_sentiment)
                print("Robbery sentiment:", robbery_sentiment)

                robbery_sentiment_record = (
                    f"Robbery post sentiment: {robbery_sentiment}\n---\n"
                )
                messages_file.write(robbery_sentiment_record)
                messages_file.flush()

                # =========================================================
                # CONTEXTUAL-BANDIT DECISION CYCLE
                # =========================================================

                current_state = get_state(robbery_sentiment['compound'])
                chosen_action = choose_action(current_state, action_values)

                if chosen_action == "0":
                    policing_content = (
                        f"Write a very positive message praising the police's quick response and bravery "
                        f"in the neighborhood of {neighborhood} after the incident. Convey confidence.{json_instruction}"
                    )
                elif chosen_action == "1":
                    policing_content = (
                        f"Write an encouraging message about how cooperation between the community in {neighborhood} "
                        f"and the police is making the area safer.{json_instruction}"
                    )
                else:
                    policing_content = (
                        f"Write an informative, positive message highlighting preventive patrols and police presence "
                        f"in the neighborhood of {neighborhood}.{json_instruction}"
                    )

                policing_fallback_en = (
                    f"Police remain present in the neighborhood of {neighborhood} "
                    "to help ensure public safety. #Community"
                )
                policing_fallback_pt = (
                    f"A polícia segue presente no bairro {neighborhood} "
                    "para garantir a segurança pública. #Comunidade"
                )

                policing_post_en, policing_post_pt = generate_bilingual_message(
                    client, policing_content, policing_fallback_en, policing_fallback_pt,
                    index, "policing post"
                )

                print(f"--- Policing Post (RL Action: {chosen_action}) ---")
                print(f"[PT]: {policing_post_pt}")
                print(f"[EN]: {policing_post_en}")

                messages_file.write(
                    f"Policing post (PT):\n{policing_post_pt}\n"
                    f"Policing post (EN):\n{policing_post_en}\n---\n"
                )

                # VADER evaluates the English text to compute the bandit proxy reward.
                policing_sentiment = sia.polarity_scores(policing_post_en)
                policing_sentiments.append(policing_sentiment)

                reward = policing_sentiment['compound']
                print(f"Policing sentiment (proxy reward): {reward:.4f}")

                # Incremental contextual-bandit action-value update.
                old_value = action_values[current_state][chosen_action]
                action_values[current_state][chosen_action] = (
                    old_value + ALPHA * (reward - old_value)
                )

                # =======================================================
                # END OF CONTEXTUAL-BANDIT DECISION CYCLE
                # =====================================================

                messages_file.write(
                    f"Policing post sentiment: {policing_sentiment}\n---\n"
                )
                messages_file.flush()

            except Exception as e:
                print(f"Error processing row {index}: {e}")

            latitude = row['LATITUDE']
            longitude = row['LONGITUDE']
            folium.Marker(
                location=[latitude, longitude],
                popup=(
                    f"<b>Bairro:</b> {row['BAIRRO']}<br>"
                    f"<b>Data:</b> {incident_datetime.strftime('%d/%m/%Y')}<br>"
                    f"<b>Hora:</b> {formatted_hour}<br>"
                    f"<b>Alerta:</b> {robbery_post_pt}"
                ),
                tooltip="Clique para detalhes",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(crime_map)

            if client is not None:
                time.sleep(3)

    save_action_values(action_values)
    print("\n--- Updated action-value estimates ---")
    print(json.dumps(action_values, indent=2))

    crime_map.save(MAP_PATH)
    print(f"Map saved to: {MAP_PATH}")

    generate_sentiment_chart(robbery_sentiments, policing_sentiments)
    print(f"Messages saved to: {MESSAGES_PATH}")
    print(f"Chart saved to: {CHART_PATH}")


def generate_sentiment_chart(robbery_sentiments, policing_sentiments):
    if not robbery_sentiments and not policing_sentiments:
        robbery_scores = []
        policing_scores = []
    else:
        robbery_scores = [s['compound'] for s in robbery_sentiments]
        policing_scores = [s['compound'] for s in policing_sentiments]

    plt.figure(figsize=(10, 5))
    if robbery_scores or policing_scores:
        plt.bar(range(len(robbery_scores)), robbery_scores, label='Robbery', color='red', alpha=0.7)
    if policing_scores:
        plt.bar(
            range(len(robbery_scores), len(robbery_scores) + len(policing_scores)),
            policing_scores,
            label='Policing',
            color='blue',
            alpha=0.7
        )
    if not robbery_scores and not policing_scores:
        plt.text(0.5, 0.5, 'No sentiment data recorded', ha='center', va='center')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.xlabel('Message Index')
    plt.ylabel('Sentiment Score (Compound)')
    plt.title('Post Sentiment Analysis')
    if robbery_scores or policing_scores:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, format='png', bbox_inches='tight')
    print(f"Chart saved to {CHART_PATH}")
    plt.close()

    if os.environ.get("APP_HEADLESS", "0").lower() not in {"1", "true", "yes"}:
        if not webbrowser.open(CHART_PATH.resolve().as_uri()):
            print("Could not open the chart automatically; open the PNG file at the path above.")


if __name__ == "__main__":
    main()