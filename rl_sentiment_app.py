import os
from pathlib import Path
import webbrowser
import folium
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

# Download the resources required by NLTK.
nltk.download('vader_lexicon', quiet=True)

# Initialize the NLTK sentiment analyzer.
sia = SentimentIntensityAnalyzer()

# ==========================================
# REINFORCEMENT LEARNING CONFIGURATION
# ==========================================
ALPHA = 0.1      # Learning rate
EPSILON = 0.2    # Exploration rate (randomly test an action 20% of the time)

def load_q_table():
    """Load the agent's previous knowledge or create a new table."""
    if Q_TABLE_PATH.exists():
        with open(Q_TABLE_PATH, "r") as f:
            return json.load(f)
    if LEGACY_Q_TABLE_PATH.exists():
        with open(LEGACY_Q_TABLE_PATH, "r") as f:
            return json.load(f)
    # States: "0" (very negative), "1" (slightly negative), "2" (neutral/positive)
    # Actions: "0", "1", "2"
    return {
        "0": {"0": 0.0, "1": 0.0, "2": 0.0},
        "1": {"0": 0.0, "1": 0.0, "2": 0.0},
        "2": {"0": 0.0, "1": 0.0, "2": 0.0}
    }

def save_q_table(q_table):
    """Save the learned values for the next run."""
    Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Q_TABLE_PATH, "w") as f:
        json.dump(q_table, f, indent=4)

def get_state(robbery_score):
    """Map the robbery sentiment score to a discrete state."""
    if robbery_score < -0.5: return "0" # Very negative
    elif robbery_score < 0:  return "1" # Slightly negative
    return "2"                        # Neutral/positive

def choose_action(state, q_table):
    """Choose which prompt to use (exploration vs. exploitation)."""
    if random.uniform(0, 1) < EPSILON:
        return str(random.choice([0, 1, 2])) # Explore
    else:
        actions = q_table[state]
        return max(actions, key=actions.get)     # Exploit the best option


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


def main():
    client = create_llm_client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MESSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    robbery_data = pd.read_excel(DATA_PATH, sheet_name='AMOSTRAGEM')
    crime_map = folium.Map(location=[-23.550520, -46.633308], zoom_start=12)

    robbery_sentiments = []
    policing_sentiments = []

    # Load the reinforcement learning Q-table.
    q_table = load_q_table()

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

            content = f"""Create a social media post warning about a real {crime_type} involving a {vehicle_type}
in the neighborhood of {neighborhood} on {incident_datetime.strftime('%d/%m/%Y')} at {formatted_hour}.
Include relevant hashtags."""

            try:
                robbery_fallback = (
                    f"Alert: a {crime_type} involving a {vehicle_type} was reported "
                    f"in the neighborhood of {neighborhood} on "
                    f"{incident_datetime.strftime('%d/%m/%Y')} at {formatted_hour}. "
                    "Please stay alert. #Safety"
                )
                robbery_post = generate_message(
                    client, content, robbery_fallback, index, "robbery post"
                )
                
                print(f"\n--- Robbery Post (Row {index}) ---\n{robbery_post}")
                
                robbery_record = f"Robbery post:\n{robbery_post}\n---\n"
                messages_file.write(robbery_record)

                robbery_sentiment = sia.polarity_scores(robbery_post)
                robbery_sentiments.append(robbery_sentiment)
                print("Robbery sentiment:", robbery_sentiment)
                
                robbery_sentiment_record = (
                    f"Robbery post sentiment: {robbery_sentiment}\n---\n"
                )
                messages_file.write(robbery_sentiment_record)
                messages_file.flush()

                # =========================================================
                # REINFORCEMENT LEARNING CYCLE
                # =========================================================
                
                current_state = get_state(robbery_sentiment['compound'])
                chosen_action = choose_action(current_state, q_table)
                
                if chosen_action == "0":
                    policing_content = f"Write a very positive message praising the police's quick response and bravery in the neighborhood of {neighborhood} after the incident. Convey confidence."
                elif chosen_action == "1":
                    policing_content = f"Write an encouraging message about how cooperation between the community in {neighborhood} and the police is making the area safer."
                else:
                    policing_content = f"Write an informative, positive message highlighting preventive patrols and police presence in the neighborhood of {neighborhood}."

                policing_fallback = (
                    f"Police remain present in the neighborhood of {neighborhood} "
                    "to help ensure public safety. #Community"
                )
                
                policing_post = generate_message(
                    client, policing_content, policing_fallback, index,
                    "policing post"
                )
                print(
                    f"--- Policing Post (RL Action: {chosen_action}) ---\n"
                    f"{policing_post}"
                )
                
                messages_file.write(f"Policing post:\n{policing_post}\n---\n")

                # The agent receives a reward.
                policing_sentiment = sia.polarity_scores(policing_post)
                policing_sentiments.append(policing_sentiment)
                
                reward = policing_sentiment['compound']
                print(f"Policing sentiment (RL reward): {reward:.4f}")

                # Update the Q-table.
                current_q = q_table[current_state][chosen_action]
                new_q = current_q + ALPHA * (reward - current_q)
                q_table[current_state][chosen_action] = new_q
                
                # =======================================================
                # END OF REINFORCEMENT LEARNING CYCLE
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
                popup=f"<b>Neighborhood:</b> {row['BAIRRO']}<br><b>Date:</b> {incident_datetime.strftime('%d/%m/%Y')}<br><b>Time:</b> {formatted_hour}",
                tooltip="Click for details",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(crime_map)

            if client is not None:
                time.sleep(3)

    save_q_table(q_table)
    print("\n--- Updated Q-table (Agent Memory) ---")
    print(json.dumps(q_table, indent=2))

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

    if not webbrowser.open(CHART_PATH.resolve().as_uri()):
        print("Could not open the chart automatically; open the PNG file at the path above.")


if __name__ == "__main__":
    main()