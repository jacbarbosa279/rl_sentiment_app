# Contextual Bandit Agent for Crisis Communication Optimization

This repository demonstrates the application of Reinforcement Learning (RL) and Large Language Models (LLMs) to automate and optimize crisis communication strategies. Designed as a portfolio project for an MSc application in Computer Science (specialization in Machine Learning and Reinforcement Learning) at the University of Alberta.

This script implements a Contextual Multi-Armed Bandit using Q-learning to dynamically select the most effective public relations messaging strategy based on the sentiment of preceding alerts.

## Overview

The system processes real-world vehicle theft data (São Paulo, Brazil) to simulate a local law enforcement or community safety social media account. It operates in two phases per incident:

1.  **Event Alert:** Generates a warning post using the Gemma-4-26b LLM and analyzes its sentiment impact using NLTK (VADER).
2.  **RL-Driven Follow-up:** Acts as a contextual bandit. It observes the negativity of the alert (the state) and selects one of three distinct follow-up prompts (actions: bravery, community cooperation, or visible presence) to generate a reassuring message. The sentiment of this generated response acts as the reward signal, updating a Q-table to improve future policy decisions.

The pipeline also includes spatial data visualization via Folium and sentiment distribution plotting via Matplotlib.

## Architecture & RL Formulation

This project models the prompt selection problem as a Contextual Multi-Armed Bandit:

*   **State Space ($\mathcal{S}$):** Discrete representation of the initial alert's negativity (derived from VADER's compound score).
    *   State 0: Very Negative ($< -0.5$)
    *   State 1: Slightly Negative ($[-0.5, 0)$)
    *   State 2: Neutral/Positive ($\ge 0$)
*   **Action Space ($\mathcal{A}$):** Three distinct LLM prompt engineering strategies for the follow-up message:
    *   Action 0: Emphasize rapid response and police bravery.
    *   Action 1: Emphasize community cooperation and unity.
    *   Action 2: Emphasize preventive patrols and visible presence.
*   **Reward Function ($\mathcal{R}$):** The continuous compound sentiment score ($[-1, 1]$) of the generated follow-up post, evaluated by NLTK. The goal is to maximize the positivity (reassurance) of the follow-up.
*   **Policy & Update Rule:** The agent employs an $\epsilon$-greedy policy ($\epsilon=0.2$) to balance exploration of prompt strategies with the exploitation of known high-reward strategies. The Q-values are updated using a simplified Bellman equation (bandit setting with no future state transitions): $Q(s,a) \leftarrow Q(s,a) + \alpha [r - Q(s,a)]$, where $\alpha=0.1$.

## Prerequisites

*   Python 3.8+
*   An API Key from [OpenRouter](https://openrouter.ai/) (or OpenAI) to query the Gemma model.

### Dependencies

Install the required packages via pip:

```bash
pip install pandas folium matplotlib nltk openai openpyxl
