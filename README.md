# Contextual Bandit Agent for Crisis Communication Optimization

This repository demonstrates the application of Reinforcement Learning (RL) and Large Language Models (LLMs) to automate and optimize crisis communication strategies. Designed as a portfolio project for an MSc application in Computer Science (specialization in Machine Learning and Reinforcement Learning) at the University of Alberta.

This script implements a Contextual Multi-Armed Bandit using Q-learning to dynamically select the most effective public relations messaging strategy based on the sentiment of preceding alerts.

## Overview

The system processes real-world vehicle theft data (São Paulo, Brazil) to simulate a local law enforcement or community safety social media account. It operates in two phases per incident:

1.  **Event Alert:** Generates a warning post using the Gemma-4-26b LLM and analyzes its sentiment impact using NLTK (VADER).
2.  **RL-Driven Follow-up:** Acts as a contextual bandit. It observes the negativity of the alert (the state) and selects one of three distinct follow-up prompts (actions: bravery, community cooperation, or visible presence) to generate a reassuring message. The sentiment of this generated response acts as the reward signal, updating a Q-table to improve future policy decisions.

The pipeline also includes spatial data visualization via Folium and sentiment distribution plotting via Matplotlib.

## Architecture & RL Formulation

```mermaid
graph TD
    %% Data Source
    Data[(Crime Data: Excel)] -->|Extract Incident| LLM1(LLM: Alert Generator)

    %% Phase 1: State Observation
    subgraph Phase 1: State Observation
        LLM1 -->|Gemma-4-26b| Alert[Alert Post]
        Alert -->|Analyze Sentiment| NLTK1(NLTK VADER)
        NLTK1 -->|Compound Score| Classifier{State Classifier}
        Classifier -->|Score < -0.5| S0[State 0: Very Negative]
        Classifier -->|-0.5 <= Score < 0| S1[State 1: Slightly Negative]
        Classifier -->|Score >= 0| S2[State 2: Neutral/Positive]
    end

    %% Phase 2: RL Agent (Contextual Bandit)
    subgraph Phase 2: Contextual Bandit Policy
        S0 & S1 & S2 --> Agent((RL Agent))
        Agent -.->|Reads state-action values| QTable[(Q-Table.json)]
        Agent -->|ε-greedy Policy| A0[Action 0: Bravery Prompt]
        Agent -->|ε-greedy Policy| A1[Action 1: Community Prompt]
        Agent -->|ε-greedy Policy| A2[Action 2: Presence Prompt]
    end

    %% Phase 3: Action Execution & Reward
    subgraph Phase 3: Execution & Reward
        A0 & A1 & A2 --> LLM2(LLM: Follow-up Generator)
        LLM2 -->|Gemma-4-26b| FollowUp[Follow-up Post]
        FollowUp -->|Analyze Sentiment| NLTK2(NLTK VADER)
        NLTK2 -->|Compound Score| Reward((Reward R))
    end

    %% Value Iteration Update Loop
    Reward -->|Bellman Update Loop| QTable
    
    %% Output Generation
    FollowUp -.-> OutTXT[mensagens_geradas.txt]
    Data -.-> OutMap[mapa_crimes.html]
    Reward -.-> OutChart[grafico_sentimentos.html]

    classDef llm fill:#f9f,stroke:#333,stroke-width:2px;
    classDef nlp fill:#bbf,stroke:#333,stroke-width:2px;
    classDef rl fill:#bfb,stroke:#333,stroke-width:2px;
    
    class LLM1,LLM2 llm;
    class NLTK1,NLTK2 nlp;
    class Agent,QTable rl;
