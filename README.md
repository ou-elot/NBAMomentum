# NBAMomentum
Analysis of NBA games from 2024-2025 for detection and quantification of momentum


# 📘 NBA Momentum Analysis Pipeline

This repository implements an end-to-end workflow for analyzing momentum in NBA games using:

- Possession-level event extraction  
- S/0/T outcome encoding  
- A 3-state Hidden Markov Model (HOT / NEUTRAL / COLD)  
- Empirical transition and emission matrices  
- League-wide statistical tests (two-proportion z-tests)

---

## 📂 Table of Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Pipeline Summary](#pipeline-summary)  
- [Step-by-Step Instructions](#step-by-step-instructions)  
  1. [Pull possessions (`dataPull.py`)](#1-pull-possessions-datapullpy)  
  2. [Compute probabilities (`probabilities.py`)](#2-compute-probabilities-probabilitiespy)  
  3. [Run HMM for one team (`specificTeam.py`)](#3-run-hmm-for-one-team-specificteampy)  
  4. [Full-season HMM (`TeamPipeline.py`)](#4-full-season-hmm-teampipelinepy)  
  5. [League-wide HMM aggregation (`allTeams.py`)](#5-league-wide-hmm-aggregation-allteamspy)  
  6. [Statistical tests (`teamZ.py` / `allZ.py`)](#6-statistical-tests-teamzpy--allzpy)  
  7. [Optional: aggregate saved states (`aggregate_from_outputs.py`)](#7-optional-aggregate-saved-states-aggregate_from_outputspy)  
- [Output Directory Structure](#output-directory-structure)  
- [Notes](#notes)

---

## Overview

This project analyzes NBA possession sequences to detect whether “momentum” exists in offensive outcomes.

Each possession is labeled as:

- **S** = score  
- **0** = no score  
- **T** = turnover  

A 3-state Hidden Markov Model is then trained to detect **HOT / NEUTRAL / COLD** underlying states.

The pipeline supports per-game, per-team, and full-season league-wide analysis.

---

## Installation

```bash
pip install pbpstats pandas numpy nba_api
```

---

## Pipeline Summary

For each game:

- Extract possessions and outcomes  
- Build S/0/T sequence  
- Compute raw + conditional probabilities  
- Fit a 3-state HMM  
- Extract Viterbi path  
- Aggregate across games/teams  
- Run statistical momentum tests

---

## Step-by-Step Instructions

### 1. Pull possessions (`dataPull.py`)

Extracts play-by-play data with `pbpstats`, constructs possessions, splits by team, and outputs S/0/T sequences.

```bash
python dataPull.py --game-id 002240XXXX
```

Creates:

```
GAMEID_outputs/
├── GAMEID_ALL_possessions.csv
├── GAMEID_TEAM_possessions.csv
└── GAMEID_TEAM_sequence.txt
```

---

### 2. Compute probabilities (`probabilities.py`)

Computes raw and conditional possession outcome probabilities for each team in a game.

```bash
python probabilities.py 002240XXXX
```

Outputs:

```
GAMEID_probs.csv
GAMEID_cond_probs.csv
```

> Required for z-tests and HMM pipeline.

---

### 3. Run HMM for one team (`specificTeam.py`)

Fits the 3-state HMM for one team in one game.

```bash
python specificTeam.py LAL 002240XXXX
```

Produces:

- HMM fit  
- Viterbi state sequence (printed)

---

### 4. Full-season HMM (`TeamPipeline.py`)

Runs the full pipeline (dataPull → probabilities → HMM) across all games of a team.

```bash
python TeamPipeline.py LAL 2024-25
```

Outputs per-season transition probabilities.

---

### 5. League-wide HMM aggregation (`allTeams.py`)

Runs HMM pipeline across all 30 teams, aggregates transitions + emissions.

```bash
python allTeams.py --season 2024-25
```

Creates:

```
TEAM_202425_HMM_transitions.csv
TEAM_202425_HMM_emissions.csv
```

---

### 6. Statistical tests (`teamZ.py` / `allZ.py`)

Two-proportion z-tests comparing:

```
P(to | from) vs P(to) (raw)
```

#### Single team:

```bash
python teamZ.py LAL
```

#### All teams:

```bash
python allZ.py
```

Outputs:

```
outputs/ztests_TEAM.csv
```

---

### 7. Optional: aggregate saved states (`aggregate_from_outputs.py`)

If you already created:

- `GAMEID_TEAM_states.csv`  
- `GAMEID_TEAM_possessions.csv` (with outcomes)  

You can aggregate transitions without re-running HMM training.

```bash
python aggregate_from_outputs.py
```

---

## Output Directory Structure

Every processed game creates:

```
002240XXXX_outputs/
├── 002240XXXX_ALL_possessions.csv
├── 002240XXXX_TEAM_possessions.csv
├── 002240XXXX_TEAM_sequence.txt
├── 002240XXXX_probs.csv
├── 002240XXXX_cond_probs.csv
└── (optional) 002240XXXX_TEAM_states.csv
```

Season- or league-level outputs:

```
TEAM_202425_HMM_transitions.csv
TEAM_202425_HMM_emissions.csv
outputs/ztests_TEAM.csv
```

---

## Notes

- `dataPull.py` must be run before anything else for a new game.  
- `probabilities.py` must run before `teamZ.py` or `allZ.py`.  
