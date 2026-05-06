# 🎾 Match Point Simulator

An interactive tennis strategy and expected value engine. This project leverages historical match data and machine learning to predict optimal shot placement and simulate rally outcomes.

## 🏗️ Architecture & Tech Stack
* **Data Source:** Jeff Sackmann's Match Charting Project
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas
* **Backend API:** FastAPI (Uvicorn, Pydantic)
* **Frontend UI:** Streamlit, Plotly
* **Infrastructure:** Docker & Docker Compose

## 🚀 Local Development Setup

### Option 1: Using Python Virtual Environment (Recommended for active dev)
1. Ensure you have Python 3.10+ installed on your M2 Mac.
2. Create and activate a virtual environment:
    `python3 -m venv venv`
    `source venv/bin/activate`
3. Install dependencies:
    `pip install -r requirements.txt`
4. Run the API and Frontend locally (in separate terminals):
    `uvicorn src.api:app --reload`
    `streamlit run frontend/app.py`

### Option 2: Using Docker (Recommended for testing portability)
Ensure you have Docker Desktop installed.
1. Build and spin up the containers:
    `docker-compose up --build`
2. The FastAPI backend will be available at: `http://localhost:8000`
3. The Streamlit frontend will be available at: `http://localhost:8501`

## 📂 Project Structure
* `src/`: Core logic including data pipelines, model training scripts, and the FastAPI backend.
* `frontend/`: The Streamlit interactive user interface.
* `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and model prototyping.
* `data/`: Directory for raw and processed datasets (ignored by Git).
* `tests/`: Pytest files for ensuring pipeline and API robustness.

## ⚖️ Data License & Attribution
The dataset used to train this model is crowdsourced shot-by-shot professional tennis data provided by [The Tennis Abstract Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject), created by Jeff Sackmann.

This project is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
* **Attribution:** All underlying shot data belongs to the Match Charting Project.
* **Non-Commercial:** This is a strictly educational portfolio project and is not monetized.
* **ShareAlike:** The code and models in this repository are shared under the same CC BY-NC-SA 4.0 license.