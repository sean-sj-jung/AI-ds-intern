# Gemini Project Context: Heart Disease Prediction

## Project Overview
This project aims to build a machine learning model to predict the presence of heart disease based on patient data. The dataset is sourced from a Tabular Playground Series competition. The goal is to train a model on `data/train.csv` and generate predictions for `data/test.csv`.

## Directory Structure

### Key Files & Directories
- **`main.py`**: The main entry point for the application. Currently contains a basic "Hello World" placeholder but is intended to orchestrate the data loading, model training, and prediction pipeline.
- **`data/`**: Directory containing the datasets.
    - `train.csv`: Training data with features and the target label `Heart Disease`. This data is generated synthetically through a deep learning model.
    - `test.csv`: Test data for which predictions need to be generated. This data is generated synthetically through a deep learning model.
    - `sample_submission.csv`: A template file showing the required submission format.
    - `Original_Heart_Disease_Prediction.csv`: This is the source data behind the deep learning model that used to generate train.csv and test.csv. It may have difference in distribution and it is not strictly necessary to use this file for model training but it may improve accuracy.
- **`pyproject.toml`**: Project configuration file defining dependencies and metadata. This project uses `uv` for package management.
- **`uv.lock`**: Lock file ensuring reproducible dependency versions.

## Data Schema
The training data (`data/train.csv`) includes the following columns:
- **Target Variable:** `Heart Disease` (Values: `Presence`, `Absence`)
- **Features:**
    - `id`: Unique identifier
    - `Age`: Age of the patient
    - `Sex`: Gender of the patient (e.g., 1 = male, 0 = female)
    - `Chest pain type`: Type of chest pain experienced
    - `BP`: Blood pressure
    - `Cholesterol`: Cholesterol levels
    - `FBS over 120`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    - `EKG results`: Resting electrocardiographic results
    - `Max HR`: Maximum heart rate achieved
    - `Exercise angina`: Exercise-induced angina (1 = yes; 0 = no)
    - `ST depression`: ST depression induced by exercise relative to rest
    - `Slope of ST`: The slope of the peak exercise ST segment
    - `Number of vessels fluro`: Number of major vessels (0-3) colored by flourosopy
    - `Thallium`: Thallium stress test result

## Development Workflow

### Prerequisites
- Python >= 3.11
- `uv` package manager

### Setup
To install the project dependencies, run:
```bash
uv sync
```

### Running the Project
To execute the main script:
```bash
uv run main.py
```

To run the TabPFN inference:
```bash
uv run python tabpfn_inference.py
```

To run the PyTabKit inference:
```bash
uv run python pytabkit_inference.py
```

## Progress & Implementation Details

### Completed Tasks
- **Repository Setup**: Initialized Git, resolved `.gitignore` conflicts, and pushed to [GitHub](https://github.com/sean-sj-jung/AI-ds-intern.git).
- **TabPFN Integration**: Implemented a classification pipeline using TabPFN (documented in `tabpfn_inference.py`). Resulted in ~0.868 accuracy.
- **PyTabKit Integration**: Implemented `RealMLP_TD_Classifier` from PyTabKit for potentially better performance.
- **Prediction Generation**: Generated `submission.csv` (TabPFN) and `submission_pytabkit.csv` (PyTabKit).

### TabPFN Implementation Notes
... (previous notes)

### PyTabKit Implementation Notes
- **Model Choice**: `RealMLP_TD_Classifier` (RealMLP with Tuned Defaults).
- **Data Strategy**: Used 30,000 training samples + original dataset.
- **Performance**: Achieved training accuracy of 0.8810. Training took ~4 minutes on Apple Silicon GPU (MPS).
- **Resources**: Utilized `mps` device for acceleration.

## Future Tasks
- Refactor `main.py` to integrate the winning model strategy.
- Perform meaningful exploratory data analysis (EDA).
- Experiment with increasing TabPFN training context (up to 10,000 for V2) if more compute is available.
- Compare TabPFN performance with other Gradient Boosted Decision Tree (GBDT) models like XGBoost or LightGBM.
