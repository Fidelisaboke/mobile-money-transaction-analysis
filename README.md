# Mobile Money Transaction Analysis with ML
## Project Overview
This project explores mobile money transaction patterns using machine learning.  
It includes data preprocessing, feature engineering, modeling, and evaluation using a synthetic dataset resembling real-world financial transactions.

## Table of Contents
- [Installation and Setup](#installation-and-setup)
    - [Pre-requisites](#pre-requisites)
    - [Setup Instructions](#setup-instructions)
- [Basic Usage](#basic-usage)
- [Dataset](#dataset)

## Installation and Setup
### Pre-requisites
- Python 3.12 and above
- [Synthetic Mobile Money Transaction Dataset (Azamuke, 2024)](#dataset)


### Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/Fidelisaboke/mobile-money-transaction-analysis.git
cd mobile-money-transaction-analysis
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create the `.env` file by copying `.env.example`:
```bash
cp .env.example .env
```
- Ensure you update the variables in the `.env` files as required.

## Basic Usage
- Access the Jupyter Notebooks in the `notebooks/` directory for an interactive outline of the model 
development process.

- The `pipeline/` folder contains the Python scripts for preprocessing the data, and building the 
model. For example, you can run `preprocessing.py` using the `python` command:
```bash
cd pipeline
python preprocessing.py
```

## Dataset
This project uses a synthetic mobile money transaction dataset:

> **Azamuke, Denish (2024)**, “[Synthetic Mobile Money Transaction Dataset](https://doi.org/10.17632/zhj366m53p.2)”, *Mendeley Data*, V2

To set up the dataset:

1. Download the dataset from [this link](https://data.mendeley.com/datasets/zhj366m53p/2)
2. Extract and move `synthetic_mobile_money_transaction_dataset.csv` into the `data/` directory
3. Rename the file to `transactions.csv` for consistency with the codebase
