# Mobile Money Transaction Analysis
## Project Overview
An ML analysis of mobile money transactions patterns.

## Installation and Setup
### Pre-requisites
- Python 3.12 and above
- [Synthetic Mobile Money Transaction Dataset (Azamuke, 2024)](https://data.mendeley.com/datasets/zhj366m53p/2):
    - Visit [this link](https://data.mendeley.com/datasets/zhj366m53p/2) and download
     `synthetic_mobile_money_transaction_dataset.csv`. Add it to the `data/` directory, and rename
     it to `transactions.csv` for easy access.

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

## Dataset Reference
We acknowledge the creators of the dataset used in this project:

Azamuke, Denish (2024), “[Synthetic Mobile Money Transaction Dataset](https://doi.org/10.17632/zhj366m53p.2)”, *Mendeley Data*, V2, doi: [10.17632/zhj366m53p.2](https://doi.org/10.17632/zhj366m53p.2)


