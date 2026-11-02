from pathlib import Path
import pandas as pd

# Localiza a pasta dos CSVs a partir da raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = PROJECT_ROOT / "data" / "csv"

# Define quais tabelas podem ser carregadas
TABLES = {
    "categories",
    "products",
    "customers",
    "orders",
    "order_items",
    "promotions",
}


def load_table(name: str) -> pd.DataFrame:
    """Carrega um dos CSVs permitidos e retorna um DataFrame."""
    # Rejeita nomes que não estão na lista de tabelas permitidas
    if name not in TABLES:
        raise ValueError(f"Tabela não permitida: {name}")

    # Verifica se o arquivo CSV da tabela existe
    path = CSV_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    # Lê o CSV e retorna os dados como DataFrame
    return pd.read_csv(path)
