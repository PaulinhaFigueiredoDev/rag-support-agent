import pytest
from emporio_assistant.services import data_loader
from emporio_assistant.services.data_loader import load_table

# Testa o carregamento de produtos
def test_load_products_shape():
    products = load_table("products")
    assert products.shape == (65, 9)

# Testa nome não permitido
def test_reject_unknown_table():
    with pytest.raises(ValueError, match="Tabela não permitida"):
        load_table("../secrets")

# Testa todas as contagens sem repetir código
@pytest.mark.parametrize(
    "name, expected_rows",
    [
        ("categories", 9),
        ("products", 65),
        ("customers", 50),
        ("orders", 20),
        ("order_items", 22),
        ("promotions", 25),
    ],
)
def test_table_row_counts(name, expected_rows):
    assert len(load_table(name)) == expected_rows

# Testa uma tabela permitida cujo arquivo CSV não existe
def test_missing_csv_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "CSV_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="CSV não encontrado"):
        load_table("products")
