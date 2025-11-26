## Evidently 0.4.5
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset   # ton environnement interne
from evidently import ColumnMapping

REFERENCE_PATH = "monitoring/application_train.csv"
CURRENT_PATH = "monitoring/application_test.csv"

OUTPUT_REPORT = "monitoring/reports/data_drift_report.html"
OUTPUT_JSON = "monitoring/reports/data_drift_summary.json"

def load_data():
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    # Normalisation des valeurs manquantes
    for df in (reference, current):
        df.replace(["", " ", "NA", "N/A", "#N/A", "null", "NULL"], pd.NA, inplace=True)

    # Supprimer column TARGET si existante
    if "TARGET" in reference.columns:
        reference = reference.drop(columns=["TARGET"])
    if "TARGET" in current.columns:
        current = current.drop(columns=["TARGET"])

    # Imputation simple : on utilise la médiane du dataset de référence
    median_ref = reference.median(numeric_only=True)

    reference = reference.fillna(median_ref)
    current = current.fillna(median_ref)

    # On aligne les colonnes (sécurité)
    common_cols = reference.columns.intersection(current.columns)
    reference = reference[common_cols]
    current = current[common_cols]

    print("✔ Données chargées")
    print("Référence:", reference.shape)
    print("Courant:", current.shape)

    return reference, current

def generate_report(reference, current):

    report = Report(metrics=[DataDriftPreset()])

    # Indique que TARGET est la colonne cible (la retire du drift)
    column_mapping = ColumnMapping()
    column_mapping.target = None

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    # Sauvegarde JSON
    with open(OUTPUT_JSON, "w") as f:
        f.write(report.json())

    # Sauvegarde HTML
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report.html())


if __name__ == "__main__":
    import os
    os.makedirs("monitoring/reports", exist_ok=True)

    reference, current = load_data()
    generate_report(reference, current)

    print("\n🎉 Analyse de data drift terminée !")
