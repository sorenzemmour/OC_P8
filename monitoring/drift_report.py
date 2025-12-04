## Evidently 0.4.5 (0.3.x ?)
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset   # ton environnement interne
from evidently import ColumnMapping

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.schemas.input_schema import FEATURE_ORDER

REFERENCE_PATH = "monitoring/application_train.csv"
CURRENT_PATH = "monitoring/application_test.csv"

OUTPUT_REPORT = "monitoring/reports/data_drift_report.html"
OUTPUT_JSON = "monitoring/reports/data_drift_summary.json"

def robust_read_csv_path(path, expected_cols=121):
    """
    Lecture robuste d'un CSV Home Credit depuis un fichier :
    - détecte les lignes compactées
    - tente de les réparer
    - renvoie df + invalid_rows
    """
    import csv

    valid_rows = []
    invalid_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.read().splitlines()

    for idx, line in enumerate(raw_lines):

        row = next(csv.reader([line]))

        # Ligne normale
        if len(row) == expected_cols:
            valid_rows.append(row)
            continue

        # Tentative de réparation brute
        repair = line.split(",")

        if len(repair) == expected_cols:
            valid_rows.append(repair)
            continue

        # Ligne irrécupérable
        invalid_rows.append({
            "line_number": idx,
            "raw_line": line,
            "reason": f"Ligne détectée avec {len(row)} colonnes"
        })

    if not valid_rows:
        raise ValueError(f"Aucune ligne valide dans {path}")

    df = pd.DataFrame(valid_rows[1:], columns=valid_rows[0])
    return df, invalid_rows


def load_data():

    print("📥 Lecture robuste du dataset de référence...")
    reference, bad_ref = robust_read_csv_path(REFERENCE_PATH, expected_cols=122)

    print("📥 Lecture robuste du dataset courant...")
    current, bad_cur = robust_read_csv_path(CURRENT_PATH, expected_cols=121)

    print(f"⚠️ Lignes corrompues retirées — référence : {len(bad_ref)}")
    print(f"⚠️ Lignes corrompues retirées — courant   : {len(bad_cur)}")

    # Nettoyage NA
    for df in (reference, current):
        df.replace(["", " ", "NA", "N/A", "#N/A", "null", "NULL"], pd.NA, inplace=True)

    # Imputation cohérente
    median_ref = reference.median(numeric_only=True)
    reference = reference.fillna(median_ref)
    current = current.fillna(median_ref)

    # Alignement strict
    cols = reference.columns.intersection(current.columns)
    reference = reference[cols]
    current = current[cols]

    # Convertir en numérique
    reference = reference.apply(pd.to_numeric, errors="coerce")
    current = current.apply(pd.to_numeric, errors="coerce")

    # Ne garder que les features du modèle
    reference = reference[FEATURE_ORDER]
    current   = current[FEATURE_ORDER]

    print("✔ Données prêtes")
    print("Reference:", reference.shape)
    print("Current:", current.shape)

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
        report.save_json(OUTPUT_JSON)

    # Sauvegarde HTML
    with open(OUTPUT_REPORT, "w") as f:
        report.save_html(OUTPUT_REPORT)


if __name__ == "__main__":
    import os
    os.makedirs("monitoring/reports", exist_ok=True)

    reference, current = load_data()
    generate_report(reference, current)

    print("\n🎉 Analyse de data drift terminée !")
