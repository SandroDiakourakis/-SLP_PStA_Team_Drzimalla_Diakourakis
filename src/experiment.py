import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pickle
from typing import Dict, Any, List
import pandas as pd


class ExperimentManager:
    """Verwaltet ML-Experimente und speichert Ergebnisse strukturiert"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.logs_dir = self.results_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.models_dir = self.results_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.db_path = self.results_dir / "experiments.db"
        self._init_database()

    def _init_database(self):
        """Initialisiert die SQLite-Datenbank"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS experiments
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           experiment_id
                           TEXT
                           UNIQUE,
                           timestamp
                           DATETIME,
                           model_name
                           TEXT,
                           hyperparameters
                           TEXT,
                           accuracy
                           REAL,
                           precision
                           REAL,
                           recall
                           REAL,
                           f1_score
                           REAL,
                           training_time
                           REAL,
                           dataset_name
                           TEXT,
                           dataset_size
                           INTEGER,
                           notes
                           TEXT,
                           model_path
                           TEXT
                       )
                       ''')

        conn.commit()
        conn.close()

    def save_experiment(self,
                        model_name: str,
                        hyperparameters: Dict[str, Any],
                        metrics: Dict[str, float],
                        training_time: float,
                        dataset_info: Dict[str, Any],
                        model_object: Any = None,
                        additional_data: Dict[str, Any] = None,
                        notes: str = "") -> str:
        """
        Speichert ein Experiment in der Datenbank und als JSON

        Returns:
            experiment_id: Eindeutige ID des Experiments
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        # In SQLite speichern
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        model_path = None
        if model_object is not None:
            model_path = str(self.models_dir / f"{experiment_id}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)

        cursor.execute('''
                       INSERT INTO experiments (experiment_id, timestamp, model_name, hyperparameters,
                                                accuracy, precision, recall, f1_score, training_time,
                                                dataset_name, dataset_size, notes, model_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           experiment_id,
                           timestamp,
                           model_name,
                           json.dumps(hyperparameters),
                           metrics.get('accuracy'),
                           metrics.get('precision'),
                           metrics.get('recall'),
                           metrics.get('f1_score'),
                           training_time,
                           dataset_info.get('name'),
                           dataset_info.get('size'),
                           notes,
                           model_path
                       ))

        conn.commit()
        conn.close()

        # Detaillierte Daten als JSON speichern
        log_data = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "training_time": training_time,
            "dataset_info": dataset_info,
            "notes": notes
        }

        if additional_data:
            log_data["additional_data"] = additional_data

        log_file = self.logs_dir / f"{experiment_id}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Experiment gespeichert: {experiment_id}")
        return experiment_id

    def get_results(self,
                    model_name: str = None,
                    top_n: int = None,
                    sort_by: str = 'f1_score') -> pd.DataFrame:
        """
        Lädt Experiment-Ergebnisse aus der Datenbank

        Args:
            model_name: Filtere nach Modellnamen
            top_n: Zeige nur die besten N Ergebnisse
            sort_by: Sortiere nach dieser Metrik
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM experiments"
        if model_name:
            query += f" WHERE model_name = '{model_name}'"

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            return df

        # Sortieren
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)

        # Top N
        if top_n:
            df = df.head(top_n)

        return df

    def get_best_model(self, metric: str = 'f1_score', model_name: str = None):
        """
        Lädt das beste Modell basierend auf einer Metrik

        Returns:
            (model_object, experiment_info)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f'''
            SELECT * FROM experiments 
            WHERE {metric} IS NOT NULL
        '''

        if model_name:
            query += f" AND model_name = '{model_name}'"

        query += f" ORDER BY {metric} DESC LIMIT 1"

        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()

        if not result:
            return None, None

        # Modell laden
        model_path = result[-1]  # Letzter Eintrag ist model_path
        if model_path and Path(model_path).exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Experiment-Info als Dictionary
            columns = ['id', 'experiment_id', 'timestamp', 'model_name',
                       'hyperparameters', 'accuracy', 'precision', 'recall',
                       'f1_score', 'training_time', 'dataset_name',
                       'dataset_size', 'notes', 'model_path']

            exp_info = dict(zip(columns, result))
            exp_info['hyperparameters'] = json.loads(exp_info['hyperparameters'])

            return model, exp_info

        return None, None

    def compare_models(self) -> pd.DataFrame:
        """Vergleicht alle Modelle und zeigt durchschnittliche Performance"""
        conn = sqlite3.connect(self.db_path)

        query = '''
                SELECT model_name, \
                       COUNT(*)           as num_experiments, \
                       AVG(accuracy)      as avg_accuracy, \
                       AVG(precision)     as avg_precision, \
                       AVG(recall)        as avg_recall, \
                       AVG(f1_score)      as avg_f1_score, \
                       MAX(f1_score)      as best_f1_score, \
                       AVG(training_time) as avg_training_time
                FROM experiments
                GROUP BY model_name
                ORDER BY avg_f1_score DESC \
                '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def load_experiment_details(self, experiment_id: str) -> Dict:
        """Lädt detaillierte Informationen eines Experiments aus dem JSON-Log"""
        log_file = self.logs_dir / f"{experiment_id}.json"

        if not log_file.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} nicht gefunden")

        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)