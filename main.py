#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Importiere deine Module
from src.experiment import ExperimentManager
from src.models import ModelTrainer
from src.preprocessing import load_and_extract_features, preprocess_data


def load_data(data_path: str = None, task: str = 'task1', phonation: str = 'A'):
    """
    Lädt ALS Audio-Daten und extrahiert Features

    Parameters:
    -----------
    data_path : str
        Pfad zum data Ordner
    task : str
        'task1' oder 'task2'
    phonation : str
        Phonation-Typ ('A', 'E', 'I', 'O', 'U')

    Returns:
    --------
    tuple : (X, y) - Features und Labels
    """
    if not data_path:
        raise ValueError("❌ Bitte gib einen Pfad zu den Daten an (--data)")

    print(f"\n{'=' * 60}")
    print(f"📂 Feature-Extraktion")
    print(f"{'=' * 60}")

    # Features extrahieren
    X, y = load_and_extract_features(
        data_path=data_path,
        task=task,
        phonation_type=phonation,
        n_mfcc=13,
        verbose=True
    )

    # Preprocessing (Skalierung)
    print(f"\n🔧 Preprocessing (StandardScaler)...")
    X, y, scaler = preprocess_data(X, y, fit_scaler=True)
    print(f"✅ Features skaliert: {X.shape}")

    return X, y


def train_command(args):
    """Trainiert ein einzelnes Modell"""
    print("\n🚀 Starte Training...")

    # Daten laden
    X, y = load_data(args.data, args.task, args.phonation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modell trainieren
    trainer = ModelTrainer(args.grid)

    hyperparams = None
    if args.params:
        hyperparams = json.loads(args.params)

    model, metrics, train_time = trainer.train_model(
        args.model, X_train, y_train, X_test, y_test, hyperparams
    )

    # Ergebnisse speichern
    exp_manager = ExperimentManager()
    exp_id = exp_manager.save_experiment(
        model_name=args.model,
        hyperparameters=hyperparams or {},
        metrics=metrics,
        training_time=train_time,
        dataset_info={
            'name': f'{args.task}_phonation{args.phonation}',
            'size': len(X)
        },
        model_object=model,
        additional_data={'confusion_matrix': metrics['confusion_matrix']},
        notes=args.notes or f"Task: {args.task}, Phonation: {args.phonation}"
    )

    print(f"\n📊 Ergebnisse:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   Training Zeit: {train_time:.2f}s")
    print(f"\n💾 Experiment ID: {exp_id}")


def grid_search_command(args):
    """Führt Grid Search für ein Modell durch"""
    print("\n🔍 Starte Grid Search...")

    # Daten laden
    X, y = load_data(args.data, args.task, args.phonation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Grid Search
    trainer = ModelTrainer(args.grid)
    model, best_params, metrics, train_time = trainer.grid_search(
        args.model, X_train, y_train, X_test, y_test, cv=args.cv
    )

    # Ergebnisse speichern
    exp_manager = ExperimentManager()
    exp_id = exp_manager.save_experiment(
        model_name=args.model,
        hyperparameters=best_params,
        metrics=metrics,
        training_time=train_time,
        dataset_info={
            'name': f'{args.task}_phonation{args.phonation}',
            'size': len(X)
        },
        model_object=model,
        additional_data={
            'confusion_matrix': metrics['confusion_matrix'],
            'cv_score': metrics.get('cv_best_score')
        },
        notes=f"Grid Search (CV={args.cv}), Task: {args.task}, Phonation: {args.phonation}"
    )

    print(f"\n🏆 Beste Ergebnisse:")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"   CV Best Score: {metrics.get('cv_best_score', 'N/A'):.4f}")
    print(f"   Best Params: {json.dumps(best_params, indent=2)}")
    print(f"\n💾 Experiment ID: {exp_id}")


def compare_all_command(args):
    """Trainiert und vergleicht alle Modelle"""
    print("\n🏁 Starte Vergleich aller Modelle...")

    # Daten laden
    X, y = load_data(args.data, args.task, args.phonation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Alle Modelle trainieren
    trainer = ModelTrainer(args.grid)
    results = trainer.train_all_models(
        X_train, y_train, X_test, y_test,
        use_grid_search=args.use_grid_search
    )

    # Ergebnisse speichern
    exp_manager = ExperimentManager()

    for model_name, result in results.items():
        if 'error' in result:
            print(f"⚠️  {model_name}: Fehler aufgetreten")
            continue

        exp_manager.save_experiment(
            model_name=model_name,
            hyperparameters=result['hyperparameters'],
            metrics=result['metrics'],
            training_time=result['training_time'],
            dataset_info={
                'name': f'{args.task}_phonation{args.phonation}',
                'size': len(X)
            },
            model_object=result['model'],
            additional_data={'confusion_matrix': result['metrics']['confusion_matrix']},
            notes=f"Batch comparison, Task: {args.task}, Phonation: {args.phonation}"
        )

    # Vergleich anzeigen
    comparison = exp_manager.compare_models()
    print("\n📊 Modell-Vergleich:")
    print(comparison.to_string(index=False))


def results_command(args):
    """Zeigt gespeicherte Ergebnisse an"""
    exp_manager = ExperimentManager()

    df = exp_manager.get_results(
        model_name=args.model,
        top_n=args.top,
        sort_by=args.sort_by
    )

    if len(df) == 0:
        print("❌ Keine Experimente gefunden")
        return

    # Wichtige Spalten anzeigen
    columns = ['experiment_id', 'model_name', 'accuracy', 'precision',
               'recall', 'f1_score', 'training_time', 'timestamp']

    display_df = df[columns].copy()

    # Formatierung
    for col in ['accuracy', 'precision', 'recall', 'f1_score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if x else "N/A")

    display_df['training_time'] = display_df['training_time'].apply(
        lambda x: f"{x:.2f}s" if x else "N/A"
    )

    print("\n📊 Experiment-Ergebnisse:")
    print(display_df.to_string(index=False))

    if args.details and len(df) > 0:
        exp_id = df.iloc[0]['experiment_id']
        details = exp_manager.load_experiment_details(exp_id)
        print(f"\n🔍 Details für {exp_id}:")
        print(json.dumps(details, indent=2, default=str))


def best_command(args):
    """Lädt das beste Modell"""
    exp_manager = ExperimentManager()

    model, info = exp_manager.get_best_model(
        metric=args.metric,
        model_name=args.model
    )

    if model is None:
        print("❌ Kein Modell gefunden")
        return

    print(f"\n🏆 Bestes Modell:")
    print(f"   Experiment ID: {info['experiment_id']}")
    print(f"   Modell: {info['model_name']}")
    print(f"   {args.metric.upper()}: {info[args.metric]:.4f}")
    print(f"   Hyperparameter: {json.dumps(info['hyperparameters'], indent=2)}")
    print(f"   Timestamp: {info['timestamp']}")
    print(f"   Dataset: {info.get('dataset_name', 'N/A')}")


def compare_command(args):
    """Vergleicht durchschnittliche Performance aller Modelle"""
    exp_manager = ExperimentManager()
    comparison = exp_manager.compare_models()

    if len(comparison) == 0:
        print("❌ Keine Experimente gefunden")
        return

    print("\n📊 Modell-Vergleich (Durchschnitte):")
    print(comparison.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="ALS Audio Klassifikation - ML Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Ein Modell trainieren
  python main.py train --model SVC --data data --task task1 --phonation A

  # Grid Search durchführen
  python main.py grid-search --model RandomForestClassifier --data data --task task1 --phonation A

  # Alle Modelle vergleichen
  python main.py compare-all --data data --task task1 --phonation A --use-grid-search

  # Ergebnisse anzeigen
  python main.py results --top 10 --sort-by f1_score

  # Bestes Modell anzeigen
  python main.py best --metric f1_score
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Befehle')

    # Gemeinsame Argumente für alle Training-Commands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data', required=True, help='Pfad zum data Ordner')
    parent_parser.add_argument('--task', default='task1', choices=['task1', 'task2'],
                               help='Task Auswahl (default: task1)')
    parent_parser.add_argument('--phonation', default='A', choices=['A', 'E', 'I', 'O', 'U'],
                               help='Phonation-Typ (default: A)')
    parent_parser.add_argument('--grid', default='grids/model_grids.json',
                               help='Pfad zur Grid-Datei')

    # Train Command
    train_parser = subparsers.add_parser('train', parents=[parent_parser],
                                         help='Trainiert ein einzelnes Modell')
    train_parser.add_argument('--model', required=True,
                              help='Modellname (SVC, LogisticRegression, RandomForestClassifier)')
    train_parser.add_argument('--params', help='Hyperparameter als JSON-String')
    train_parser.add_argument('--notes', help='Notizen zum Experiment')

    # Grid Search Command
    grid_parser = subparsers.add_parser('grid-search', parents=[parent_parser],
                                        help='Grid Search für ein Modell')
    grid_parser.add_argument('--model', required=True, help='Modellname')
    grid_parser.add_argument('--cv', type=int, default=5, help='Cross-Validation Folds (default: 5)')

    # Compare All Command
    compare_all_parser = subparsers.add_parser('compare-all', parents=[parent_parser],
                                               help='Vergleicht alle Modelle')
    compare_all_parser.add_argument('--use-grid-search', action='store_true',
                                    help='Nutze Grid Search für alle Modelle')

    # Results Command
    results_parser = subparsers.add_parser('results', help='Zeigt gespeicherte Ergebnisse')
    results_parser.add_argument('--model', help='Filter nach Modellname')
    results_parser.add_argument('--top', type=int, help='Zeige Top N Ergebnisse')
    results_parser.add_argument('--sort-by', default='f1_score',
                                help='Sortiere nach Metrik (default: f1_score)')
    results_parser.add_argument('--details', action='store_true', help='Zeige Details')

    # Best Command
    best_parser = subparsers.add_parser('best', help='Zeigt bestes Modell')
    best_parser.add_argument('--metric', default='f1_score',
                             help='Metrik für "best" (default: f1_score)')
    best_parser.add_argument('--model', help='Filter nach Modellname')

    # Compare Command
    compare_parser = subparsers.add_parser('compare', help='Vergleicht Modell-Performance')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Command ausführen
    commands = {
        'train': train_command,
        'grid-search': grid_search_command,
        'compare-all': compare_all_command,
        'results': results_command,
        'best': best_command,
        'compare': compare_command
    }

    try:
        commands[args.command](args)
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()