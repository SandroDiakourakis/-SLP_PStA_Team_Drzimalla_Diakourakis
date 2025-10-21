#!/usr/bin/env python3
"""
Main CLI entry point for SAND Task 1 pipeline
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sandcli import prep, split, featurize, train, evaluate, predict, ensemble

log = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Configure logging"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Dynamischer config_path
_project_root = Path(__file__).parent.parent
_config_path = str(_project_root / "conf")


@hydra.main(version_base=None, config_path=_config_path, config_name="config")
def cli(cfg: DictConfig) -> None:
    """
    Main CLI dispatcher

    Usage:
        python -m sandcli.main command=prep
        python -m sandcli.main command=split
        python -m sandcli.main command=featurize
        python -m sandcli.main command=train
        python -m sandcli.main command=evaluate    # ← CHANGED
        python -m sandcli.main command=predict
        python -m sandcli.main command=ensemble
    """
    setup_logging(cfg.get('verbose', True))

    command = cfg.get('command', None)

    if command is None:
        log.error("No command specified. Use command=<prep|split|featurize|train|evaluate|predict|ensemble>")
        sys.exit(1)

    log.info(f"Running command: {command}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Dispatch to appropriate module
    commands = {
        'prep': prep.run,
        'split': split.run,
        'featurize': featurize.run,
        'train': train.run,
        'evaluate': evaluate.run,
        'predict': predict.run,
        'ensemble': ensemble.run,
    }

    if command not in commands:
        log.error(f"Unknown command: {command}")
        log.error(f"Available commands: {list(commands.keys())}")
        sys.exit(1)

    # Execute command
    try:
        commands[command](cfg)
        log.info(f"Command '{command}' completed successfully!")
    except Exception as e:
        log.error(f"Command '{command}' failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
