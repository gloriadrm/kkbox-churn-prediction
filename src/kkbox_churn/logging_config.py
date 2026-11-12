import logging

from kkbox_churn.config import LOGS_DIR


def setup_logging(level=logging.INFO):
    """Configura los logs para mostrarse en consola y guardarse en archivo."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "kkbox_churn.log", encoding="utf-8"),
        ],
        force=True,
    )