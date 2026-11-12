import argparse
import logging
import sys
from pathlib import Path

from kkbox_churn.config import (CUTOFF, PREDICTIONS_PATH, TARGET,
                                TRAIN_PATH, VALIDATION_PATH)
from kkbox_churn.logging_config import setup_logging

logger = logging.getLogger("kkbox_churn.cli")


def cmd_prepare(args):
    from kkbox_churn.data.make_dataset import prepare_data

    train, val = prepare_data(cutoff=args.cutoff)
    logger.info("Train: %s filas | Validación: %s filas", f"{len(train):,}", f"{len(val):,}")


def cmd_train(args):
    from kkbox_churn.data.load import load_dataset
    from kkbox_churn.models.train import train_model, save_model

    train = load_dataset(args.input)
    logger.info("Entrenando con %s filas...", f"{len(train):,}")
    model = train_model(train)
    path = save_model(model)
    logger.info("Modelo guardado en %s (umbral %.4f)", path, model.best_threshold_)


def cmd_predict(args):
    from kkbox_churn.data.load import load_dataset, save_dataset
    from kkbox_churn.models.predict import load_model, predict
    from kkbox_churn.models.evaluate import evaluate

    df = load_dataset(args.input)
    model = load_model()
    preds = predict(model, df)

    save_dataset(preds, args.output)
    logger.info("Predicciones guardadas en %s (%.1f %% marcados como churn)",
                args.output, 100 * preds["churn_pred"].mean())

    # Si los datos traen el target, se evalúa el modelo
    if TARGET in df.columns:
        metrics = evaluate(df[TARGET], preds["churn_proba"], preds["churn_pred"])
        for name, value in metrics.items():
            logger.info("%-22s %.4f", name, value)


def build_parser():
    parser = argparse.ArgumentParser(prog="kkbox-churn",
                                     description="Pipeline de predicción de churn de KKBox")
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra logs de depuración")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Construye features y separa train/validación")
    p.add_argument("--cutoff", default=CUTOFF, help=f"Fecha de corte (por defecto {CUTOFF})")
    p.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="Entrena el modelo y lo guarda")
    t.add_argument("--input", type=Path, default=TRAIN_PATH)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Genera predicciones (y métricas si hay target)")
    pr.add_argument("--input", type=Path, default=VALIDATION_PATH)
    pr.add_argument("--output", type=Path, default=PREDICTIONS_PATH)
    pr.set_defaults(func=cmd_predict)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        logger.info("Inicio: %s", args.command)
        args.func(args)
        logger.info("Fin: %s", args.command)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))          # error esperado: mensaje claro, sin traceback
        sys.exit(1)
    except Exception:
        logger.exception("Error inesperado")   # error no previsto: traceback completo al log
        sys.exit(1)


if __name__ == "__main__":
    main()