"""Punto de entrada para los experimentos de MiBici."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Ejecutar experimentos MLOps de MiBici")

    parser.add_argument(
        "--pregunta",
        choices=["P1", "P2"],
        required=True,
        help="Pregunta de investigación que se ejecutará",
    )

    args = parser.parse_args()

    if args.pregunta == "P1":
        from mibici_mlops.modeling.p1 import main as ejecutar_p1

        run_id = ejecutar_p1()

    elif args.pregunta == "P2":
        from mibici_mlops.modeling.p2 import main as ejecutar_p2

        run_id = ejecutar_p2()

    print(f"Experimento {args.pregunta} completado.")
    print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()
