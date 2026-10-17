"""P2 - Clasificación del riesgo de desabasto y saturación.

Ejecuta el baseline estacional y el modelo Random Forest,
registrando los resultados en MLflow.

Uso:
    python -m mibici_mlops.modeling.p2
"""

from mibici_mlops.entrenar_p2 import main as entrenar_p2


def main() -> str:
    """Ejecuta los experimentos P2 y devuelve el run ID del Random Forest."""
    return entrenar_p2()


if __name__ == "__main__":
    main()
