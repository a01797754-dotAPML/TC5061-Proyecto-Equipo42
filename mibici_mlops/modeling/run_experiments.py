"""Orquesta los experimentos base de P1, P2 y P3 en un solo comando.

Corre el main() de cada pregunta, sigue con las demás si alguna falla y termina
con código de salida 1 si hubo errores. Por reproducibilidad, se niega a correr
con cambios sin commitear (salvo --permitir-cambios).

Uso desde la raíz del repo:  python -m mibici_mlops.modeling.run_experiments
"""

from dataclasses import dataclass
import importlib

from loguru import logger
import typer

from mibici_mlops.tracking import obtener_git_commit

app = typer.Typer()


@dataclass
class Experimento:
    """Qué correr: la pregunta y el módulo cuyo main() registra su run."""

    pregunta: str
    modulo: str


@dataclass
class Resultado:
    """Qué pasó al correr un experimento."""

    pregunta: str
    estado: str  # "ok", "no_existe" o "error"
    detalle: str  # run_id si salió bien; mensaje si no


EXPERIMENTOS = [
    Experimento("P1", "mibici_mlops.modeling.p1"),
    Experimento("P2", "mibici_mlops.modeling.p2"),
    Experimento("P3", "mibici_mlops.modeling.p3"),
]


class OrquestadorExperimentos:
    def __init__(self, experimentos: list[Experimento]):
        self.experimentos = experimentos

    def _correr_uno(self, exp: Experimento) -> Resultado:
        # 1: importar el módulo con importlib.import_module(exp.modulo)
        #         - si no existe (ModuleNotFoundError) → Resultado(..., "no_existe", ...)
        try:
            modulo = importlib.import_module(exp.modulo)
        except ModuleNotFoundError as e:
            if e.name != exp.modulo:
                return Resultado(
                    exp.pregunta,
                    "error",
                    f"{exp.modulo} requiere un módulo que no está instalado: {e.name}",
                )
            return Resultado(exp.pregunta, "no_existe", f"No existe el módulo {exp.modulo}")
        # 2: llamar modulo.main() y devolver Resultado(..., "ok", run_id)
        #         - si truena cualquier otra excepción → Resultado(..., "error", str(e))
        #           (pista: logger.exception registra el traceback completo)
        try:
            run_id = modulo.main()
        except Exception as e:  # noqa: BLE001 - un experimento que falla no debe detener a los demás
            logger.exception(f"{exp.pregunta} falló")
            return Resultado(exp.pregunta, "error", str(e))
        return Resultado(exp.pregunta, "ok", run_id)

    def ejecutar(self) -> list[Resultado]:
        # 3: correr cada experimento en orden y devolver la lista de resultados
        resultados = []
        for exp in self.experimentos:
            logger.info(f"Corriendo {exp.pregunta}...")
            resultados.append(self._correr_uno(exp))
        return resultados

    @staticmethod
    def resumen(resultados: list[Resultado]) -> None:
        # 4: imprimir una línea por pregunta: pregunta · estado · detalle
        logger.info("Resumen de experimentos:")
        for r in resultados:
            logger.info(f" {r.pregunta} * {r.estado} * {r.detalle}")


@app.command()
def main(
    permitir_cambios: bool = typer.Option(
        False, help="Correr aunque haya cambios sin commitear (el run quedará marcado)."
    ),
):
    # 5: con obtener_git_commit(), si hay cambios y no se permite → logger.error con
    #         la razón y raise typer.Exit(code=1)
    commit, con_cambios = obtener_git_commit()
    if con_cambios and not permitir_cambios:
        logger.error(
            "Hay cambios sin commitear: el run quedaría ligado a un commit que no "
            "corresponde al código ejecutado. Haz commit o usa --permitir-cambios."
        )
        raise typer.Exit(code=1)
    logger.info(f"Commit: {commit[:7]}")

    # 6: crear el orquestador, ejecutar, imprimir el resumen y, si algún resultado
    #         no es "ok", terminar con typer.Exit(code=1)
    orquestador = OrquestadorExperimentos(EXPERIMENTOS)
    resultados = orquestador.ejecutar()
    OrquestadorExperimentos.resumen(resultados)

    if any(r.estado == "error" for r in resultados):
        logger.warning("Hubo experimentos con error (ver resumen).")
        raise typer.Exit(code=1)
    if any(r.estado == "no_existe" for r in resultados):
        logger.warning("Algunas preguntas aún no tienen módulo (ver resumen).")
    else:
        logger.success("Todos los experimentos se registraron en MLflow.")


if __name__ == "__main__":
    app()
