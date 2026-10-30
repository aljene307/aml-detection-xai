"""Mesure du temps et de la mémoire d'un traitement."""

import time
from contextlib import contextmanager

import psutil


def available_ram_mb() -> float:
    return psutil.virtual_memory().available / 2**20


def peak_memory_mb() -> float:
    """Pic de mémoire du processus depuis son démarrage (en Mo).

    Sous Windows, le système conserve ce pic (peak_wset). Ailleurs, on se
    rabat sur la mémoire actuelle : la mesure doit alors être faite dans un
    processus dédié, juste après le traitement.
    """
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20


@contextmanager
def measure(result: dict):
    """Remplit `result` avec la durée (s) et le pic de mémoire (Mo)."""
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duree_s"] = round(time.perf_counter() - start, 2)
        result["pic_memoire_mo"] = round(peak_memory_mb())
