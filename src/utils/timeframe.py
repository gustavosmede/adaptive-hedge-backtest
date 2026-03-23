from __future__ import annotations

ANNUALIZATION = {
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
}


def annualization_factor(interval: str) -> int:
    if interval not in ANNUALIZATION:
        raise ValueError(f"Intervalo não suportado para anualização: {interval}")
    return ANNUALIZATION[interval]
