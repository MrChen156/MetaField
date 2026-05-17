"""Material fitting and material-database maintenance helpers."""

from .drude_lorentz_fit import (
    DrudeLorentzFitConfig,
    fit_and_save_material,
    get_or_create_material_code,
    load_material_mappings,
    load_optical_table,
    save_fit_result,
    save_material_mappings,
    upsert_material_record,
)

__all__ = [
    "DrudeLorentzFitConfig",
    "fit_and_save_material",
    "get_or_create_material_code",
    "load_material_mappings",
    "load_optical_table",
    "save_fit_result",
    "save_material_mappings",
    "upsert_material_record",
]
