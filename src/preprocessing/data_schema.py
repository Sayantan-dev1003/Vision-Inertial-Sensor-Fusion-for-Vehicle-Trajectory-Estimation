from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ENUOrigin:
    """Geodetic reference origin for local East-North-Up projection."""

    lat0_deg: float
    lon0_deg: float
    alt0_m: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SensorSchema:
    """Encapsulates raw CSV layout and active sensor channel definitions."""

    RAW_COLUMNS: Tuple[str, ...] = (
        "gps_lat",
        "gps_lon",
        "gps_alt",
        "acc_x",
        "acc_y",
        "acc_z",
        "lin_acc_x",
        "lin_acc_y",
        "lin_acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
        "sensor_type",
        "timestamp_ms",
        "datetime_str",
    )

    ACTIVE_SENSOR_COLUMNS: Dict[str, List[str]] = {
        "gps": ["gps_lat", "gps_lon", "gps_alt"],
        "accr": ["acc_x", "acc_y", "acc_z"],
        "gyro": ["gyro_x", "gyro_y", "gyro_z"],
    }

    IGNORED_SENSOR_TYPES: Tuple[str, ...] = ("lin_accr", "mag_field")

    @classmethod
    def active_sensor_types(cls) -> Tuple[str, ...]:
        return tuple(cls.ACTIVE_SENSOR_COLUMNS.keys())

    @classmethod
    def all_known_sensor_types(cls) -> set[str]:
        return set(cls.ACTIVE_SENSOR_COLUMNS.keys()) | set(cls.IGNORED_SENSOR_TYPES)

    @classmethod
    def get_active_columns(cls, sensor_type: str) -> List[str]:
        if sensor_type not in cls.ACTIVE_SENSOR_COLUMNS:
            raise KeyError(f"Unsupported or ignored sensor_type: '{sensor_type}'")
        return list(cls.ACTIVE_SENSOR_COLUMNS[sensor_type])