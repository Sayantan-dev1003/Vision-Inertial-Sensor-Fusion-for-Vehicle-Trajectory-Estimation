from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass(frozen=True)
class IMUMountingConfig:
    forward_axis: List[float]
    up_axis_hint: List[float]
    yaw_sign: float


@dataclass
class RecordingConfig:
    recording_id: str
    video_path: Path
    sensor_path: Path
    interim_dir: Path
    processed_dir: Path
    output_dir: Path
    csv_has_header: bool
    timestamp_unit: str
    imu_mounting: IMUMountingConfig
    video_to_sensor_offset_s: float
    camera_calibrated: bool

    def ensure_directories(self) -> None:
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class RecordingManifest:
    """Loads project defaults and recording-specific manifests from YAML."""

    def __init__(
        self,
        default_yaml: str | Path = "configs/default.yaml",
        recordings_yaml: str | Path = "configs/recordings.yaml",
    ) -> None:
        self.default_yaml = Path(default_yaml)
        self.recordings_yaml = Path(recordings_yaml)
        self._defaults: Dict[str, Any] = self._read_yaml(self.default_yaml)
        self._recordings: Dict[str, Any] = self._read_yaml(self.recordings_yaml)

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def preprocessing_params(self) -> Dict[str, Any]:
        return dict(self._defaults.get("preprocessing", {}))

    @property
    def active_sensors(self) -> List[str]:
        return list(
            self._defaults.get("sensors", {}).get(
                "active_types", ["accr", "gyro", "gps"]
            )
        )

    def get_recording(self, recording_id: str) -> RecordingConfig:
        recs = self._recordings.get("recordings", {})
        if recording_id not in recs:
            raise KeyError(f"Recording '{recording_id}' not found in {self.recordings_yaml}")

        rec = recs[recording_id]
        imu_raw = rec["imu_mounting"]
        mounting = IMUMountingConfig(
            forward_axis=[float(v) for v in imu_raw["forward_axis"]],
            up_axis_hint=[float(v) for v in imu_raw["up_axis_hint"]],
            yaw_sign=float(imu_raw.get("yaw_sign", 1.0)),
        )

        cfg = RecordingConfig(
            recording_id=recording_id,
            video_path=Path(rec["video_path"]),
            sensor_path=Path(rec["sensor_path"]),
            interim_dir=Path(rec["interim_dir"]),
            processed_dir=Path(rec["processed_dir"]),
            output_dir=Path(rec["output_dir"]),
            csv_has_header=bool(rec.get("csv_has_header", False)),
            timestamp_unit=str(rec.get("timestamp_unit", "ms")),
            imu_mounting=mounting,
            video_to_sensor_offset_s=float(rec.get("video_to_sensor_offset_s", 0.0)),
            camera_calibrated=bool(rec.get("camera_calibrated", False)),
        )
        cfg.ensure_directories()
        return cfg