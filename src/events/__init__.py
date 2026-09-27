"""Event detectors + shared evidence layer (temporal confirmation).

`EventManager` is the single detector pool for Part A; `TemporalEventEngine`
turns frame-level evidence into confirmed, non-overlapping segments.
"""

from .accident import AccidentDetector
from .congestion import CongestionDetector
from .failure_to_yield import FailureToYieldDetector
from .jaywalking import JaywalkingDetector
from .manager import (DEFAULT_PROVIDERS, DETECTOR_CLASSES, LEGACY_LABELS,
                      NO_PROVIDER_CLASSES, NO_PROVIDER_LABELS, EventManager)
from .red_light import RedLightDetector
from .road_obstacle import RoadObstacleDetector
from .solid_line_crossing import SolidLineCrossingDetector
from .stop_line import StopLineDetector
from .stopped_vehicle import StoppedVehicleDetector
from .temporal import EventSegment, TemporalEventEngine

__all__ = ["AccidentDetector", "CongestionDetector", "DEFAULT_PROVIDERS",
           "DETECTOR_CLASSES", "EventManager", "EventSegment",
           "FailureToYieldDetector", "JaywalkingDetector", "LEGACY_LABELS",
           "NO_PROVIDER_CLASSES", "NO_PROVIDER_LABELS", "RedLightDetector",
           "RoadObstacleDetector", "SolidLineCrossingDetector",
           "StopLineDetector", "StoppedVehicleDetector",
           "TemporalEventEngine"]