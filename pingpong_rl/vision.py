"""RGB-only table-tennis ball reconstruction from calibrated cameras.

The inference path consumes rendered RGB arrays and robot/camera calibration
only.  It never reads depth, object handles, simulator poses, or a checker.
Camera poses are world-from-camera extrinsics supplied in a small JSON file or
constructed by the caller.  A privileged simulator can still be used to train
the policy; this module is the visual observation path used at inference.

Pixel convention: ``(u, v)`` is ``(column, row)`` in the RGB image.  ``R_wc``
maps a camera-frame vector to world coordinates and ``center_world`` is the
camera optical center in world coordinates.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


def rotation_xyzw(quaternion):
    x,y,z,w=np.asarray(quaternion,dtype=float).reshape(4)
    norm=math.sqrt(x*x+y*y+z*z+w*w)
    if norm<1e-12:raise ValueError('camera quaternion is zero')
    x,y,z,w=x/norm,y/norm,z/norm,w/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    R_wc: np.ndarray
    center_world: np.ndarray
    width: int | None = None
    height: int | None = None
    distortion: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "K", np.asarray(self.K, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "R_wc", np.asarray(self.R_wc, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "center_world", np.asarray(self.center_world, dtype=np.float64).reshape(3))
        if self.distortion is not None:
            object.__setattr__(self, "distortion", np.asarray(self.distortion, dtype=np.float64).reshape(-1))
        if not all(np.isfinite(a).all() for a in (self.K, self.R_wc, self.center_world)):
            raise ValueError("camera calibration must be finite")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(self.R_wc.T @ self.R_wc, np.eye(3), atol=1e-5) or np.linalg.det(self.R_wc) < 0:
            raise ValueError("R_wc must be a proper camera-to-world rotation")

    @classmethod
    def look_at(cls, name, K, center_world, target_world, *, world_up=(0, 0, 1), width=None, height=None):
        """Construct an optical +Z-forward, +X-right, +Y-down calibration."""
        center = np.asarray(center_world, dtype=float)
        forward = np.asarray(target_world, dtype=float) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(world_up, dtype=float))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return cls(name, K, np.column_stack((right, down, forward)), center, width, height)

    @classmethod
    def from_dict(cls, name: str, value: Mapping):
        """Read the portable calibration format emitted by ``calibration_json``."""
        k = value.get("K", value.get("matrix"))
        if k is None:
            k = [[value["fx"], 0.0, value["cx"]], [0.0, value["fy"], value["cy"]], [0.0, 0.0, 1.0]]
        r = value.get("R_wc", value.get("rotation_world_from_camera"))
        if r is None:
            if "quat_xyzw_ros" not in value:raise ValueError('camera extrinsics missing R_wc or quat_xyzw_ros')
            r=rotation_xyzw(value["quat_xyzw_ros"])
        c = value.get("center_world", value.get("camera_center_world", value.get("position_world", value.get("t_wc"))))
        if c is None:raise ValueError('camera optical center missing')
        return cls(name, k, r, c, value.get("width"), value.get("height"), value.get("distortion"))

    def ray(self, pixel: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
        d = None if self.distortion is None else self.distortion
        normalized = cv2.undistortPoints(p, self.K, d).reshape(2)
        ray_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
        ray_world = self.R_wc @ ray_camera
        ray_world /= max(np.linalg.norm(ray_world), 1e-12)
        return self.center_world.copy(), ray_world

    def project(self, point_world: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_world, dtype=np.float64) - self.center_world
        point_camera = self.R_wc.T @ point
        if point_camera[2] <= 1e-9:
            return np.array([np.nan, np.nan])
        normalized = (point_camera[:2] / point_camera[2])[None, None, :]
        if self.distortion is None or not np.any(self.distortion):
            return np.array([self.K[0, 0] * normalized[0, 0, 0] + self.K[0, 2],
                             self.K[1, 1] * normalized[0, 0, 1] + self.K[1, 2]])
        distorted, _ = cv2.projectPoints(point_camera.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.distortion)
        return distorted.reshape(2)


@dataclass(frozen=True)
class BallDetection:
    pixel: np.ndarray
    radius_px: float
    area_px: float
    confidence: float
    bbox: tuple[int, int, int, int]

    def __post_init__(self):
        object.__setattr__(self, "pixel", np.asarray(self.pixel, dtype=np.float64).reshape(2))


@dataclass(frozen=True)
class BallEstimate:
    timestamp_s: float
    position_world: np.ndarray | None
    velocity_world: np.ndarray
    visible: bool
    confidence: float
    reprojection_error_px: float
    camera_count: int
    processing_latency_s: float
    measurement_position_world: np.ndarray | None
    prediction_age_s: float

    def as_dict(self) -> dict:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


def detect_orange_candidates(rgb: np.ndarray, *, hue_range=(10, 42), min_area=5,
                             max_area_fraction=0.004, max_candidates=12) -> list[BallDetection]:
    """Keep plausible orange components for stereo and temporal association.

    A wooden racket handle may have the same hue as the ball. Returning only
    the largest blob loses the ball before stereo geometry can disambiguate it.
    ``hue_range`` uses OpenCV's 0..179 hue scale.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB image HxWx3, got {image.shape}")
    image = image[..., :3].astype(np.uint8, copy=False)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lo, hi = hue_range
    mask = cv2.inRange(hsv, np.array([lo, 70, 70], np.uint8), np.array([hi, 255, 255], np.uint8))
    # A 4 cm ball can project to only 5–10 pixels across. Opening erases a
    # partly occluded ball; component size/circularity reject isolated pixels.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < min_area or area > max_area_fraction * h * w:
            continue
        component = (labels[y:y+height, x:x+width] == index).astype(np.uint8)
        perimeter = cv2.arcLength(cv2.findContours(component, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-9)
        aspect = min(width, height) / max(width, height)
        if circularity < 0.25 or aspect < 0.35:
            continue
        radius = math.sqrt(area / math.pi)
        confidence = float(np.clip(.5 * min(1., circularity) + .5 * aspect, 0., 1.))
        # Compact, round regions receive priority; component area must not
        # let a large brown handle eclipse the smaller orange ball.
        candidates.append(BallDetection(centroids[index], radius, float(area),
            confidence, (int(x), int(y), int(width), int(height))))
    return sorted(candidates, key=lambda d: d.confidence, reverse=True)[:max_candidates]


def detect_orange_ball(rgb: np.ndarray, **kwargs) -> BallDetection | None:
    """Compatibility single-frame detector; the tracker uses all candidates."""
    candidates=detect_orange_candidates(rgb, **kwargs)
    return candidates[0] if candidates else None


def triangulate_rays(rays: Sequence[tuple[np.ndarray, np.ndarray]], *, max_condition=1e8) -> tuple[np.ndarray, float]:
    """Closest point to two or more 3-D rays and mean ray distance residual."""
    if len(rays) < 2:
        raise ValueError("at least two camera rays are required")
    matrix = np.zeros((3, 3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    for center, direction in rays:
        direction = np.asarray(direction, dtype=np.float64)
        direction /= max(np.linalg.norm(direction), 1e-12)
        projection = np.eye(3) - np.outer(direction, direction)
        matrix += projection
        vector += projection @ np.asarray(center, dtype=np.float64)
    if np.linalg.cond(matrix) > max_condition:
        raise ValueError("camera rays are nearly parallel; triangulation is ill-conditioned")
    point = np.linalg.solve(matrix, vector)
    residuals = []
    for center, direction in rays:
        residuals.append(np.linalg.norm(np.cross(point - center, direction)))
    return point, float(np.mean(residuals))


def triangulate_observations(observations: Sequence[tuple[CameraModel, BallDetection]], *,
                             max_reprojection_px=12.0) -> tuple[np.ndarray, float, float] | None:
    """Triangulate detections and reject pairs inconsistent in rendered pixels."""
    if len(observations) < 2:
        return None
    candidates = []
    for pair in itertools.combinations(observations, 2):
        _cams, detections = zip(*pair)
        try:
            point, _residual = triangulate_rays([cam.ray(det.pixel) for cam, det in pair])
        except ValueError:
            continue
        errors = [float(np.linalg.norm(cam.project(point) - det.pixel)) for cam, det in pair]
        if not np.isfinite(errors).all() or max(errors) > max_reprojection_px:
            continue
        confidence = float(np.mean([det.confidence for det in detections])) * math.exp(-float(np.mean(errors)) / 8.0)
        candidates.append((float(np.mean(errors)), -confidence, point, confidence))
    if not candidates:
        return None
    error, _neg_conf, point, confidence = min(candidates, key=lambda value: (value[0], value[1]))
    return point, confidence, error


class MultiCameraBallTracker:
    """Synchronized RGB detector, triangulator, and short ballistic tracker.

    ``update`` requires cameras from the same rendered simulation timestamp.
    This prevents silently triangulating a moving ball from different times.
    ``predict`` extrapolates to the controller timestamp without new sensing.
    """
    def __init__(self, cameras: Mapping[str, CameraModel], *, alpha=0.85, beta=0.55,
                 max_reprojection_px=2.0, max_prediction_s=0.10, max_speed_mps=8.0,
                 gravity_world=(0.0, 0.0, -9.81), motion_margin_m=0.025,
                 velocity_jump_mps=0.8, ball_radius_m=.02, min_measurement_confidence=.80,
                 workspace_bounds=((-1.4,-.85,-.35),(1.4,.85,2.1))):
        self.cameras = dict(cameras)
        self.alpha, self.beta = float(alpha), float(beta)
        self.max_reprojection_px = float(max_reprojection_px)
        self.max_prediction_s = float(max_prediction_s)
        self.max_speed_mps = float(max_speed_mps)
        self.motion_margin_m = float(motion_margin_m)
        self.velocity_jump_mps = float(velocity_jump_mps)
        self.gravity = np.asarray(gravity_world, dtype=float).reshape(3)
        self.ball_radius_m=float(ball_radius_m)
        self.workspace_bounds=None if workspace_bounds is None else np.asarray(workspace_bounds,dtype=float).reshape(2,3)
        self.last_association={}
        self.min_measurement_confidence=float(min_measurement_confidence)
        self._measurements=[]
        self.position = None
        self.velocity = np.zeros(3, dtype=np.float64)
        self.timestamp_s = None
        self.last_measurement_s = None
        self.last_measurement_position = None
        self.last_estimate: BallEstimate | None = None

    def update(self, frames: Mapping[str, np.ndarray], timestamp_s: float | None = None) -> BallEstimate:
        started = time.perf_counter()
        timestamp = float(time.monotonic() if timestamp_s is None else timestamp_s)
        if self.timestamp_s is not None and timestamp <= self.timestamp_s:
            raise ValueError("camera timestamp must strictly increase; repeated stale RGB is not a new observation")
        candidates_by_camera = []
        for name, frame in frames.items():
            camera = self.cameras.get(name)
            if camera is None:
                continue
            detections = [frame] if isinstance(frame, BallDetection) else detect_orange_candidates(frame)
            if detections:
                candidates_by_camera.append((camera,detections))
        # Jointly choose camera candidates using only calibrated rays, expected
        # ball size, static workspace bounds and previous RGB measurements.
        # Apply temporal gating before selection, so a wrong background pair
        # cannot suppress an otherwise valid ball pair.
        measurement_dt = None if self.last_measurement_s is None else timestamp-self.last_measurement_s
        matches=[]
        for (cam_a,dets_a),(cam_b,dets_b) in itertools.combinations(candidates_by_camera,2):
            for det_a,det_b in itertools.product(dets_a,dets_b):
                result=triangulate_observations([(cam_a,det_a),(cam_b,det_b)],
                    max_reprojection_px=self.max_reprojection_px)
                if result is None:continue
                point,conf,error=result
                if conf < self.min_measurement_confidence:continue
                if self.workspace_bounds is not None and (
                    np.any(point<self.workspace_bounds[0]) or np.any(point>self.workspace_bounds[1])):continue
                if self.last_measurement_position is not None and measurement_dt is not None:
                    travel=np.linalg.norm(point-self.last_measurement_position)
                    if travel>self.max_speed_mps*measurement_dt+self.motion_margin_m:continue
                if (len(self._measurements)>=2 and self.last_measurement_position is not None
                        and measurement_dt is not None and measurement_dt<=self.max_prediction_s):
                    secant=(point-self.last_measurement_position)/measurement_dt+.5*self.gravity*measurement_dt
                    advance=timestamp-self.timestamp_s
                    expected_v=self.velocity+self.gravity*advance
                    x_reversal=(expected_v[0]*secant[0]<0 and abs(secant[0])>.35
                                )
                    table_rebound=(expected_v[2]<-.10 and secant[2]>.20
                        and min(point[2],self.last_measurement_position[2])<.82
                        and abs(point[0])<.72 and abs(point[1])<.37)
                    if np.linalg.norm(secant-expected_v)>1.2 and not (x_reversal or table_rebound):continue
                ratios=[]
                for cam,det in ((cam_a,det_a),(cam_b,det_b)):
                    depth=float((cam.R_wc.T@(point-cam.center_world))[2])
                    if depth<=self.ball_radius_m:break
                    expected_px=self.ball_radius_m*math.sqrt(cam.K[0,0]*cam.K[1,1])/depth
                    ratios.append(det.radius_px/max(expected_px,1e-6))
                if len(ratios)!=2 or min(ratios)<.30 or max(ratios)>2.0:continue
                size_cost=float(np.mean(np.abs(np.log(ratios))))
                shape_cost=(1-det_a.confidence)+(1-det_b.confidence)
                temporal_cost=0.
                if self.position is not None and self.timestamp_s is not None:
                    elapsed=timestamp-self.timestamp_s
                    expected=self.position+self.velocity*elapsed+.5*self.gravity*elapsed*elapsed
                    # Allow rebounds while preferring continuity among several
                    # epipolar-compatible orange objects in the same frame.
                    temporal_cost=np.linalg.norm(point-expected)/max(.06,self.max_speed_mps*elapsed)
                cost=error/self.max_reprojection_px+1.5*size_cost+2.*shape_cost+.7*temporal_cost
                matches.append((cost,result,(cam_a.name,cam_b.name),(det_a,det_b)))
        match=min(matches,key=lambda x:x[0]) if matches else None
        measurement=None if match is None else match[1]
        self.last_association={'candidate_counts':{c.name:len(d) for c,d in candidates_by_camera},
            'valid_pairs':len(matches),'selected_cameras':None if match is None else list(match[2]),
            'selected_pixels':None if match is None else [d.pixel.tolist() for d in match[3]],
            'cost':None if match is None else float(match[0])}
        measured_position = measurement[0] if measurement is not None else None
        confidence = measurement[1] if measurement is not None else 0.0
        reprojection = float(measurement[2]) if measurement is not None else float("inf")
        camera_count = 0 if measurement is None else 2
        dt = None if self.timestamp_s is None else max(1e-4, timestamp - self.timestamp_s)
        predicted = None if self.position is None else self.position + self.velocity * (dt or 0.0) + .5 * self.gravity * (dt or 0.0)**2
        if measured_position is not None:
            # Reject weak stereo pairs before they can pollute the velocity.
            # All samples here are RGB measurements with calibrated rays.
            fresh = (self.position is None or self.last_measurement_s is None
                     or timestamp-self.last_measurement_s > self.max_prediction_s)
            old_sample = self._measurements[-1] if self._measurements else None
            if fresh:
                self._measurements=[]
                self.velocity=np.zeros(3,dtype=float)
            secant=None
            if old_sample is not None and not fresh:
                sample_dt=timestamp-old_sample[0]
                secant=(measured_position-old_sample[1])/sample_dt + .5*self.gravity*sample_dt
                # Piecewise ballistic motion: a table bounce is supported by
                # its calibrated plane, and a return by an observed RGB X reversal. No contact signal is consumed.
                x_flip=(self.velocity[0]*secant[0]<0 and abs(self.velocity[0])>.35
                        and abs(secant[0])>.35  )
                table_bounce=(self.velocity[2]<-.10 and secant[2]>.20
                    and min(measured_position[2],old_sample[1][2])<.82
                    and abs(measured_position[0])<.72 and abs(measured_position[1])<.37)
                if x_flip or table_bounce:
                    # Keep only points after the change; mixed pre/post-bounce
                    # slopes must not enter the robust smooth-flight fit.
                    self._measurements=[]
                    self.velocity=secant
            self._measurements.append((timestamp,measured_position.copy()))
            self._measurements=[x for x in self._measurements if timestamp-x[0]<=.12][-5:]
            if len(self._measurements)>=2:
                times=np.array([x[0]-timestamp for x in self._measurements])
                positions=np.stack([x[1] for x in self._measurements])
                adjusted=positions-.5*times[:,None]**2*self.gravity
                slopes=[(adjusted[j]-adjusted[i])/(times[j]-times[i])
                        for i in range(len(times)) for j in range(i+1,len(times))]
                # Theil-Sen slopes tolerate one bad sample among five frames.
                self.velocity=np.median(np.stack(slopes),axis=0)
                self.position=np.median(adjusted-times[:,None]*self.velocity,axis=0)
            else:
                self.position=measured_position.copy()
            speed=np.linalg.norm(self.velocity)
            if speed>self.max_speed_mps:self.velocity*=self.max_speed_mps/speed
            visible = True
            self.last_measurement_s = timestamp
            self.last_measurement_position = measured_position.copy()
        elif (predicted is not None and dt is not None and self.last_measurement_s is not None
              and timestamp - self.last_measurement_s <= self.max_prediction_s):
            self.position = predicted
            self.velocity += self.gravity * dt
            visible = False
            confidence = 0.0
        else:
            self.position = None
            self.velocity[:] = 0.0
            visible = False
        self.timestamp_s = timestamp
        estimate = BallEstimate(timestamp, None if self.position is None else self.position.copy(),
                                 self.velocity.copy(), visible, confidence, reprojection, camera_count,
                                 time.perf_counter() - started,
                                 None if measured_position is None else measured_position.copy(),
                                 0.0 if visible else (float("inf") if self.last_measurement_s is None else timestamp-self.last_measurement_s))
        self.last_estimate = estimate
        return estimate

    def predict(self, timestamp_s):
        """Return position/velocity at an action timestamp, or None if stale."""
        if self.position is None or self.timestamp_s is None or self.last_measurement_s is None:
            return None
        dt = float(timestamp_s) - self.timestamp_s
        if dt < 0:
            raise ValueError("cannot predict to a timestamp before the latest frame")
        if timestamp_s - self.last_measurement_s > self.max_prediction_s:
            return None
        return (self.position + self.velocity * dt + .5*self.gravity*dt*dt,
                self.velocity + self.gravity*dt)

    def reset(self):
        self.position = None
        self.velocity[:] = 0.0
        self.timestamp_s = None
        self.last_measurement_s = None
        self.last_measurement_position = None
        self.last_estimate = None
        self.last_association={}
        self._measurements=[]


def load_calibration(path: str | Path) -> dict[str, CameraModel]:
    data = json.loads(Path(path).read_text())
    cameras = data.get("cameras", data)
    return {name: CameraModel.from_dict(name, value) for name, value in cameras.items()}


class RGBBallObserver:
    """environment adapter: only camera_rgb(), camera_calibration(), and a clock.

    Static camera calibration is cached at construction. No object-state or
    privileged_state method is referenced. The caller must pass capture time,
    and may use predict() to advance the state to the control timestamp.
    """
    def __init__(self, camera_owner, *, camera_names=('camera_left','camera_right'), env_index=0,
                 calibration=None, **tracker_options):
        self.camera_owner=camera_owner;self.names=tuple(camera_names);self.env_index=int(env_index)
        raw=camera_owner.camera_calibration() if calibration is None else calibration
        self.cameras={name:CameraModel.from_dict(name,raw[name]) for name in self.names}
        self.tracker=MultiCameraBallTracker(self.cameras,**tracker_options)
        self.last_frames=None

    def update(self,capture_timestamp_s):
        raw=self.camera_owner.camera_rgb();frames={}
        for name in self.names:
            image=raw[name]
            if hasattr(image,'detach'):image=image.detach().cpu().numpy()
            image=np.asarray(image)
            if image.ndim==4:image=image[self.env_index]
            frames[name]=image[...,:3]
        self.last_frames=frames
        return self.tracker.update(frames,capture_timestamp_s)

    def predict(self,control_timestamp_s):return self.tracker.predict(control_timestamp_s)
    def reset(self):self.tracker.reset()


def calibration_json(cameras: Mapping[str, CameraModel]) -> dict:
    return {"version": 1, "convention": "R_wc maps camera rays to world; center_world is optical center",
            "cameras": {name: {"K": camera.K.tolist(), "R_wc": camera.R_wc.tolist(),
                                "center_world": camera.center_world.tolist(), "width": camera.width,
                                "height": camera.height,
                                "distortion": None if camera.distortion is None else camera.distortion.tolist()}
                        for name, camera in cameras.items()}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration")
    parser.add_argument("--print", action="store_true", dest="print_result")
    args = parser.parse_args()
    if args.print_result:
        print(json.dumps(calibration_json(load_calibration(args.calibration)), indent=2))
