"""Deterministic synthetic geometry checks; no simulator or hidden state input."""
import unittest

import cv2
import numpy as np

from pingpong_rl.vision import (BallDetection, CameraModel, MultiCameraBallTracker,
                       detect_orange_ball, triangulate_observations,
                       triangulate_rays)


def fixture():
    k = np.array([[550., 0, 320], [0, 550., 240], [0, 0, 1]])
    return {name: CameraModel.look_at(name, k, center, [0, 0, .20], width=640, height=480)
            for name, center in [("left_rgb", [-.85, -1.05, 1.15]), ("right_rgb", [.85, -1.05, 1.15])]}


def projected(camera, position, noise=None):
    pixel = camera.project(position)
    if noise is not None:
        pixel += noise
    return BallDetection(pixel, 5., 80., 1., (0, 0, 10, 10))


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.cameras = fixture()

    def test_random_triangulation_and_camera_axis(self):
        rng = np.random.default_rng(13)
        for _ in range(500):
            p = rng.uniform([-.65, -.25, .08], [.65, .25, .6])
            obs = [(c, projected(c, p)) for c in self.cameras.values()]
            out = triangulate_observations(obs)
            self.assertIsNotNone(out)
            self.assertLess(np.linalg.norm(out[0]-p), 1e-10)
            self.assertLess(out[2], 1e-8)

    def test_subpixel_noise_error(self):
        rng = np.random.default_rng(29)
        errors = []
        for _ in range(250):
            p = rng.uniform([-.65, -.25, .08], [.65, .25, .6])
            obs = [(c, projected(c, p, rng.normal(0, .3, 2))) for c in self.cameras.values()]
            out = triangulate_observations(obs)
            self.assertIsNotNone(out)
            errors.append(np.linalg.norm(out[0]-p))
        self.assertLess(np.percentile(errors, 95), .003)

    def test_rgb_blob_centroid_and_full_pipeline(self):
        point = np.array([.1, -.07, .22])
        images = {}
        for name, camera in self.cameras.items():
            image = np.full((480, 640, 3), [24, 87, 130], np.uint8)
            center = np.rint(camera.project(point)).astype(int)
            cv2.circle(image, tuple(center), 7, (255, 120, 12), -1)
            det = detect_orange_ball(image)
            self.assertIsNotNone(det)
            self.assertLess(np.linalg.norm(det.pixel-center), .1)
            images[name] = image
        out = MultiCameraBallTracker(self.cameras).update(images, 0.)
        self.assertTrue(out.visible)
        self.assertLess(np.linalg.norm(out.position_world-point), .004)

    def test_ballistic_velocity_and_control_prediction(self):
        tracker = MultiCameraBallTracker(self.cameras)
        p0 = np.array([-.3, .0, .7]); v0 = np.array([2.0, .05, 1.5]); g = np.array([0, 0, -9.81])
        for i in range(20):
            t = i/60
            p = p0 + v0*t + .5*g*t*t
            out = tracker.update({n:projected(c,p) for n,c in self.cameras.items()}, t)
        self.assertLess(np.linalg.norm(out.velocity_world-(v0+g*t)), .02)
        predicted = tracker.predict(t+1/120)
        true = p0+v0*(t+1/120)+.5*g*(t+1/120)**2
        self.assertLess(np.linalg.norm(predicted[0]-true), .001)

    def test_total_occlusion_ttl_and_duplicate_frame(self):
        tracker = MultiCameraBallTracker(self.cameras,max_prediction_s=.1)
        tracker.update({n:projected(c,[0,0,.3]) for n,c in self.cameras.items()}, 0)
        for i in range(1,10):out=tracker.update({},i/60)
        self.assertIsNone(out.position_world)
        self.assertIsNone(tracker.predict(.2))
        with self.assertRaises(ValueError):tracker.update({},9/60)

    def test_consistent_but_impossible_stereo_jump_is_rejected(self):
        tracker=MultiCameraBallTracker(self.cameras)
        for index in range(4):
            p=np.array([index*.025,0.,.3])
            out=tracker.update({n:projected(c,p) for n,c in self.cameras.items()},index/60)
        out=tracker.update({n:projected(c,[.1,1.3,.3]) for n,c in self.cameras.items()},4/60)
        self.assertFalse(out.visible)
        self.assertIsNone(out.measurement_position_world)
        self.assertLess(abs(out.position_world[1]),.01)

    def test_velocity_reversal_recovers_from_pixels(self):
        tracker=MultiCameraBallTracker(self.cameras,gravity_world=(0,0,0))
        for index in range(12):
            x=index/30 if index<=6 else .2-(index-6)/30
            out=tracker.update({n:projected(c,[x,0,.3]) for n,c in self.cameras.items()},index/60)
            if index==8:self.assertLess(abs(out.velocity_world[0]+2),.15)

    def test_parallel_and_mismatched_rays_rejected(self):
        with self.assertRaises(ValueError):
            triangulate_rays([(np.zeros(3),np.array([0.,0,1.])),(np.array([1.,0,0]),np.array([0.,0,1.]))])
        a,b=self.cameras.values()
        self.assertIsNone(triangulate_observations([(a,projected(a,[0,0,.1])),(b,projected(b,[0,.4,.6]))]))

    def test_distorted_camera_roundtrip(self):
        base=next(iter(self.cameras.values()))
        cam=CameraModel('distorted',base.K,base.R_wc,base.center_world,640,480,np.array([.03,-.01,.002,-.001,.0]))
        point=np.array([.3,.1,.2]);center,ray=cam.ray(cam.project(point))
        self.assertLess(np.linalg.norm(np.cross(point-center,ray)),1e-7)
