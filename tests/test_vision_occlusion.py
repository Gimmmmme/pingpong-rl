"""Regression cases for final-approach occlusion and contaminated ball slopes."""
import unittest
import numpy as np
from pingpong_rl.vision import BallDetection,MultiCameraBallTracker
from test_vision import fixture
class RobustApproachTests(unittest.TestCase):
 def setUp(self):
  self.cams=fixture();self.p=np.array([-.3,.04,.8]);self.v=np.array([1.8,-.02,.7]);self.g=np.array([0.,0.,-9.81])
 def truth(self,t):return self.p+self.v*t+.5*self.g*t*t,self.v+self.g*t
 def frames(self,p,confidence=1.):
  return {name:BallDetection(c.project(p),5.,80.,confidence,(0,0,10,10)) for name,c in self.cams.items()}
 def test_80ms_occlusion_keeps_state_then_expires(self):
  tracker=MultiCameraBallTracker(self.cams,max_prediction_s=.10)
  for i in range(10):
   p,v=self.truth(i/60);out=tracker.update(self.frames(p) if i<5 else {},i/60)
   self.assertIsNotNone(out.position_world);self.assertLess(np.linalg.norm(out.position_world-p),1e-8)
  self.assertIsNone(tracker.update({},11/60).position_world)
 def test_three_weak_measurements_do_not_pollute_velocity(self):
  tracker=MultiCameraBallTracker(self.cams,max_prediction_s=.10)
  for i in range(10):
   p,v=self.truth(i/60);bad=5<=i<=7;observed=p+np.array([0,.045,-.03]) if bad else p
   out=tracker.update(self.frames(observed,.65 if bad else 1.),i/60)
   if i>=2:
    self.assertLess(np.linalg.norm(out.velocity_world-v),1e-7)
    self.assertLess(np.linalg.norm(out.position_world-p),1e-7)
   if bad:self.assertFalse(out.visible)
 def test_high_confidence_lateral_jump_is_rejected(self):
  tracker=MultiCameraBallTracker(self.cams)
  for i in range(9):
   p,v=self.truth(i/60);observed=p+np.array([0,.06,0]) if i==5 else p
   out=tracker.update(self.frames(observed),i/60)
   if i==5:self.assertFalse(out.visible)
   if i>=2:self.assertLess(np.linalg.norm(out.velocity_world-v),1e-7)
if __name__=='__main__':unittest.main(verbosity=2)
