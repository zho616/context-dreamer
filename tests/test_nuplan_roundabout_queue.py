"""Protect context activation, scenario fairness and prerequisite failure handling."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
from scripts.run_nuplan_roundabout_queue import (job_config, validate_environment, run, ORDER,
    MASS_FRICTION_ORDER, mass_friction_reference, process_running)


class FollowupQueueTests(unittest.TestCase):
    def reference(self):
        return OmegaConf.create(dict(seed=0,logdir='/old',env=dict(task='metadrive_nuplan',
            steps=500000,context_enabled='${model.context.enabled}',reward=dict(mode='paper'),
            chassis=dict(mass_scales=[.8,1.,1.2])),model=dict(rep_loss='r2dreamer',
            context=dict(enabled=False,injection='recurrent_concat',actor_critic=False,
                         aux_weight=1.,swap_weight=0.))))

    def test_context_activates_history_and_preserves_environment(self):
        before=self.reference()
        after=job_config(before,Path('/new'),500000,'r2dreamer',True)
        self.assertTrue(after.model.context.enabled)
        self.assertTrue(after.env.context_enabled)
        self.assertEqual(after.model.context.wm_grad_scale,0.)
        self.assertEqual(after.model.context.swap_weight,1.)
        self.assertFalse(before.model.context.enabled)
        validate_environment(before,after)

    def test_dynamics_changes_are_rejected(self):
        before=self.reference();after=job_config(before,Path('/new'),500000,'dreamer',False)
        after.env.chassis.mass_scales=[.6,1.4]
        with self.assertRaises(ValueError):validate_environment(before,after)

    def test_requested_order(self):
        self.assertEqual(ORDER,('nuplan_r2dreamer_context','roundabout_sac','roundabout_ppo',
                               'roundabout_dreamerv3','roundabout_dreamerv3_context'))

    def test_mass_friction_protocol_replaces_steering_and_adds_both_r2_runs(self):
        before=self.reference();before.env.chassis.models=['default','m','l']
        before.env.chassis.steering_scales=[.8,.9,1.,1.1,1.2]
        after=mass_friction_reference(before)
        self.assertEqual(list(after.env.chassis.mass_scales),[.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4])
        self.assertEqual(list(after.env.chassis.friction_scales),list(after.env.chassis.mass_scales))
        self.assertEqual(list(after.env.chassis.steering_scales),[1.])
        self.assertEqual(list(after.env.chassis.eval_friction_scales),list(after.env.chassis.friction_scales))
        self.assertEqual(list(before.env.chassis.steering_scales),[.8,.9,1.,1.1,1.2])
        self.assertEqual(len(MASS_FRICTION_ORDER),7)
        self.assertEqual(MASS_FRICTION_ORDER[1:3],('roundabout_r2dreamer','roundabout_r2dreamer_context'))
        self.assertEqual(len(set(MASS_FRICTION_ORDER)),7)

    def test_adoption_rejects_a_pid_with_the_wrong_command(self):
        command=[v.decode()for v in Path(f'/proc/{os.getpid()}/cmdline').read_bytes().split(b'\0')if v]
        self.assertTrue(process_running(os.getpid(),command))
        with self.assertRaises(RuntimeError):process_running(os.getpid(),['unrelated-process'])

    def test_failed_prerequisite_does_not_start_jobs(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);prior=root/'prior.json'
            prior.write_text(json.dumps(dict(phase='failed',runner_pid=0)))
            (root/'plan.json').write_text(json.dumps(dict(sources={},after_status=str(prior),jobs=[])))
            (root/'status.json').write_text(json.dumps(dict(phase='prepared',jobs={})))
            with contextlib.redirect_stderr(io.StringIO()):result=run(root)
            self.assertEqual(result,1)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['phase'],'failed')
            self.assertIn('Prerequisite',state['error'])


if __name__=='__main__':unittest.main()
