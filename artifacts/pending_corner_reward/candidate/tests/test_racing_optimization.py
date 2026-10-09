import sys, json, tempfile, pickle, unittest, os
os.environ.setdefault("SCIPY_ARRAY_API", "1")
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_racing_optimization as racing
from run_speed_curriculum import Runner
from test_speed_curriculum import result
from a2rl_drone_training.config import PlannerConfig
from a2rl_drone_training.hierarchical.planner import validate_reference, offset_gate_seed
from a2rl_drone_training.hierarchical.reference import Reference

class RacingTests(unittest.TestCase):
    def test_racing_rewards_favor_clean_fast_completion_and_punish_failures(self):
        import jax.numpy as jnp
        from a2rl_drone_training.hierarchical.environment import racing_event_rewards
        def rewards(passed=False,finished=False,crashed=False,missed=False,deadline=False,elapsed=13.82,segment=1.):
            return racing_event_rewards(*[jnp.array([v]) for v in (passed,finished,crashed,missed,deadline)],
                jnp.array([segment]),jnp.array([1.]),jnp.array([elapsed]),13.82,.002)
        normal=rewards(passed=True)
        faster=rewards(passed=True,segment=.8)
        self.assertGreater(float(faster['reward_gate'][0]),float(normal['reward_gate'][0]))
        self.assertGreater(float(rewards(finished=True,elapsed=13.)['reward_finish'][0]),float(rewards(finished=True,elapsed=14.)['reward_finish'][0]))
        crash=rewards(passed=True,finished=True,crashed=True,elapsed=1.)
        self.assertEqual(float(crash['reward_gate'][0]),0.)
        self.assertEqual(float(crash['reward_finish'][0]),0.)
        self.assertEqual(float(crash['reward_crash'][0]),-500.)
        self.assertEqual(float(rewards(missed=True)['reward_miss'][0]),-400.)
        self.assertEqual(float(rewards(deadline=True)['reward_miss'][0]),-400.)
        self.assertEqual(float(rewards(segment=.001)['reward_gate'][0]),0.)
        self.assertLess(sum(float(x[0]) for x in rewards().values()),0.)

    def test_racing_action_limits_speed_in_turns_near_frames_and_off_track(self):
        import jax.numpy as jnp
        from a2rl_drone_training.hierarchical.environment import ResidualMPCEnv
        from a2rl_drone_training.config import ResidualConfig
        env=object.__new__(ResidualMPCEnv)
        env.residual=ResidualConfig(racing_objective=True,recovery_critical_areas=True,position_limit_m=.1,speed_fraction=.15,speedup_fraction=.05)
        env.config=SimpleNamespace(vehicle_radius_m=.15)
        env.base=SimpleNamespace(gate_counter=jnp.zeros(4,dtype=int),frame_geometry=SimpleNamespace(clearance=lambda *a:jnp.array([.2,.2,.2,.01])))
        env.reference=SimpleNamespace(project=lambda *a:jnp.zeros(4))
        env.recovery_authority=lambda x:jnp.array([0.,0.,.5,0.])
        env.critical_area_weight=lambda t:jnp.array([0.,1.,0.,0.])
        target=env.residual_target(env.racing_action(jnp.ones((4,4)),jnp.zeros((4,17))))
        np.testing.assert_allclose(target[:,3],[.05,.0125,0.,0.],atol=1e-7)
        self.assertLessEqual(float(jnp.max(jnp.abs(target[:,:3]))),.1)
        slow=env.racing_action(-jnp.ones((4,4)),jnp.zeros((4,17)))
        np.testing.assert_array_equal(slow[:,3],-np.ones(4))

    def test_recovery_braking_limits(self):
        import jax.numpy as jnp
        from a2rl_drone_training.hierarchical.environment import ResidualMPCEnv
        from a2rl_drone_training.config import ResidualConfig
        env=object.__new__(ResidualMPCEnv)
        env.residual=ResidualConfig(racing_objective=True,recovery_braking=True,recovery_critical_areas=True,position_limit_m=.1,speed_fraction=.05,speedup_fraction=.15)
        env.config=SimpleNamespace(vehicle_radius_m=.15)
        env.base=SimpleNamespace(gate_counter=jnp.zeros(4,dtype=int),frame_geometry=SimpleNamespace(clearance=lambda *a:jnp.array([.2,.2,.2,.01])))
        env.reference=SimpleNamespace(project=lambda *a:jnp.zeros(4))
        env.recovery_authority=lambda x:jnp.array([0.,0.,.5,0.])
        env.critical_area_weight=lambda t:jnp.array([0.,1.,0.,0.])
        fast=env.residual_target(env.racing_action(jnp.ones((4,4)),jnp.zeros((4,17))))
        slow=env.residual_target(env.racing_action(-jnp.ones((4,4)),jnp.zeros((4,17))))
        np.testing.assert_allclose(fast[:,3],[.15,.0375,0.,0.],atol=1e-7)
        np.testing.assert_allclose(slow[:,3],[0.,0.,-.05,-.05],atol=1e-7)

    def test_reference_only_training_selects_new_objective_and_bound(self):
        from a2rl_drone_training.train import build_parser
        cmd=racing.racing_training(Path('original.npz'),Path('new'),16,racing_objective=True)
        args=build_parser().parse_args(cmd[3:])
        self.assertTrue(args.residual_racing_objective)
        self.assertEqual(args.reference_path,Path('original.npz'))
        self.assertEqual(args.residual_speedup_fraction,.15)
        self.assertEqual(args.residual_speed_fraction,.05)
        self.assertTrue(args.residual_recovery_braking)
        self.assertTrue(args.residual_corner_geometry_reward)
        self.assertIsNone(args.restore_checkpoint)

    def test_promotion_judges_rl_not_mpc(self):
        racing.require_racing_pair(result(20.),result(19.8),8)
        for bad in (result(19.801),result(20.)):
            with self.assertRaises(ValueError):racing.require_racing_pair(result(),bad,8)
        bad=result(19.);bad['collision'][0]=True
        with self.assertRaises(ValueError):racing.require_racing_pair(result(),bad,8)
        bad=result(19.);bad['lap_time'][0]=np.nan
        with self.assertRaises(ValueError):racing.require_racing_pair(result(),bad,8)
        racing.require_racing_pair(result(),result(20.4),8,1.02)

    def test_fresh_training_and_matched_resume_args(self):
        from a2rl_drone_training.train import build_parser
        cmd=racing.racing_training(Path('ref'),Path('checkpoints'),16)
        args=build_parser().parse_args(cmd[3:])
        self.assertIsNone(args.restore_checkpoint)
        self.assertTrue(args.allow_mpc_baseline_failure)
        self.assertTrue(args.residual_recovery_only)
        self.assertTrue(args.residual_recovery_critical_areas)
        self.assertEqual(args.num_envs,4)
        self.assertEqual(args.horizon,256)
        self.assertEqual(args.schedule_env_steps,131072)
        cmd=racing.racing_training(Path('ref'),Path('checkpoints'),32,Path('latest'))
        self.assertIn('--restore-checkpoint',cmd)

    def test_trainer_baseline_failure_requires_explicit_opt_in(self):
        from a2rl_drone_training.config import TrainingConfig
        from a2rl_drone_training.trainer import PPOTrainer
        obj=object.__new__(PPOTrainer)
        obj.env=SimpleNamespace(course=None)
        obj._pre_correction_v2_checkpoint=True
        for allow in (False,True):
            cfg=TrainingConfig(controller='residual_mpc',allow_mpc_baseline_failure=allow)
            obj.config=replace(cfg,env=replace(cfg.env,reward_version='v2'))
            message='Pre-correction' if allow else 'Zero-offset MPC failed'
            with patch('a2rl_drone_training.hierarchical.evaluation.run_trials',return_value={'success':np.array([False])}):
                with self.assertRaisesRegex(RuntimeError,message):obj.train()

    def test_off_center_seed_keeps_start_and_usable_aperture(self):
        time=np.arange(5,dtype=float);guess=np.zeros((5,17));gates=np.array([2,4])
        course=SimpleNamespace(widths=np.array([2.,2.]),centers=np.zeros((2,3)),right_axes=np.array([[0.,1.,0.]]*2))
        for fraction in (-.5,.5):
            changed=offset_gate_seed(time,guess,gates,course,.25,fraction)
            np.testing.assert_allclose(changed[gates,1],fraction*.75)
            np.testing.assert_allclose(changed[0,:6],guess[0,:6])
            self.assertTrue(np.all(np.abs(changed[gates,1])<.75))
            np.testing.assert_array_equal(guess,np.zeros((5,17)))

    def test_invalid_reference_cannot_be_loaded(self):
        x=np.zeros((2,17));x[:,9]=1
        ref=Reference(np.array([0.,1.]),x,np.zeros((2,4)),np.array([1]),dict(validated=False))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'invalid.npz';ref.save(p)
            with self.assertRaisesRegex(ValueError,'dense feasibility'):Reference.load(p)

    def test_failure_classification_preserves_all_reasons(self):
        states=np.zeros((2,17));states[:,9]=1;states[:,3]=1;states[:,13:]=1
        ref=SimpleNamespace(time=np.array([0.,.002]),states=states,actions=np.zeros((2,4)),gate_indices=np.array([1]))
        course=SimpleNamespace(bounds_min=np.full(3,-100.),bounds_max=np.full(3,100.),centers=np.zeros((1,3)),
                              normals=np.array([[1.,0,0]]),right_axes=np.array([[0.,1,0]]),widths=np.array([2.]),heights=np.array([2.]))
        trajectory=states[None].copy()
        model=SimpleNamespace(rpm_min=0.,rpm_max=2.,rpm_hover=1.,rollout_intervals=lambda *a:trajectory)
        boxes=SimpleNamespace(clearance=lambda *a:np.ones((1,2)),swept_collision=lambda *a:np.zeros((1,1),dtype=bool))
        cfg=PlannerConfig(max_speed_m_s=None)
        with patch('a2rl_drone_training.hierarchical.planner.frame_boxes',return_value=boxes):
            self.assertTrue(validate_reference(ref,course,model,cfg,.15)['validated'])
            for col,value,reason in [(0,200.,'arena_clearance'),(0,.1,'premature_crossing'),(10,20.,'body_rate'),(13,4.,'rotor_speed'),(3,np.nan,'nonfinite')]:
                saved=trajectory.copy();trajectory[:,:,col]=value
                report=validate_reference(ref,course,model,cfg,.15)
                self.assertIn(reason,[f['reason'] for f in report['failures']])
                json.dumps(report['failures'],allow_nan=False)
                trajectory[:]=saved
            ref.actions[0,0]=1.1
            self.assertIn('motor_command',[f['reason'] for f in validate_reference(ref,course,model,cfg,.15)['failures']])
            ref.actions[0,0]=0;ref.states[1,1]=2
            self.assertIn('gate_aperture',[f['reason'] for f in validate_reference(ref,course,model,cfg,.15)['failures']])
            boxes.swept_collision=lambda *a:np.ones((1,1),dtype=bool)
            self.assertIn('frame_collision',[f['reason'] for f in validate_reference(ref,course,model,cfg,.15)['failures']])

    def test_short_orchestration_mpc_failure_does_not_block_rl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=Runner(root/'run',{},1);run.state['search_elapsed_limit']=1e9
            inc=root/'inc.npz';inc.touch();ckpt=root/'inc.pkl';ckpt.touch();seeds=[]
            def metrics(path):return dict(fingerprint=str(path),duration_s=20.,planner_cap_m_s=None)
            def execute(key,command,outputs,inputs=(),**kwargs):
                for out in outputs:
                    out.parent.mkdir(parents=True,exist_ok=True)
                    if out.suffix=='.pkl':
                        update=int(command[command.index('--total-env-steps')+1])//1024
                        ref=command[command.index('--reference-path')+1]
                        out.write_bytes(pickle.dumps(dict(controller='residual_mpc',reference_fingerprint=ref,updates=update)))
                    else:out.touch()
            def evaluate(*args,**kwargs):
                x=result(n=args[4]);x['success'][:]=False;x['collision'][:]=True
                return x,None
            def paired(r,key,ref,checkpoint,inc_ref,inc_ckpt,condition,count,seed,**kwargs):
                seeds.extend(range(seed,seed+count));return result(n=count),result(19.,n=count)
            with patch.object(run,'run',side_effect=execute),patch.object(racing,'reference_metrics',side_effect=metrics),patch.object(racing,'evaluate',side_effect=evaluate),patch.object(racing,'paired',side_effect=paired):
                racing.pipeline(run,inc,ckpt,[(.1,.8)])
            self.assertEqual(run.state['stage'],'development_validation_passed')
            self.assertEqual(len(seeds),len(set(seeds)))
            report=json.loads((run.directory/'development_report.json').read_text())
            self.assertFalse(report['accepted']);self.assertTrue(report['development_validation_passed'])
            self.assertEqual(len(report['conditions']),7)
    def test_fixed_reference_orchestration_never_replans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=Runner(root/'run',{},1);run.state['search_elapsed_limit']=1e9
            inc=root/'inc.npz';inc.touch();ckpt=root/'inc.pkl';ckpt.touch();seeds=[]
            def metrics(path):return dict(fingerprint=str(path),duration_s=20.,planner_cap_m_s=None)
            def execute(key,command,outputs,inputs=(),**kwargs):
                self.assertNotIn('/planning',key)
                for out in outputs:
                    out.parent.mkdir(parents=True,exist_ok=True)
                    if out.suffix=='.pkl':
                        update=int(command[command.index('--total-env-steps')+1])//1024
                        ref=command[command.index('--reference-path')+1]
                        out.write_bytes(pickle.dumps(dict(controller='residual_mpc',reference_fingerprint=ref,updates=update)))
                    else:out.touch()
            def evaluate(*args,**kwargs):
                x=result(n=args[4]);x['success'][:]=False;x['collision'][:]=True
                return x,None
            def paired(r,key,ref,checkpoint,inc_ref,inc_ckpt,condition,count,seed,**kwargs):
                seeds.extend(range(seed,seed+count));return result(n=count),result(19.,n=count)
            with patch.object(run,'run',side_effect=execute),patch.object(racing,'reference_metrics',side_effect=metrics),patch.object(racing,'evaluate',side_effect=evaluate),patch.object(racing,'paired',side_effect=paired):
                racing.pipeline(run,inc,ckpt,[(.1,.8)],reference_only=True)
            self.assertEqual(run.state['stage'],'development_validation_passed')
            self.assertEqual(len(seeds),len(set(seeds)))
            report=json.loads((run.directory/'development_report.json').read_text())
            self.assertFalse(report['accepted']);self.assertTrue(report['development_validation_passed'])
            self.assertEqual(len(report['conditions']),7)

if __name__=='__main__':unittest.main()
