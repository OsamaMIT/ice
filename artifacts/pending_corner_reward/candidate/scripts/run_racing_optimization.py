"""Measured-risk racing search with recovery RL and incumbent-controller promotion."""
import argparse, fcntl, json, os, signal, sys
from pathlib import Path
import numpy as np
from run_speed_curriculum import (ROOT,Runner,BudgetExceeded,atomic_json,digest,build_spec,
    reference_metrics,checkpoint_updates,training_command,evaluate,combine,CONDITIONS)
from run_staged_validation import require_clean_flights
from evaluate_seeded_condition import load_pair
from evaluate_racing_pair import signature_for

RISK_SCHEDULE=[(.10,.80),(.075,.80),(.075,.85),(.05,.90)]
SEEDS=[0.,.5,-.5]

def require_racing_pair(incumbent,candidate,trials,ratio=.99):
    require_clean_flights(candidate,trials)
    require_clean_flights(incumbent,trials)
    if np.mean(candidate['lap_time'])>ratio*np.mean(incumbent['lap_time'])+1e-9:
        raise ValueError(f'Candidate lap time exceeds permitted incumbent ratio {ratio}')

def racing_training(reference,folder,target,restore=None,racing_objective=False):
    cmd=training_command(reference,folder,target)
    cmd+=['--allow-mpc-baseline-failure','--residual-recovery-only','--residual-recovery-critical-areas',
          '--residual-position-limit','.1','--residual-speed-fraction','.15','--residual-speedup-fraction','0']
    if racing_objective:
        cmd+=['--residual-racing-objective','--residual-recovery-braking','--residual-corner-geometry-reward']
        cmd[cmd.index('--residual-speed-fraction')+1]='.05'
        cmd[cmd.index('--residual-speedup-fraction')+1]='.15'
    for flag,value in (('--spawn-position-std','.08'),('--spawn-velocity-std','.15')):
        cmd[cmd.index(flag)+1]=value
    if restore:cmd+=['--restore-checkpoint',str(restore)]
    return cmd

def paired(runner,key,reference,checkpoint,inc_ref,inc_ckpt,condition,count,seed,completed_before=0):
    folder=runner.directory/key
    runner.run(key,[sys.executable,str(ROOT/'scripts/evaluate_racing_pair.py'),
        '--reference',str(reference),'--checkpoint',str(checkpoint),'--incumbent-reference',str(inc_ref),
        '--incumbent-checkpoint',str(inc_ckpt),'--condition',condition,'--trials',str(count),
        '--seed-start',str(seed),'--output-dir',str(folder)], [folder/'completed.json'],
        inputs=[reference,checkpoint,inc_ref,inc_ckpt],progress_file=folder/'progress.json',completed_before=completed_before)
    old=[];new=[]
    for offset in range(count):
        signature=signature_for(reference,checkpoint,inc_ref,inc_ckpt,condition,seed+offset)
        pair=load_pair(folder/f'seed{seed+offset}',signature)
        if pair is None:raise ValueError('Incomplete evaluation pair')
        old.append(pair[0]);new.append(pair[1])
    return combine(old),combine(new)

def pipeline(runner,inc_ref,inc_ckpt,schedule,reference_only=False):
    from a2rl_drone_training.hierarchical.evaluation import summarize
    selected_path=runner.directory/"screening_qualified.json"
    selected=json.loads(selected_path.read_text()) if selected_path.exists() else []
    seen=set(); inc_metrics=reference_metrics(inc_ref)
    # Leave a fixed portion of remaining active time for broad development validation.
    search_deadline=runner.state['search_elapsed_limit']
    for attempt,(margin,motor) in enumerate(schedule[:1] if reference_only else schedule):
        for seed_index,fraction in enumerate([0.] if reference_only else SEEDS):
            if runner.used+(__import__('time').monotonic()-runner.started)>=search_deadline:break
            name=f'attempt{attempt+1}_seed{seed_index}'
            folder=runner.directory/name;folder.mkdir(exist_ok=True)
            if reference_only:
                reference=inc_ref
                metrics=inc_metrics
                runner.save("fixed_reference_ready", candidate=name, reference=metrics)
            else:
                reference=folder/'reference.npz'
                runner.save('planning',candidate=name,requested_margin_m=margin,requested_motor_limit=motor,
                            gate_offset_fraction=fraction,reason=None)
                try:
                    runner.run(name+'/planning',[str(ROOT/'.venv/bin/a2rl-drone-plan'),'--max-speed','none',
                        '--warm-start',str(inc_ref),'--motor-command-limit',str(motor),'--tracking-margin',str(margin),
                        '--gate-offset-fraction',str(fraction),'--diagnostics-dir',str(folder/'diagnostics'),
                        '--nodes-per-segment','24','--max-refinements','1','--max-iterations','2000',
                        '--output',str(reference)],[reference],inputs=[inc_ref],cpu=True)
                except RuntimeError as exc:
                    if isinstance(exc,BudgetExceeded):raise
                    runner.save('candidate_planning_failed',reason=str(exc));continue
                metrics=reference_metrics(reference)
                failures=[]
                for report in sorted((folder/'diagnostics').glob('*.json')):
                    entry=json.loads(report.read_text());failures.append(dict(file=str(report),report=entry))
                atomic_json(folder/'diagnosis.json',dict(reproduction=attempt==0 and seed_index==0,candidates=failures))
                if metrics['fingerprint'] in seen or metrics['fingerprint']==inc_metrics['fingerprint']:
                    runner.save('unchanged_reference_skipped');continue
                seen.add(metrics['fingerprint'])
            for label,count,nominal in [('nominal',1,True),('perturbed',4,False)]:
                baseline,_=evaluate(runner,name+'/'+label,reference,folder/label,count,170929,nominal=nominal)
                atomic_json(folder/(label+'_diagnostic.json'),summarize(baseline))
            checkpoints=folder/'checkpoints'
            for block,target in enumerate(range(16,129,16),1):
                if runner.used+(__import__('time').monotonic()-runner.started)>=search_deadline:break
                latest=checkpoints/'checkpoint_latest.pkl';checkpoint=checkpoints/f'checkpoint_{target:06d}.pkl'
                if latest.exists():
                    current=checkpoint_updates(latest,metrics)
                    if current>=target and not checkpoint.exists():raise ValueError('Missing numbered checkpoint')
                runner.run(name+f'/train{block}',racing_training(reference,checkpoints,target,latest if latest.exists() else None,racing_objective=reference_only),
                    [checkpoint],inputs=[reference],checkpoints=checkpoints)
                if checkpoint_updates(checkpoint,metrics)!=target:raise ValueError('Wrong update count')
                reports={};ratios=[];passed=True
                for index,condition in enumerate(('exact_state','position_1cm')):
                    seed=20000000+attempt*100000+seed_index*10000+block*100+index*20
                    old,new=paired(runner,name+f'/screen{block}/'+condition,reference,checkpoint,
                        inc_ref,inc_ckpt,condition,8,seed)
                    reports[condition]=dict(incumbent=summarize(old),candidate=summarize(new))
                    try:require_racing_pair(old,new,8)
                    except ValueError as exc:
                        reports[condition]['failure']=str(exc);passed=False
                    ratios.append(float(np.mean(new['lap_time'])/np.mean(old['lap_time'])))
                atomic_json(folder/f'screen{block}.json',dict(passed=passed,conditions=reports))
                if passed:
                    selected.append(dict(reference=str(reference),checkpoint=str(checkpoint),score=max(ratios),conditions=reports))
                    atomic_json(runner.directory/'screening_qualified.json',selected)
                    runner.save('candidate_screening_passed',selected=selected)
                    break
    if not selected:
        runner.save('no_qualified_improvement',reason='No candidate passed faster RL-MPC screening within search allocation')
        return
    best=min(selected,key=lambda item:item['score']); reports={}
    for index,condition in enumerate(CONDITIONS):
        old,new=paired(runner,'development/'+condition,Path(best['reference']),Path(best['checkpoint']),
            inc_ref,inc_ckpt,condition,20,30000000+index*1000,completed_before=index*20)
        reports[condition]=dict(incumbent=summarize(old),candidate=summarize(new))
        try:require_racing_pair(old,new,20,1.02)
        except ValueError as exc:
            reports[condition]['failure']=str(exc)
            atomic_json(runner.directory/'development_report.json',dict(accepted=False,development_validation_passed=False,conditions=reports))
            runner.save('development_failed',reason=str(exc));return
        atomic_json(runner.directory/'development_report.json',dict(accepted=False,development_validation_passed=False,conditions=reports))
    atomic_json(runner.directory/'development_report.json',dict(accepted=False,development_validation_passed=True,
        label='development validation passed',selected=best,conditions=reports))
    runner.save('development_validation_passed',selected=best,reason=None)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/racing_optimization'))
    parser.add_argument('--incumbent-reference',type=Path,default=Path('artifacts/recovery_sensitive/uncapped/reference.npz'))
    parser.add_argument('--incumbent-checkpoint',type=Path,default=Path('artifacts/recovery_sensitive/uncapped/checkpoints/checkpoint_000016.pkl'))
    parser.add_argument('--carry-budget-from',type=Path,default=Path('artifacts/faster_reference/status.json'))
    parser.add_argument('--risk-schedule',default='0.10:0.80,0.075:0.80,0.075:0.85,0.05:0.90')
    parser.add_argument('--budget-hours',type=float,default=24.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reference-only',action='store_true',help='Learn faster racing corrections around the incumbent reference without replanning')
    args=parser.parse_args();os.chdir(ROOT)
    schedule=[tuple(map(float,item.split(':'))) for item in args.risk_schedule.split(',')]
    if any(len(x)!=2 or not np.isfinite(x).all() or x[0]<.05 or not 0<x[1]<=.9 for x in schedule):parser.error('Invalid measured-risk schedule')
    with (ROOT/'artifacts/.speed_curriculum.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if int(proc.name)==os.getpid() or (proc/'cwd').resolve()!=ROOT:continue
                parts=(proc/'cmdline').read_bytes().split(b'\0')
                if any(Path(x.decode()).name in ('run_speed_curriculum.py','run_residual_curriculum.py','optimize_faster_reference.py','run_racing_optimization.py','a2rl-drone-plan','a2rl-drone-evaluate','evaluate_seeded_condition.py','evaluate_racing_pair.py') or x==b'a2rl_drone_training.train' for x in parts):raise RuntimeError(f'Active pipeline PID {proc.name}')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        prior=json.loads(args.carry_budget_from.read_text())
        if prior.get('process_alive'):raise ValueError('Prior pipeline still active')
        spec=build_spec(args.incumbent_reference)
        spec.update(reference_only=args.reference_only,racing_runner=digest(Path(__file__)),paired_worker=digest(ROOT/'scripts/evaluate_racing_pair.py'),
            incumbent_checkpoint=digest(args.incumbent_checkpoint),risk_schedule=schedule,prior_run=digest(args.carry_budget_from))
        spec=json.loads(json.dumps(spec));runner=Runner(args.output_dir,spec,args.budget_hours,args.resume)
        if not args.resume:
            runner.used=prior['elapsed_seconds']
            runner.save('initializing',search_elapsed_limit=runner.used+.6*runner.remaining(),budget_carried_from=str(args.carry_budget_from))
        def stop(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,stop)
        try:pipeline(runner,args.incumbent_reference,args.incumbent_checkpoint,schedule,reference_only=args.reference_only)
        except BudgetExceeded as exc:runner.save('budget_exhausted',reason=str(exc))
        except BaseException as exc:
            runner.save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'blocked',reason=str(exc));raise
        finally:runner.save(process_alive=False,child_pid=None)

if __name__=='__main__':main()
