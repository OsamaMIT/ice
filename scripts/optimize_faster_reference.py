"""Budgeted uncapped speed optimization followed by MPC qualification."""
import argparse, fcntl, json, os, signal
from pathlib import Path
from run_speed_curriculum import ROOT, Runner, build_spec, digest, reference_metrics, evaluate, atomic_json, BudgetExceeded
from run_staged_validation import require_clean_flights

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    with (ROOT/'artifacts/.speed_curriculum.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path('artifacts/recovery_sensitive/uncapped/reference.npz')
        prior_path=Path('artifacts/recovery_sensitive/status.json')
        prior=json.loads(prior_path.read_text())
        if prior['process_alive']:raise RuntimeError('Previous evaluation is still active')
        spec=build_spec(source)
        spec.update(speed_optimizer=digest(Path(__file__)),prior_run=digest(prior_path))
        runner=Runner(Path('artifacts/faster_reference'),spec,24,args.resume)
        if not args.resume:
            runner.used=prior['elapsed_seconds']
            runner.save('initializing',reason=None,budget_carried_from=str(prior_path))
        def stop(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,stop)
        try:
            candidate=runner.directory/'reference.npz'
            runner.run('planning',[str(ROOT/'.venv/bin/a2rl-drone-plan'),
                '--max-speed','none','--motor-command-limit','.8','--warm-start',str(source),
                '--nodes-per-segment','24','--max-refinements','1','--max-iterations','2000',
                '--output',str(candidate)],[candidate],inputs=[source],cpu=True)
            old,new=reference_metrics(source),reference_metrics(candidate)
            report=dict(previous=old,candidate=new,qualified=False,
                        planned_time_improvement_s=old['duration_s']-new['duration_s'])
            atomic_json(runner.directory/'comparison.json',report)
            runner.save('candidate_ready',comparison=report)
            if new['duration_s']>=old['duration_s']:
                runner.save('no_faster_validated_candidate',reason='Existing trajectory retained; no speed improvement demonstrated')
                return
            from a2rl_drone_training.hierarchical.evaluation import summarize
            for name,count,nominal in [('nominal',1,True),('perturbed',4,False)]:
                baseline,_=evaluate(runner,name,candidate,runner.directory/name,count,170929,nominal=nominal)
                report[name]=summarize(baseline)
                atomic_json(runner.directory/'comparison.json',report)
                require_clean_flights(baseline,count)
            report['qualified']=True
            atomic_json(runner.directory/'comparison.json',report)
            runner.save('faster_reference_mpc_qualified',comparison=report,
                        reason='Fresh RL training required against the changed reference')
        except BudgetExceeded as exc:runner.save('budget_exhausted',reason=str(exc))
        except BaseException as exc:
            runner.save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'blocked',reason=str(exc))
            raise
        finally:runner.save(process_alive=False,child_pid=None)

if __name__=='__main__':main()
