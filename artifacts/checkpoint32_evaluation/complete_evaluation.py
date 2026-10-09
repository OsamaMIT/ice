from pathlib import Path
import os, sys, json, time, signal, subprocess, fcntl, pickle
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'scripts'))
from run_speed_curriculum import atomic_json, combine, digest
from evaluate_racing_pair import signature_for
from evaluate_seeded_condition import load_pair
from run_racing_optimization import require_racing_pair

OUT=ROOT/'artifacts/checkpoint32_evaluation'
OLD=ROOT/'artifacts/reference_racing_policy_15pct_active/attempt1_seed0'
REF=ROOT/'artifacts/recovery_sensitive/uncapped/reference.npz'
CKPT=OLD/'checkpoints/checkpoint_000032.pkl'
INC=ROOT/'artifacts/recovery_sensitive/uncapped/checkpoints/checkpoint_000016.pkl'
state=dict(pid=os.getpid(),child_pid=None,process_alive=True,update_count=32,completed_pairs=0,total_pairs=16,stage='initializing',evaluation_only=True)
def save(**kw):
    state.update(kw,updated_at=time.time());atomic_json(OUT/'status.json',state)
def interrupted(signum,frame):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupted)

def main():
    with (ROOT/'artifacts/.speed_curriculum.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CKPT.open('rb') as f: payload=pickle.load(f)
        assert payload['updates']==32
        assert not getattr(payload['config'].residual,'corner_geometry_reward',False)
        save(checkpoint=str(CKPT),checkpoint_sha256=digest(CKPT),corner_geometry_reward=False)
        reports={};passed=True
        try:
            for index,condition in enumerate(('exact_state','position_1cm')):
                seed=20000200+index*20;folder=OLD/'screen2'/condition
                log=OUT/(condition+'.log')
                cmd=[sys.executable,str(ROOT/'scripts/evaluate_racing_pair.py'),'--reference',str(REF),'--checkpoint',str(CKPT),'--incumbent-reference',str(REF),'--incumbent-checkpoint',str(INC),'--condition',condition,'--trials','8','--seed-start',str(seed),'--output-dir',str(folder)]
                env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
                env.pop('A2RL_RUN_DEADLINE',None)
                save(stage=condition,current_log=str(log))
                with log.open('a') as stream:
                    child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    save(child_pid=child.pid)
                    try:
                        while child.poll() is None:
                            progress=folder/'progress.json'
                            if progress.exists():
                                done=json.loads(progress.read_text()).get('completed_pairs',0)
                                save(completed_condition_pairs=done,completed_pairs=index*8+done)
                            time.sleep(2)
                    finally:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=5)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        save(child_pid=None)
                if child.returncode:raise RuntimeError(f'{condition} worker exited {child.returncode}')
                baseline=[];candidate=[]
                for offset in range(8):
                    sig=signature_for(REF,CKPT,REF,INC,condition,seed+offset)
                    pair=load_pair(folder/f'seed{seed+offset}',sig)
                    if pair is None:raise RuntimeError('Incomplete or hash-mismatched evaluation pair')
                    baseline.append(pair[0]);candidate.append(pair[1])
                report=json.loads((folder/'completed.json').read_text())
                try:require_racing_pair(combine(baseline),combine(candidate),8)
                except ValueError as exc:report['failure']=str(exc);passed=False
                reports[condition]=report
                atomic_json(OUT/'report.json',dict(complete=len(reports)==2,passed=passed and len(reports)==2,conditions=reports,checkpoint=str(CKPT)))
                save(completed_condition_pairs=8,completed_pairs=(index+1)*8)
            atomic_json(OLD/'screen2.json',dict(passed=passed,conditions=reports))
            save(stage='completed',screening_passed=passed,reason=None)
        except BaseException as exc:
            save(stage='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',reason=str(exc));raise
        finally:save(process_alive=False,child_pid=None)
if __name__=='__main__': main()
