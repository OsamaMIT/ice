import json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
sys.path.insert(0,str(ROOT/'scripts'))
from run_speed_curriculum import atomic_json, digest, checkpoint_updates, build_spec
folder=ROOT/'artifacts/rl_speed_curriculum'
status=folder/'status.json'
initial=json.loads(status.read_text())
assert initial['stage']=='12mps/train1' and initial['process_alive']
start=initial['update_count']; pid=initial['pid']
old_spec=initial['spec']
# Only scheduling/orchestration changes are allowed for this upgrade.
current=build_spec(Path('artifacts/residual_headroom/reference.npz'))
for key in ('source','initial_reference','dependencies','staging_helper','evaluation_worker'):
    assert old_spec[key]==current[key], f'Unexpected compatibility change: {key}'
marker=folder/'early_screening_upgrade.json'
atomic_json(marker,dict(state='waiting_for_checkpoint',after_update=start,screen_every=16))
latest=folder/'12mps/checkpoints/checkpoint_latest.pkl'
while True:
    state=json.loads(status.read_text())
    assert state['pid']==pid and state['process_alive'] and state['stage']=='12mps/train1', 'Runner changed before upgrade'
    try:
        update=checkpoint_updates(latest,state['reference'])
    except (OSError,EOFError):
        time.sleep(2); continue
    if update>start: break
    time.sleep(2)
command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
assert b'scripts/run_speed_curriculum.py' in command and b'artifacts/rl_speed_curriculum' in command
os.kill(pid,signal.SIGTERM)
for _ in range(60):
    stopped=json.loads(status.read_text())
    if not stopped.get('process_alive'): break
    time.sleep(1)
else: raise RuntimeError('Runner did not stop; no migration performed')
child=state.get('child_pid')
if child and (Path('/proc')/str(child)).exists():
    raise RuntimeError('Child still exists; refusing concurrent launch')
for _ in range(60):
    proc_path=Path('/proc')/str(pid)
    if not proc_path.exists() or not (proc_path/'cmdline').read_bytes(): break
    time.sleep(1)
else: raise RuntimeError('Old runner has not exited; refusing restart')
atomic_json(folder/'status.before_early_screening.json',stopped)
spec=dict(old_spec,runner=current['runner'],updates_per_block=16,max_blocks=8)
stopped['spec']=spec
atomic_json(status,stopped)
args=[sys.executable,'scripts/run_speed_curriculum.py','--output-dir','artifacts/rl_speed_curriculum',
 '--previous-reference','artifacts/residual_headroom/reference.npz','--caps','12','24','none',
 '--motor-command-limit','0.8','--train-before-qualification','--budget-hours','24',
 '--carry-budget-from','artifacts/residual_headroom/status.json','--screen-every','16','--resume']
with (folder/'early_screening_restart.log').open('ab',buffering=0) as log:
    proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(marker,dict(state='restarted',saved_update=update,new_pid=proc.pid,screen_every=16,
                       preserved_elapsed_seconds=stopped['elapsed_seconds']))
print('Restarted with earlier screening',proc.pid,'checkpoint',update,flush=True)
