"""Install tested reward changes only after the specified pipeline finishes."""
from pathlib import Path
import fcntl, hashlib, json, os, time

TERMINAL = {'budget_exhausted', 'no_qualified_improvement', 'development_validation_passed', 'development_failed'}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path, data):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_bytes(data)
    os.replace(temporary, path)
def try_apply(root, staging):
    state = json.loads((root/'artifacts/reference_racing_policy_15pct_active/status.json').read_text())
    if state.get('process_alive') or state.get('stage') not in TERMINAL:
        return 'waiting'
    with (root/'artifacts/.speed_curriculum.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return 'waiting'
        state = json.loads((root/'artifacts/reference_racing_policy_15pct_active/status.json').read_text())
        if state.get('process_alive') or state.get('stage') not in TERMINAL:
            return 'waiting'
        manifest = json.loads((staging/'manifest.json').read_text())
        for name, hashes in manifest.items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()): raise ValueError('Path outside project')
            before = sha(target) if target.exists() else None
            if before != hashes['before']: raise ValueError('Source changed: ' + name)
            if sha(staging/'candidate'/name) != hashes['after']: raise ValueError('Staged source changed: ' + name)
        installed=[]
        try:
            for name in manifest:
                target=root/name; backup=staging/'backup'/name
                if target.exists():
                    backup.parent.mkdir(parents=True,exist_ok=True); backup.write_bytes(target.read_bytes())
                target.parent.mkdir(parents=True,exist_ok=True)
                atomic(target,(staging/'candidate'/name).read_bytes()); installed.append(name)
        except BaseException:
            for name in reversed(installed):
                target=root/name; backup=staging/'backup'/name
                if backup.exists(): atomic(target,backup.read_bytes())
                elif target.exists(): target.unlink()
            raise
        return 'applied'
def main():
    staging=Path(__file__).resolve().parent
    root=staging.parent.parent
    status=staging/'status.json'
    while True:
        try:
            result=try_apply(root,staging)
            detail=dict(stage=result,pid=os.getpid(),updated_at=time.time(),auto_training_launch=False)
        except Exception as exc:
            result='blocked'; detail=dict(stage=result,reason=str(exc),pid=os.getpid(),updated_at=time.time())
        atomic(status,json.dumps(detail,indent=2).encode())
        if result != 'waiting':
            print(json.dumps(detail),flush=True); return
        time.sleep(15)
if __name__=='__main__': main()
