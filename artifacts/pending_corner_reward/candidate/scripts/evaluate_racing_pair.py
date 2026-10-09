"""Resumable paired RL-MPC evaluation across different validated references."""
import argparse, json, pickle, time
from dataclasses import replace
from pathlib import Path
from run_speed_curriculum import CONDITIONS, atomic_json, digest, combine
from evaluate_seeded_condition import load_pair, save_pair

def signature_for(reference, checkpoint, incumbent_reference, incumbent_checkpoint, condition, seed):
    return dict(reference=digest(reference), checkpoint=digest(checkpoint),
        incumbent_reference=digest(incumbent_reference), incumbent_checkpoint=digest(incumbent_checkpoint),
        condition=condition, settings=list(CONDITIONS[condition]), seed=seed, implementation='racing-pairs-v1')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','checkpoint','incumbent-reference','incumbent-checkpoint','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',choices=list(CONDITIONS),required=True)
    p.add_argument('--seed-start',type=int,required=True)
    p.add_argument('--trials',type=int,required=True)
    args=p.parse_args()
    if args.trials<1:p.error('Trials must be positive')
    from a2rl_drone_training.runtime import configure_runtime
    configure_runtime('gpu',.5)
    from a2rl_drone_training.trainer import PPOTrainer, _deterministic_action
    from a2rl_drone_training.hierarchical.environment import residual_training_config
    from a2rl_drone_training.hierarchical.evaluation import EvaluationPool, run_trials, summarize
    from a2rl_drone_training.course import course_by_name
    course=course_by_name('arena_38m_stacked')
    pool=EvaluationPool(); trainers={}; configs={}
    def setup(name,reference,checkpoint):
        if name in trainers:return
        with checkpoint.open('rb') as stream:cfg=pickle.load(stream)['config']
        position,velocity,delay=CONDITIONS[args.condition]
        cfg=residual_training_config(replace(cfg,
            env=replace(cfg.env,device='gpu',num_envs=1,max_episode_time_s=120),
            residual=replace(cfg.residual,reference_path=reference,spawn_position_std_m=.02,
                spawn_velocity_std_m_s=.05,estimation_noise_std=0.,position_noise_std_m=position,
                velocity_noise_std_m_s=velocity,sensor_delay_s=delay,sensor_dropout_prob=0.,
                position_bias_m=(0.,0.,0.),velocity_bias_m_s=(0.,0.,0.))))
        configs[name]=cfg;trainers[name]=PPOTrainer(cfg,course=course,restore_checkpoint=checkpoint)
    baselines=[];learned=[]
    try:
        for index in range(args.trials):
            seed=args.seed_start+index;folder=args.output_dir/f'seed{seed}'
            signature=signature_for(args.reference,args.checkpoint,args.incumbent_reference,args.incumbent_checkpoint,args.condition,seed)
            pair=load_pair(folder,signature)
            if pair is None:
                results=[]
                for name,reference,checkpoint in [('incumbent',args.incumbent_reference,args.incumbent_checkpoint),('candidate',args.reference,args.checkpoint)]:
                    setup(name,reference,checkpoint)
                    cfg=configs[name];trainer=trainers[name]
                    policy=lambda obs: _deterministic_action(trainer.state.actor_params,trainer._normalize(obs),cfg.obs)
                    results.append(run_trials(cfg,course,trials=1,seed=seed,policy=policy,
                        record_trace=True,pool=pool,diagnostics_dir=folder/(name+'_live')))
                save_pair(folder,signature,*results)
                pair=results
            baselines.append(pair[0]);learned.append(pair[1])
            atomic_json(args.output_dir/'progress.json',dict(completed_pairs=index+1,trials=args.trials,
                condition=args.condition,seed=seed,updated_at=time.time()))
            print(f'{args.condition}: {index+1}/{args.trials} controller pairs recorded',flush=True)
        atomic_json(args.output_dir/'completed.json',dict(trials=args.trials,seed_start=args.seed_start,
            incumbent=summarize(combine(baselines)),candidate=summarize(combine(learned))))
    finally:
        pool.close()
        for trainer in trainers.values():trainer.close()

if __name__=='__main__':main()
