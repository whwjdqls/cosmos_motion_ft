"""Finish only the missing M2V CFG2 scores; never regenerate saved predictions."""
import gc,hashlib,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'plan.json').read_text());run=next(x for x in plan['runs'] if x['id']==sys.argv[1]);out=Path(run['out']);work=out/'part_all';recovery=out/'recovery_score'
recovery.mkdir(exist_ok=True);snapshot=Path(run['snapshot'])
for part in reversed(('motion_expert_t2m_edge','motion_expert_joint_attention','nymeria_world','')):sys.path.insert(0,str(snapshot/part))
torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.30)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
os.environ['PHASE3_IMAGE_JOINT_PATH_MAP']=json.dumps(run['path_map'])
from motion_expert_phase3_edge_joint.settings import RECIPE,atomic_json,digest_json
from motion_expert_phase3_edge_image_joint.train import configure_local_paths
from motion_expert_phase3_edge_joint.validation import read_configuration
from motion_expert_phase3_edge_joint.validation_metrics import video_scores
from metric_hub import pinned_metric_hub
training=json.loads((Path(run['run_dir'])/'contract.json').read_text());RECIPE.clear();RECIPE.update(training['recipe']);configure_local_paths()
origin=json.loads((work/'configuration.json').read_text());assert digest_json(training)==origin['training_fingerprint']
config,records,identity=read_configuration(plan['validation_config']);assert identity==origin['evaluator_assets']
assert json.loads((work/'acceptance.json').read_text())['passed']
assert json.loads((work/'GENERATION_COMPLETE.json').read_text())['weights_unchanged']
folder=out/'m2v_cfg2';assert (folder/'GENERATION_COMPLETE.json').exists() and not (folder/'COMPLETE.json').exists()
print(json.dumps(dict(stage='scoring_saved_M2V_CFG2',model=run['id'],gpu=os.environ['EVAL_PHYSICAL_GPU'])),flush=True)
policy={**training['recipe']['validation'],'weights':'ema','guidance':2.,'decode_batch_size':4}
with torch.inference_mode(),pinned_metric_hub(config):
    metrics=video_scores(config,records,folder,origin['sources']['phase2_contract'],policy)
atomic_json(folder/'summary.json',metrics);atomic_json(folder/'COMPLETE.json',dict(predictions=71,metrics=metrics,camera_enabled=True,recovered_scoring=True))
results={v['id']:json.loads((out/v['id']/'summary.json').read_text()) for v in plan['variants']}
atomic_json(work/'COMPLETE.json',dict(predictions=284,checkpoint_step=run['step'],metrics=results,scoring_recovered=True))
receipt=dict(predictions_reused=284,new_predictions=0,new_scores=71,metric_guard_preserved=True,
    original_configuration_sha256=hashlib.sha256((work/'configuration.json').read_bytes()).hexdigest(),
    scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    metric_hub_sha256=hashlib.sha256((ROOT/'metric_hub.py').read_bytes()).hexdigest(),
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),metrics=metrics)
atomic_json(recovery/'COMPLETE.json',receipt);print(json.dumps(receipt),flush=True)
