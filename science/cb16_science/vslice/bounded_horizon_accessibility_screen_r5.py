"""R12 VS-D R5 bounded horizon accessibility screening experiment."""
from __future__ import annotations

import hashlib, json, math, os, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from ..normalization.transforms import n0_endpoint_log_ratios
from . import controlled_tasks as tasks
from . import historical_market_canary_r0 as r0
from . import qualification as q
from . import representation_accessibility_r3 as r3
from .contracts import ContractError

RESULT_COMMAND="cb16.vs-d-bounded-horizon-accessibility-screen-r5@v1"
EXPERIMENT_ID="r12.vs_d.bounded_horizon_accessibility_screen.r5"
REPO_ROOT=Path(__file__).resolve().parents[3]
CONFIG_PATH=REPO_ROOT/'config/experiments/r12_vs_d_bounded_horizon_accessibility_screen_r5.json'
SPEC_FILE_SHA256='f85afb41c66ec4b452185418626dabe51d0abadeb59dddc37cc89cb6c22e935e'
SPEC_FILENAME,RESULT_FILENAME,REPORT_FILENAME='experiment_spec.json','RESULT.json','REPORT.md'
EXIT_PASS,EXIT_SCIENTIFIC_FAIL,EXIT_EXECUTION_BLOCKED,EXIT_CONTRACT_MISMATCH=0,1,3,4

@dataclass(frozen=True)
class HorizonDataset:
    fit_x: np.ndarray
    validation_x: np.ndarray
    fit_targets: Mapping[int,np.ndarray]
    validation_targets: Mapping[int,np.ndarray]
    evidence: Mapping[str,Any]


def load_spec()->Dict[str,Any]:
    raw=CONFIG_PATH.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest!=SPEC_FILE_SHA256: raise ContractError(f'R5 spec byte SHA mismatch: {digest}')
    try: return dict(tasks.spec_mapping(json.loads(raw),'R5 spec'))
    except json.JSONDecodeError as exc: raise ContractError(f'R5 spec invalid JSON: {exc}') from exc


def validate_spec(spec:Mapping[str,Any])->None:
    try: validate_experiment_spec(spec)
    except EvidenceScopeError as exc: raise ContractError(f'R5 claim-authority spec invalid: {exc}') from exc
    if spec.get('experiment_id')!=EXPERIMENT_ID: raise ContractError('R5 experiment_id mismatch')
    if spec.get('experiment_role')!='SCREENING': raise ContractError('R5 must remain SCREENING')
    data=tasks.spec_mapping(spec['data'],'data')
    names=[tasks.spec_str(x['name'],'archive.name') for x in data['allowed_archives']]
    if names!=['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip']: raise ContractError('R5 allowed archives must be Jan-Feb')
    if int(data['hourly_aggregation']['expected_hourly_bars'])!=1440: raise ContractError('R5 hourly count mismatch')
    if int(data['common_fit']['expected_decisions'])!=655 or int(data['common_validation']['expected_decisions'])!=672: raise ContractError('R5 common-window counts mismatch')
    screen=tasks.spec_mapping(spec['screen'],'screen')
    if list(screen['horizons_hours'])!=[1,4,12,24] or list(screen['candidate_horizons_hours'])!=[4,12,24]: raise ContractError('R5 horizon set mismatch')
    if float(screen['ridge_lambda'])!=1.0: raise ContractError('R5 ridge lambda mismatch')
    if tuple(int(x) for x in screen['control_seeds'])!=tuple(range(16001,16009)): raise ContractError('R5 control seeds mismatch')
    u=tasks.spec_mapping(spec['uncertainty_protocol'],'uncertainty_protocol')
    if int(u['bootstrap_replicates'])!=4096 or int(u['bootstrap_seed'])!=17001 or int(u['block_length_hours'])!=48 or int(u['sample_length_hours'])!=672 or int(u['lower_bound_order_index_zero_based'])!=204: raise ContractError('R5 bootstrap contract mismatch')


def _forward_target(hourly:np.ndarray, first_indices:np.ndarray, horizon:int)->np.ndarray:
    idx=np.asarray(first_indices,dtype=np.int64); h=int(horizon)
    if h<1 or idx.ndim!=1: raise ContractError('R5 target horizon/index invalid')
    end=idx+h-1
    if end.size and int(end.max())>=len(hourly): raise ContractError('R5 target reaches outside allowed hourly data')
    opened=np.asarray(hourly[idx,0],dtype=np.float64); closed=np.asarray(hourly[end,3],dtype=np.float64)
    if not np.isfinite(opened).all() or not np.isfinite(closed).all() or not bool((opened>0).all()): raise ContractError('R5 target prices invalid')
    out=closed/opened-1.0
    if not np.isfinite(out).all(): raise ContractError('R5 targets must be finite')
    return out


def load_dataset(spec:Mapping[str,Any])->HorizonDataset:
    times,minutes,evidence=r0._read_allowed_minutes(spec)
    hourly_times,hourly=r0.aggregate_utc_hourly(times,minutes)
    if len(hourly_times)!=1440: raise ContractError('R5 expected 1440 Jan-Feb hourly bars')
    data=spec['data']; context=data['context']; fit=data['common_fit']; val=data['common_validation']
    fit_idx=r0._consequence_indices(hourly_times,r0._parse_utc_ms(fit['first_consequence_hour_start_utc'],'fit.start'),r0._parse_utc_ms(fit['first_consequence_hour_end_utc'],'fit.end'),int(fit['expected_decisions']))
    val_idx=r0._consequence_indices(hourly_times,r0._parse_utc_ms(val['first_consequence_hour_start_utc'],'val.start'),r0._parse_utc_ms(val['first_consequence_hour_end_utc'],'val.end'),int(val['expected_decisions']))
    if np.intersect1d(fit_idx,val_idx).size: raise ContractError('R5 fit/validation indices overlap')
    raw_window=int(context['raw_window_hours'])
    fit_market=n0_endpoint_log_ratios(hourly[r0._window_indices(fit_idx,raw_window)]).values
    val_market=n0_endpoint_log_ratios(hourly[r0._window_indices(val_idx,raw_window)]).values
    if fit_market.shape!=(655,64,5) or val_market.shape!=(672,64,5): raise ContractError('R5 N0 common-window shape mismatch')
    fit_x=np.asarray(fit_market,dtype=np.float64).reshape(655,320); val_x=np.asarray(val_market,dtype=np.float64).reshape(672,320)
    horizons=tuple(int(h) for h in spec['screen']['horizons_hours'])
    fit_targets={h:_forward_target(hourly,fit_idx,h) for h in horizons}; val_targets={h:_forward_target(hourly,val_idx,h) for h in horizons}
    details=dict(evidence); details.update({'hourly_bars':1440,'common_fit_decisions':655,'common_validation_decisions':672,'max_horizon_hours':24,'march_opened':False,'causal_window_relation':'N0_state_strictly_before_first_consequence_hour'})
    return HorizonDataset(fit_x,val_x,fit_targets,val_targets,details)


def fit_predictions(dataset:HorizonDataset,spec:Mapping[str,Any])->Dict[int,Dict[str,Any]]:
    scaler=r3.fit_standardizer(dataset.fit_x); ridge=float(spec['screen']['ridge_lambda']); seeds=[int(x) for x in spec['screen']['control_seeds']]
    out={}
    for h in [int(x) for x in spec['screen']['horizons_hours']]:
        y=dataset.fit_targets[h]
        true_probe=r3.fit_ridge(dataset.fit_x,y,ridge_lambda=ridge,standardizer=scaler)
        true_pred=true_probe.predict(dataset.validation_x)
        ctrls=[]
        for seed in seeds:
            order=np.asarray(tasks.no_fixed_point_permutation(len(y),seed=seed,generation=0),dtype=np.int64)
            shuffled=np.asarray(y)[order]
            if not np.array_equal(np.sort(shuffled),np.sort(y)): raise ContractError('R5 shuffled control must preserve target multiset')
            ctrls.append(r3.fit_ridge(dataset.fit_x,shuffled,ridge_lambda=ridge,standardizer=scaler).predict(dataset.validation_x))
        out[h]={'true_prediction':true_pred,'control_predictions':tuple(ctrls),'active_feature_count':int(scaler.active.sum()),'input_feature_count':int(scaler.active.size)}
    return out


def circular_block_samples(length:int,block_length:int,replicates:int,seed:int)->Tuple[np.ndarray,...]:
    if length<=0 or block_length<=0 or length%block_length: raise ContractError('R5 circular bootstrap requires divisible positive length/block')
    rng=random.Random(seed); blocks=length//block_length; offsets=np.arange(block_length,dtype=np.int64); samples=[]
    for _ in range(replicates):
        starts=[rng.randrange(length) for _ in range(blocks)]
        samples.append(np.concatenate([((s+offsets)%length) for s in starts]))
    return tuple(samples)


def _lcb(values:Sequence[float],index:int)->float:
    ordered=sorted(float(v) for v in values)
    if not 0<=index<len(ordered): raise ContractError('R5 LCB index invalid')
    return ordered[index]


def screen_horizons(dataset:HorizonDataset,predictions:Mapping[int,Mapping[str,Any]],spec:Mapping[str,Any])->Dict[str,Any]:
    u=spec['uncertainty_protocol']; samples=circular_block_samples(672,int(u['block_length_hours']),int(u['bootstrap_replicates']),int(u['bootstrap_seed'])); order_index=int(u['lower_bound_order_index_zero_based'])
    records={}
    ref=1; ref_pred=np.asarray(predictions[ref]['true_prediction'])
    for h in [int(x) for x in spec['screen']['horizons_hours']]:
        y=np.asarray(dataset.validation_targets[h]); true=np.asarray(predictions[h]['true_prediction']); ctrls=[np.asarray(x) for x in predictions[h]['control_predictions']]
        point_true=r3.pearson_corr(true,y); point_ctrls=[r3.pearson_corr(c,y) for c in ctrls]; point_delta=point_true-float(np.median(point_ctrls))
        true_bs=[]; delta_bs=[]; gain_bs=[]
        for idx in samples:
            tb=r3.pearson_corr(true[idx],y[idx]); cb=[r3.pearson_corr(c[idx],y[idx]) for c in ctrls]
            true_bs.append(tb); delta_bs.append(tb-float(np.median(cb)))
            if h!=ref:
                r_y=np.asarray(dataset.validation_targets[ref]); gain_bs.append(tb-r3.pearson_corr(ref_pred[idx],r_y[idx]))
        rec={'horizon_hours':h,'point':{'true_corr':point_true,'median_shuffle_corr':float(np.median(point_ctrls)),'true_minus_median_shuffle_corr':point_delta},'bootstrap':{'true_corr_lcb':_lcb(true_bs,order_index),'true_minus_median_shuffle_corr_lcb':_lcb(delta_bs,order_index)}}
        if h!=ref:
            gain_point=point_true-r3.pearson_corr(ref_pred,np.asarray(dataset.validation_targets[ref])); gain_lcb=_lcb(gain_bs,order_index)
            eligible=rec['bootstrap']['true_corr_lcb']>0 and rec['bootstrap']['true_minus_median_shuffle_corr_lcb']>0 and gain_lcb>0
            rec['point']['true_minus_1h_corr']=gain_point; rec['bootstrap']['true_minus_1h_corr_lcb']=gain_lcb; rec['screen_eligible']=eligible
        records[h]=rec
    eligible=[h for h in [4,12,24] if records[h]['screen_eligible']]
    candidate=None
    if eligible: candidate=sorted(eligible,key=lambda h:(-records[h]['point']['true_corr'],h))[0]
    return {'records':records,'eligible_horizons':eligible,'candidate_horizon_hours':candidate}


def run_experiment(spec:Mapping[str,Any],dataset:HorizonDataset,implementation:Mapping[str,Any])->Dict[str,Any]:
    validate_spec(spec); gate=q.implementation_test_gate(implementation,commit=q.commit_sha())
    if not gate['passed']: raise ContractError('R5 exact-commit implementation test gate is not green')
    predictions=fit_predictions(dataset,spec); screen=screen_horizons(dataset,predictions,spec); candidate=screen['candidate_horizon_hours']
    promotion='NO_PROMOTION' if candidate is None else f'PROMOTE_H{candidate}_CANDIDATE_HYPOTHESIS'
    supported=candidate is not None; classification='PASS' if supported else 'SCIENTIFIC_FAIL'
    clean={str(h):rec for h,rec in screen['records'].items()}
    result={'schema':'cb16.result.v2','experiment_id':EXPERIMENT_ID,'result_command':os.environ.get('CB16_RESULT_COMMAND',RESULT_COMMAND),'commit_sha':q.commit_sha(),'classification':classification,'execution_status':'EXECUTED','validity_status':'VALID','scientific_outcomes':{'horizon_screening_priority':('CANDIDATE_FOUND' if supported else 'NO_CANDIDATE')},'implementation_tests':{**dict(implementation),'gate':gate,'required_by_global_gate':True,'runner_executed_suite':True},'preregistered_spec':{'path':str(CONFIG_PATH.relative_to(REPO_ROOT)),'file_sha256':SPEC_FILE_SHA256,'declared_status':spec.get('status')},'data_evidence':{**dict(dataset.evidence),'manifest_final_holdout_accessed_post_run':False},'horizons':clean,'screen_summary':{'eligible_horizons':screen['eligible_horizons'],'candidate_horizon_hours':candidate},'gate_results':[{'gate_id':'G_HORIZON_SCREEN_CANDIDATE','passed':supported,'metrics':{'eligible_horizons':screen['eligible_horizons'],'candidate_horizon_hours':candidate}}],'claim_assessments':[{'claim_id':'C_HORIZON_SCREENING_PRIORITY','qualification_status':'NOT_APPLICABLE','inference_status':'SUPPORTED' if supported else 'GATE_NOT_MET','attribution_status':'UNRESOLVED'}],'prior_evidence_assessment':[{'claim_ref':'r12.vs_d.representation_accessibility_diagnostic.r3','status':'UNAFFECTED_IN_ORIGINAL_SCOPE'},{'claim_ref':'r12.vs_d.bounded_nonlinear_accessibility_diagnostic.r4','status':'UNAFFECTED_IN_ORIGINAL_SCOPE'}],'promotion_decision':{'decision':promotion,'reason':'R5 is a screening experiment; any candidate requires a new qualification/transfer experiment'},'provenance':{'commit_sha':q.commit_sha(),'spec_file_sha256':SPEC_FILE_SHA256,'parent_r4_run_id':34925584016,'parent_r4_artifact_digest':'sha256:54ffbbbe8e0605a8c9e09b805d18b93961607a7adf0e2d1952efd742a045484f','freshness':'SCREENING_ADAPTIVELY_REUSED_JAN_FEB_AFTER_R4','march_read':False,'final_holdout_accessed':False}}
    try: validate_result_against_spec(result,spec)
    except EvidenceScopeError as exc: raise ContractError(f'R5 result violates claim authority: {exc}') from exc
    return result


def render_report(result:Mapping[str,Any])->str:
    lines=['# R12 VS-D Bounded Horizon Accessibility Screen R5 — REPORT','',f"- classification: **{result['classification']}**",f"- candidate: **{result.get('screen_summary',{}).get('candidate_horizon_hours')}**",f"- promotion: **{result.get('promotion_decision',{}).get('decision','NO_PROMOTION')}**",'- role: **SCREENING — not qualification**','- data: **common Jan fit / Feb adaptively reused development; March not read**','','## Horizon screen','','| h | corr | shuffle median | delta | corr LCB | delta LCB | gain vs 1h | gain LCB | eligible |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for h in ['1','4','12','24']:
        r=result.get('horizons',{}).get(h,{}); p=r.get('point',{}); b=r.get('bootstrap',{})
        lines.append(f"| {h} | {p.get('true_corr',float('nan')):.9f} | {p.get('median_shuffle_corr',float('nan')):.9f} | {p.get('true_minus_median_shuffle_corr',float('nan')):.9f} | {b.get('true_corr_lcb',float('nan')):.9f} | {b.get('true_minus_median_shuffle_corr_lcb',float('nan')):.9f} | {p.get('true_minus_1h_corr',float('nan')):.9f} | {b.get('true_minus_1h_corr_lcb',float('nan')):.9f} | {r.get('screen_eligible',False)} |")
    lines += ['','## Authority boundary','','- This screen may only prioritize a horizon candidate for a new experiment.','- It does not qualify historical market information or profitability.','- A negative screen does not falsify unscreened horizons or other representations.','- March and final holdout remain unopened.','']
    return '\n'.join(lines)


def failure_result(spec:Mapping[str,Any]|None,classification:str,detail:str,implementation:Mapping[str,Any]|None=None)->Dict[str,Any]:
    result={'schema':'cb16.result.v2','experiment_id':EXPERIMENT_ID,'result_command':os.environ.get('CB16_RESULT_COMMAND',RESULT_COMMAND),'commit_sha':q.commit_sha(),'classification':classification,'execution_status':'NOT_EXECUTED','validity_status':'INVALID','scientific_outcomes':{},'gate_results':[],'claim_assessments':[{'claim_id':'C_HORIZON_SCREENING_PRIORITY','qualification_status':'NOT_APPLICABLE','inference_status':'INVALID_FOR_CLAIM','attribution_status':'NOT_APPLICABLE'}],'prior_evidence_assessment':[],'promotion_decision':{'decision':'NO_PROMOTION','reason':'formal run did not reach valid screening adjudication'},'provenance':{'commit_sha':q.commit_sha(),'spec_file_sha256':SPEC_FILE_SHA256},'error':{'type':classification,'detail':detail[:2000]}}
    if implementation is not None: result['implementation_tests']=dict(implementation)
    if spec is not None:
        try: validate_result_against_spec(result,spec)
        except EvidenceScopeError as exc: raise ContractError(f'R5 failure result violates claim authority: {exc}') from exc
    return result


def write_artifacts(result_dir:Path,spec:Mapping[str,Any],result:Mapping[str,Any])->None:
    result_dir.mkdir(parents=True,exist_ok=True); (result_dir/SPEC_FILENAME).write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n'); (result_dir/RESULT_FILENAME).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); (result_dir/REPORT_FILENAME).write_text(render_report(result))


def run_qualification(result_dir:Path)->Tuple[Dict[str,Any],Dict[str,Any],int]:
    spec=load_spec(); validate_spec(spec); implementation=q.run_implementation_test_suite(); gate=q.implementation_test_gate(implementation,commit=q.commit_sha())
    if not gate['passed']:
        classification,code=q.implementation_failure_classification(implementation); result=failure_result(spec,classification,q.implementation_test_failure_detail(classification,implementation),{**dict(implementation),'gate':gate}); write_artifacts(result_dir,spec,result); return spec,result,code
    dataset=load_dataset(spec); result=run_experiment(spec,dataset,implementation); write_artifacts(result_dir,spec,result); return spec,result,(EXIT_PASS if result['classification']=='PASS' else EXIT_SCIENTIFIC_FAIL)


def main()->int:
    raw=(os.environ.get('CB16_RESULT_DIR') or '').strip()
    if not raw or not (os.environ.get('CB16_COMMIT_SHA') or '').strip(): return EXIT_EXECUTION_BLOCKED
    result_dir=Path(raw)
    try: _s,result,code=run_qualification(result_dir)
    except r0.HistoricalDataUnavailable as exc:
        try: spec=load_spec()
        except Exception: spec=None
        result=failure_result(spec,'EXECUTION_BLOCKED',f'{type(exc).__name__}: {exc}'); write_artifacts(result_dir,spec or {},result); return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try: spec=load_spec()
        except Exception: spec=None
        result=failure_result(spec,'CONTRACT_MISMATCH',f'{type(exc).__name__}: {exc}'); write_artifacts(result_dir,spec or {},result); return EXIT_CONTRACT_MISMATCH
    except Exception as exc:
        try: spec=load_spec()
        except Exception: spec=None
        result=failure_result(spec,'EXECUTION_BLOCKED',f'{type(exc).__name__}: {exc}'); write_artifacts(result_dir,spec or {},result); return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} candidate={result['screen_summary']['candidate_horizon_hours']}"); return code

if __name__=='__main__': raise SystemExit(main())
