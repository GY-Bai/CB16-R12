"""R12 VS-D R6 bounded context-length accessibility screening experiment."""
from __future__ import annotations

import hashlib, json, os, sys
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
from . import bounded_horizon_accessibility_screen_r5 as r5
from .contracts import ContractError

RESULT_COMMAND="cb16.vs-d-bounded-context-length-accessibility-screen-r6@v1"
EXPERIMENT_ID="r12.vs_d.bounded_context_length_accessibility_screen.r6"
REPO_ROOT=Path(__file__).resolve().parents[3]
CONFIG_PATH=REPO_ROOT/'config/experiments/r12_vs_d_bounded_context_length_accessibility_screen_r6.json'
SPEC_FILE_SHA256='3e4706036bb7227aff885ec768d1c6a1db32e17e5d4f4aed997c42c091d3c8dd'
SPEC_FILENAME,RESULT_FILENAME,REPORT_FILENAME='experiment_spec.json','RESULT.json','REPORT.md'
EXIT_PASS,EXIT_SCIENTIFIC_FAIL,EXIT_EXECUTION_BLOCKED,EXIT_CONTRACT_MISMATCH=0,1,3,4

@dataclass(frozen=True)
class ContextDataset:
    fit_surfaces: Mapping[int,np.ndarray]
    validation_surfaces: Mapping[int,np.ndarray]
    fit_target: np.ndarray
    validation_target: np.ndarray
    evidence: Mapping[str,Any]


def load_spec()->Dict[str,Any]:
    raw=CONFIG_PATH.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest!=SPEC_FILE_SHA256: raise ContractError(f'R6 spec byte SHA mismatch: {digest}')
    try: return dict(tasks.spec_mapping(json.loads(raw),'R6 spec'))
    except json.JSONDecodeError as exc: raise ContractError(f'R6 spec invalid JSON: {exc}') from exc


def validate_spec(spec:Mapping[str,Any])->None:
    try: validate_experiment_spec(spec)
    except EvidenceScopeError as exc: raise ContractError(f'R6 claim-authority spec invalid: {exc}') from exc
    if spec.get('experiment_id')!=EXPERIMENT_ID: raise ContractError('R6 experiment_id mismatch')
    if spec.get('experiment_role')!='SCREENING': raise ContractError('R6 must remain SCREENING')
    data=tasks.spec_mapping(spec['data'],'data')
    names=[tasks.spec_str(x['name'],'archive.name') for x in data['allowed_archives']]
    if names!=['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip']: raise ContractError('R6 allowed archives must be Jan-Feb')
    contexts=[(int(x['represented_hours']),int(x['raw_window_hours']),int(x['feature_width'])) for x in data['contexts']]
    if contexts!=[(16,17,80),(32,33,160),(64,65,320),(128,129,640)]: raise ContractError('R6 context contract mismatch')
    if int(data['common_fit']['expected_decisions'])!=615 or int(data['common_validation']['expected_decisions'])!=672: raise ContractError('R6 common-window counts mismatch')
    screen=tasks.spec_mapping(spec['screen'],'screen')
    if list(screen['context_lengths_hours'])!=[16,32,64,128] or list(screen['candidate_contexts_hours'])!=[16,32,128] or int(screen['reference_context_hours'])!=64: raise ContractError('R6 context set/reference mismatch')
    if float(screen['ridge_lambda'])!=1.0: raise ContractError('R6 ridge lambda mismatch')
    if tuple(int(x) for x in screen['control_seeds'])!=tuple(range(18001,18009)): raise ContractError('R6 control seeds mismatch')
    if not bool(screen['same_target_permutation_across_contexts']): raise ContractError('R6 requires matched target permutations across contexts')
    u=tasks.spec_mapping(spec['uncertainty_protocol'],'uncertainty_protocol')
    if int(u['bootstrap_replicates'])!=4096 or int(u['bootstrap_seed'])!=19001 or int(u['block_length_hours'])!=48 or int(u['sample_length_hours'])!=672 or int(u['lower_bound_order_index_zero_based'])!=204: raise ContractError('R6 bootstrap contract mismatch')


def load_dataset(spec:Mapping[str,Any])->ContextDataset:
    times,minutes,evidence=r0._read_allowed_minutes(spec)
    hourly_times,hourly=r0.aggregate_utc_hourly(times,minutes)
    if len(hourly_times)!=1440: raise ContractError('R6 expected 1440 Jan-Feb hourly bars')
    data=spec['data']; fit=data['common_fit']; val=data['common_validation']
    fit_idx=r0._consequence_indices(hourly_times,r0._parse_utc_ms(fit['consequence_hour_start_utc'],'fit.start'),r0._parse_utc_ms(fit['consequence_hour_end_utc'],'fit.end'),int(fit['expected_decisions']))
    val_idx=r0._consequence_indices(hourly_times,r0._parse_utc_ms(val['consequence_hour_start_utc'],'val.start'),r0._parse_utc_ms(val['consequence_hour_end_utc'],'val.end'),int(val['expected_decisions']))
    if np.intersect1d(fit_idx,val_idx).size: raise ContractError('R6 fit/validation indices overlap')
    fit_surfaces={}; val_surfaces={}
    for c in data['contexts']:
        represented=int(c['represented_hours']); raw=int(c['raw_window_hours']); width=int(c['feature_width'])
        fit_n0=n0_endpoint_log_ratios(hourly[r0._window_indices(fit_idx,raw)]).values
        val_n0=n0_endpoint_log_ratios(hourly[r0._window_indices(val_idx,raw)]).values
        if fit_n0.shape!=(615,represented,5) or val_n0.shape!=(672,represented,5): raise ContractError(f'R6 N0 shape mismatch for {represented}h')
        fit_flat=np.asarray(fit_n0,dtype=np.float64).reshape(615,width); val_flat=np.asarray(val_n0,dtype=np.float64).reshape(672,width)
        if not np.isfinite(fit_flat).all() or not np.isfinite(val_flat).all(): raise ContractError('R6 N0 surfaces must be finite')
        fit_surfaces[represented]=fit_flat; val_surfaces[represented]=val_flat
    fit_target=r3._target(hourly[fit_idx,0],hourly[fit_idx,3]); val_target=r3._target(hourly[val_idx,0],hourly[val_idx,3])
    details=dict(evidence); details.update({'hourly_bars':1440,'common_fit_decisions':615,'common_validation_decisions':672,'context_lengths_hours':[16,32,64,128],'march_opened':False,'causal_window_relation':'each_N0_state_strictly_before_one_hour_consequence'})
    return ContextDataset(fit_surfaces,val_surfaces,fit_target,val_target,details)


def control_orders(length:int,seeds:Sequence[int])->Mapping[int,np.ndarray]:
    out={}
    for seed in seeds:
        order=np.asarray(tasks.no_fixed_point_permutation(length,seed=int(seed),generation=0),dtype=np.int64)
        if order.shape!=(length,) or np.array_equal(order,np.arange(length)): raise ContractError('R6 control permutation invalid')
        out[int(seed)]=order
    return out


def fit_predictions(dataset:ContextDataset,spec:Mapping[str,Any])->Dict[int,Dict[str,Any]]:
    screen=spec['screen']; ridge=float(screen['ridge_lambda']); seeds=[int(x) for x in screen['control_seeds']]
    orders=control_orders(len(dataset.fit_target),seeds)
    out={}
    for context in [int(x) for x in screen['context_lengths_hours']]:
        fit_x=dataset.fit_surfaces[context]; val_x=dataset.validation_surfaces[context]
        scaler=r3.fit_standardizer(fit_x)
        true_probe=r3.fit_ridge(fit_x,dataset.fit_target,ridge_lambda=ridge,standardizer=scaler)
        true_pred=true_probe.predict(val_x); ctrls=[]
        for seed in seeds:
            shuffled=np.asarray(dataset.fit_target)[orders[seed]]
            if not np.array_equal(np.sort(shuffled),np.sort(dataset.fit_target)): raise ContractError('R6 shuffled control must preserve target multiset')
            ctrls.append(r3.fit_ridge(fit_x,shuffled,ridge_lambda=ridge,standardizer=scaler).predict(val_x))
        out[context]={'true_prediction':true_pred,'control_predictions':tuple(ctrls),'active_feature_count':int(scaler.active.sum()),'input_feature_count':int(scaler.active.size)}
    return out


def _lcb(values:Sequence[float],index:int)->float:
    ordered=sorted(float(v) for v in values)
    if not 0<=index<len(ordered): raise ContractError('R6 LCB index invalid')
    return ordered[index]


def screen_contexts(dataset:ContextDataset,predictions:Mapping[int,Mapping[str,Any]],spec:Mapping[str,Any])->Dict[str,Any]:
    u=spec['uncertainty_protocol']; samples=r5.circular_block_samples(672,int(u['block_length_hours']),int(u['bootstrap_replicates']),int(u['bootstrap_seed'])); order_index=int(u['lower_bound_order_index_zero_based'])
    y=np.asarray(dataset.validation_target); ref=64; ref_pred=np.asarray(predictions[ref]['true_prediction']); ref_point=r3.pearson_corr(ref_pred,y)
    records={}
    for context in [int(x) for x in spec['screen']['context_lengths_hours']]:
        true=np.asarray(predictions[context]['true_prediction']); ctrls=[np.asarray(x) for x in predictions[context]['control_predictions']]
        point_true=r3.pearson_corr(true,y); point_ctrls=[r3.pearson_corr(c,y) for c in ctrls]
        true_bs=[]; delta_bs=[]; gain_bs=[]
        for idx in samples:
            tb=r3.pearson_corr(true[idx],y[idx]); cb=[r3.pearson_corr(c[idx],y[idx]) for c in ctrls]
            true_bs.append(tb); delta_bs.append(tb-float(np.median(cb)))
            if context!=ref: gain_bs.append(tb-r3.pearson_corr(ref_pred[idx],y[idx]))
        rec={'context_hours':context,'point':{'true_corr':point_true,'median_shuffle_corr':float(np.median(point_ctrls)),'true_minus_median_shuffle_corr':point_true-float(np.median(point_ctrls))},'bootstrap':{'true_corr_lcb':_lcb(true_bs,order_index),'true_minus_median_shuffle_corr_lcb':_lcb(delta_bs,order_index)},'active_feature_count':int(predictions[context]['active_feature_count']),'input_feature_count':int(predictions[context]['input_feature_count'])}
        if context!=ref:
            gain_point=point_true-ref_point; gain_lcb=_lcb(gain_bs,order_index)
            eligible=rec['bootstrap']['true_corr_lcb']>0 and rec['bootstrap']['true_minus_median_shuffle_corr_lcb']>0 and gain_lcb>0
            rec['point']['true_minus_64h_corr']=gain_point; rec['bootstrap']['true_minus_64h_corr_lcb']=gain_lcb; rec['screen_eligible']=eligible
        records[context]=rec
    eligible=[c for c in [16,32,128] if records[c]['screen_eligible']]
    candidate=sorted(eligible,key=lambda c:(-records[c]['point']['true_corr'],c))[0] if eligible else None
    return {'records':records,'eligible_contexts':eligible,'candidate_context_hours':candidate}


def run_experiment(spec:Mapping[str,Any],dataset:ContextDataset,implementation:Mapping[str,Any])->Dict[str,Any]:
    validate_spec(spec); gate=q.implementation_test_gate(implementation,commit=q.commit_sha())
    if not gate['passed']: raise ContractError('R6 exact-commit implementation test gate is not green')
    predictions=fit_predictions(dataset,spec); screened=screen_contexts(dataset,predictions,spec); candidate=screened['candidate_context_hours']; supported=candidate is not None
    promotion='NO_PROMOTION' if candidate is None else f'PROMOTE_CONTEXT{candidate}_CANDIDATE_HYPOTHESIS'
    result={'schema':'cb16.result.v2','experiment_id':EXPERIMENT_ID,'result_command':os.environ.get('CB16_RESULT_COMMAND',RESULT_COMMAND),'commit_sha':q.commit_sha(),'classification':'PASS' if supported else 'SCIENTIFIC_FAIL','execution_status':'EXECUTED','validity_status':'VALID','scientific_outcomes':{'context_length_screening_priority':'CANDIDATE_FOUND' if supported else 'NO_CANDIDATE'},'implementation_tests':{**dict(implementation),'gate':gate,'required_by_global_gate':True,'runner_executed_suite':True},'preregistered_spec':{'path':str(CONFIG_PATH.relative_to(REPO_ROOT)),'file_sha256':SPEC_FILE_SHA256,'declared_status':spec.get('status')},'data_evidence':{**dict(dataset.evidence),'manifest_final_holdout_accessed_post_run':False},'contexts':{str(c):r for c,r in screened['records'].items()},'screen_summary':{'eligible_contexts':screened['eligible_contexts'],'candidate_context_hours':candidate},'gate_results':[{'gate_id':'G_CONTEXT_SCREEN_CANDIDATE','passed':supported,'metrics':{'eligible_contexts':screened['eligible_contexts'],'candidate_context_hours':candidate}}],'claim_assessments':[{'claim_id':'C_CONTEXT_LENGTH_SCREENING_PRIORITY','qualification_status':'NOT_APPLICABLE','inference_status':'SUPPORTED' if supported else 'GATE_NOT_MET','attribution_status':'UNRESOLVED'}],'prior_evidence_assessment':[{'claim_ref':'r12.vs_d.bounded_horizon_accessibility_screen.r5','status':'UNAFFECTED_IN_ORIGINAL_SCOPE'},{'claim_ref':'r12.vs_d.representation_accessibility_diagnostic.r3','status':'UNAFFECTED_IN_ORIGINAL_SCOPE'}],'promotion_decision':{'decision':promotion,'reason':'R6 is a screening experiment; any candidate requires a new qualification/transfer experiment'},'provenance':{'commit_sha':q.commit_sha(),'spec_file_sha256':SPEC_FILE_SHA256,'parent_r5_run_id':34926920975,'parent_r5_artifact_digest':'sha256:eebea1c4c6b3bf9bbede409cfc72919daf15e85a51c4b76fe8d516e6e48af80e','freshness':'SCREENING_ADAPTIVELY_REUSED_JAN_FEB_AFTER_R5','march_read':False,'final_holdout_accessed':False}}
    try: validate_result_against_spec(result,spec)
    except EvidenceScopeError as exc: raise ContractError(f'R6 result violates claim authority: {exc}') from exc
    return result


def render_report(result:Mapping[str,Any])->str:
    lines=['# R12 VS-D Bounded Context-Length Accessibility Screen R6 — REPORT','',f"- classification: **{result['classification']}**",f"- candidate: **{result.get('screen_summary',{}).get('candidate_context_hours')}**",f"- promotion: **{result.get('promotion_decision',{}).get('decision','NO_PROMOTION')}**",'- role: **SCREENING — not qualification**','- data: **common Jan fit / Feb adaptively reused development; March not read**','','## Context screen','','| context | corr | shuffle median | delta | corr LCB | delta LCB | gain vs 64h | gain LCB | active/features | eligible |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for c in ['16','32','64','128']:
        r=result.get('contexts',{}).get(c,{}); p=r.get('point',{}); b=r.get('bootstrap',{})
        lines.append(f"| {c} | {p.get('true_corr',float('nan')):.9f} | {p.get('median_shuffle_corr',float('nan')):.9f} | {p.get('true_minus_median_shuffle_corr',float('nan')):.9f} | {b.get('true_corr_lcb',float('nan')):.9f} | {b.get('true_minus_median_shuffle_corr_lcb',float('nan')):.9f} | {p.get('true_minus_64h_corr',float('nan')):.9f} | {b.get('true_minus_64h_corr_lcb',float('nan')):.9f} | {r.get('active_feature_count','?')}/{r.get('input_feature_count','?')} | {r.get('screen_eligible',False)} |")
    lines += ['','## Authority boundary','','- This screen may only prioritize a context-length candidate for a new experiment.','- Different context lengths have different feature dimensions; fixed lambda does not resolve regularization equivalence.','- It does not qualify historical market information or profitability.','- A negative screen does not falsify unscreened context lengths or richer representations.','- March and final holdout remain unopened.','']
    return '\n'.join(lines)


def failure_result(spec:Mapping[str,Any]|None,classification:str,detail:str,implementation:Mapping[str,Any]|None=None)->Dict[str,Any]:
    result={'schema':'cb16.result.v2','experiment_id':EXPERIMENT_ID,'result_command':os.environ.get('CB16_RESULT_COMMAND',RESULT_COMMAND),'commit_sha':q.commit_sha(),'classification':classification,'execution_status':'NOT_EXECUTED','validity_status':'INVALID','scientific_outcomes':{},'gate_results':[],'claim_assessments':[{'claim_id':'C_CONTEXT_LENGTH_SCREENING_PRIORITY','qualification_status':'NOT_APPLICABLE','inference_status':'INVALID_FOR_CLAIM','attribution_status':'NOT_APPLICABLE'}],'prior_evidence_assessment':[],'promotion_decision':{'decision':'NO_PROMOTION','reason':'formal run did not reach valid screening adjudication'},'provenance':{'commit_sha':q.commit_sha(),'spec_file_sha256':SPEC_FILE_SHA256},'error':{'type':classification,'detail':detail[:2000]}}
    if implementation is not None: result['implementation_tests']=dict(implementation)
    if spec is not None:
        try: validate_result_against_spec(result,spec)
        except EvidenceScopeError as exc: raise ContractError(f'R6 failure result violates claim authority: {exc}') from exc
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
    print(f"{RESULT_COMMAND}: classification={result['classification']} candidate={result['screen_summary']['candidate_context_hours']}"); return code

if __name__=='__main__': raise SystemExit(main())
