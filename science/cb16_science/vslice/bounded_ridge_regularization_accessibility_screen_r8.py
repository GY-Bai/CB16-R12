"""R12 VS-D R8 bounded ridge-regularization accessibility screen."""
from __future__ import annotations

import hashlib, json, math, os
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

RESULT_COMMAND="cb16.vs-d-bounded-ridge-regularization-accessibility-screen-r8@v1"
EXPERIMENT_ID="r12.vs_d.bounded_ridge_regularization_accessibility_screen.r8"
REPO_ROOT=Path(__file__).resolve().parents[3]
CONFIG_PATH=REPO_ROOT/'config/experiments/r12_vs_d_bounded_ridge_regularization_accessibility_screen_r8.json'
SPEC_FILE_SHA256='8e6b51a30b4f12ebeaf6ec8b6d4ad0092ada445e9cb1c1721d139e3fb12f6d50'
SPEC_FILENAME,RESULT_FILENAME,REPORT_FILENAME='experiment_spec.json','RESULT.json','REPORT.md'
EXIT_PASS,EXIT_SCIENTIFIC_FAIL,EXIT_EXECUTION_BLOCKED,EXIT_CONTRACT_MISMATCH=0,1,3,4

@dataclass(frozen=True)
class RidgeDataset:
    fit_x: np.ndarray
    validation_x: np.ndarray
    fit_target: np.ndarray
    validation_target: np.ndarray
    evidence: Mapping[str,Any]


def load_spec()->Dict[str,Any]:
    raw=CONFIG_PATH.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest!=SPEC_FILE_SHA256: raise ContractError(f'R8 spec byte SHA mismatch: {digest}')
    try: return dict(tasks.spec_mapping(json.loads(raw),'R8 spec'))
    except json.JSONDecodeError as exc: raise ContractError(f'R8 spec invalid JSON: {exc}') from exc


def validate_spec(spec:Mapping[str,Any])->None:
    try: validate_experiment_spec(spec)
    except EvidenceScopeError as exc: raise ContractError(f'R8 claim-authority spec invalid: {exc}') from exc
    if spec.get('experiment_id')!=EXPERIMENT_ID or spec.get('experiment_role')!='SCREENING': raise ContractError('R8 identity/role mismatch')
    data=tasks.spec_mapping(spec['data'],'data'); names=[tasks.spec_str(x['name'],'archive.name') for x in data['allowed_archives']]
    if names!=['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip']: raise ContractError('R8 allowed archives must be Jan-Feb')
    if int(data['hourly_aggregation']['expected_hourly_bars'])!=1440: raise ContractError('R8 hourly count mismatch')
    if int(data['fit']['expected_decisions'])!=615 or int(data['validation']['expected_decisions'])!=672: raise ContractError('R8 fit/validation counts mismatch')
    if (int(data['context']['represented_hours']),int(data['context']['raw_window_hours']),int(data['context']['feature_width']))!=(64,65,320): raise ContractError('R8 context mismatch')
    s=tasks.spec_mapping(spec['screen'],'screen')
    if [float(x) for x in s['ridge_lambdas']]!=[0.1,1.0,10.0,100.0,1000.0] or float(s['reference_lambda'])!=1.0: raise ContractError('R8 lambda grid/reference mismatch')
    if [float(x) for x in s['candidate_lambdas']]!=[0.1,10.0,100.0,1000.0]: raise ContractError('R8 candidate lambdas mismatch')
    if tuple(int(x) for x in s['control_seeds'])!=tuple(range(21001,21009)) or not bool(s['same_target_permutation_across_lambdas']): raise ContractError('R8 controls mismatch')
    u=tasks.spec_mapping(spec['uncertainty_protocol'],'uncertainty_protocol')
    if (int(u['bootstrap_replicates']),int(u['bootstrap_seed']),int(u['block_length_hours']),int(u['sample_length_hours']),int(u['lower_bound_order_index_zero_based']))!=(4096,22001,48,672,204): raise ContractError('R8 bootstrap mismatch')


def load_dataset(spec:Mapping[str,Any])->RidgeDataset:
    times,minutes,evidence=r0._read_allowed_minutes(spec); hourly_times,hourly=r0.aggregate_utc_hourly(times,minutes)
    if len(hourly_times)!=1440: raise ContractError('R8 expected 1440 Jan-Feb hourly bars')
    data=spec['data']; fit=data['fit']; val=data['validation']; raw=int(data['context']['raw_window_hours'])
    fit_idx=r0._consequence_indices(hourly_times,r0._parse_utc_ms(fit['consequence_hour_start_utc'],'fit.start'),r0._parse_utc_ms(fit['consequence_hour_end_utc'],'fit.end'),int(fit['expected_decisions']))
    val_idx=r0._consequence_indices(hourly_times,r0._parse_utc_ms(val['consequence_hour_start_utc'],'val.start'),r0._parse_utc_ms(val['consequence_hour_end_utc'],'val.end'),int(val['expected_decisions']))
    fit_n0=n0_endpoint_log_ratios(hourly[r0._window_indices(fit_idx,raw)]).values; val_n0=n0_endpoint_log_ratios(hourly[r0._window_indices(val_idx,raw)]).values
    if fit_n0.shape!=(615,64,5) or val_n0.shape!=(672,64,5): raise ContractError('R8 N0 shapes mismatch')
    fit_x=np.asarray(fit_n0,dtype=np.float64).reshape(615,320); val_x=np.asarray(val_n0,dtype=np.float64).reshape(672,320)
    fit_target=r3._target(hourly[fit_idx,0],hourly[fit_idx,3]); val_target=r3._target(hourly[val_idx,0],hourly[val_idx,3])
    details=dict(evidence); details.update({'hourly_bars':1440,'fit_decisions':615,'validation_decisions':672,'march_opened':False,'causal_window_relation':'64h_N0_strictly_before_one_hour_consequence'})
    return RidgeDataset(fit_x,val_x,fit_target,val_target,details)


def control_orders(length:int,seeds:Sequence[int])->Mapping[int,np.ndarray]:
    out={}
    for seed in seeds:
        order=np.asarray(tasks.no_fixed_point_permutation(length,seed=int(seed),generation=0),dtype=np.int64)
        if order.shape!=(length,) or np.any(order==np.arange(length)): raise ContractError('R8 control permutation invalid')
        out[int(seed)]=order
    return out


def fit_predictions(dataset:RidgeDataset,spec:Mapping[str,Any])->Dict[float,Dict[str,Any]]:
    s=spec['screen']; seeds=[int(x) for x in s['control_seeds']]; orders=control_orders(len(dataset.fit_target),seeds); scaler=r3.fit_standardizer(dataset.fit_x); out={}
    for lam in [float(x) for x in s['ridge_lambdas']]:
        true=r3.fit_ridge(dataset.fit_x,dataset.fit_target,ridge_lambda=lam,standardizer=scaler).predict(dataset.validation_x); ctrls=[]
        for seed in seeds:
            shuffled=np.asarray(dataset.fit_target)[orders[seed]]
            if not np.array_equal(np.sort(shuffled),np.sort(dataset.fit_target)): raise ContractError('R8 shuffled target multiset mismatch')
            ctrls.append(r3.fit_ridge(dataset.fit_x,shuffled,ridge_lambda=lam,standardizer=scaler).predict(dataset.validation_x))
        out[lam]={'true_prediction':true,'control_predictions':tuple(ctrls),'active_feature_count':int(scaler.active.sum()),'input_feature_count':int(scaler.active.size)}
    return out


def _lcb(values:Sequence[float],index:int)->float:
    ordered=sorted(float(x) for x in values)
    if not 0<=index<len(ordered): raise ContractError('R8 LCB index invalid')
    return ordered[index]


def screen_lambdas(dataset:RidgeDataset,predictions:Mapping[float,Mapping[str,Any]],spec:Mapping[str,Any])->Dict[str,Any]:
    u=spec['uncertainty_protocol']; samples=r5.circular_block_samples(672,int(u['block_length_hours']),int(u['bootstrap_replicates']),int(u['bootstrap_seed'])); oi=int(u['lower_bound_order_index_zero_based'])
    y=np.asarray(dataset.validation_target); ref=1.0; ref_pred=np.asarray(predictions[ref]['true_prediction']); ref_point=r3.pearson_corr(ref_pred,y); records={}
    for lam in [float(x) for x in spec['screen']['ridge_lambdas']]:
        true=np.asarray(predictions[lam]['true_prediction']); ctrls=[np.asarray(x) for x in predictions[lam]['control_predictions']]
        point_true=r3.pearson_corr(true,y); point_ctrls=[r3.pearson_corr(c,y) for c in ctrls]; true_bs=[]; delta_bs=[]; gain_bs=[]
        for idx in samples:
            tb=r3.pearson_corr(true[idx],y[idx]); cb=[r3.pearson_corr(c[idx],y[idx]) for c in ctrls]; true_bs.append(tb); delta_bs.append(tb-float(np.median(cb)))
            if lam!=ref: gain_bs.append(tb-r3.pearson_corr(ref_pred[idx],y[idx]))
        rec={'lambda':lam,'point':{'true_corr':point_true,'median_shuffle_corr':float(np.median(point_ctrls)),'true_minus_median_shuffle_corr':point_true-float(np.median(point_ctrls))},'bootstrap':{'true_corr_lcb':_lcb(true_bs,oi),'true_minus_median_shuffle_corr_lcb':_lcb(delta_bs,oi)},'active_feature_count':int(predictions[lam]['active_feature_count']),'input_feature_count':int(predictions[lam]['input_feature_count'])}
        if lam!=ref:
            gain=point_true-ref_point; gain_lcb=_lcb(gain_bs,oi); eligible=rec['bootstrap']['true_corr_lcb']>0 and rec['bootstrap']['true_minus_median_shuffle_corr_lcb']>0 and gain_lcb>0
            rec['point']['true_minus_lambda1_corr']=gain; rec['bootstrap']['true_minus_lambda1_corr_lcb']=gain_lcb; rec['screen_eligible']=eligible
        records[lam]=rec
    eligible=[lam for lam in [0.1,10.0,100.0,1000.0] if records[lam]['screen_eligible']]
    candidate=sorted(eligible,key=lambda lam:(-records[lam]['point']['true_corr'],abs(math.log10(lam/ref)),lam))[0] if eligible else None
    return {'records':records,'eligible_lambdas':eligible,'candidate_lambda':candidate}


def _promotion(candidate:float|None)->str:
    if candidate is None: return 'NO_PROMOTION'
    return {0.1:'PROMOTE_RIDGE_LAMBDA_0P1_CANDIDATE_HYPOTHESIS',10.0:'PROMOTE_RIDGE_LAMBDA_10_CANDIDATE_HYPOTHESIS',100.0:'PROMOTE_RIDGE_LAMBDA_100_CANDIDATE_HYPOTHESIS',1000.0:'PROMOTE_RIDGE_LAMBDA_1000_CANDIDATE_HYPOTHESIS'}[candidate]


def run_experiment(spec:Mapping[str,Any],dataset:RidgeDataset,implementation:Mapping[str,Any])->Dict[str,Any]:
    validate_spec(spec); gate=q.implementation_test_gate(implementation,commit=q.commit_sha())
    if not gate['passed']: raise ContractError('R8 exact-commit implementation test gate is not green')
    screened=screen_lambdas(dataset,fit_predictions(dataset,spec),spec); candidate=screened['candidate_lambda']; supported=candidate is not None; promotion=_promotion(candidate)
    result={'schema':'cb16.result.v2','experiment_id':EXPERIMENT_ID,'result_command':os.environ.get('CB16_RESULT_COMMAND',RESULT_COMMAND),'commit_sha':q.commit_sha(),'classification':'PASS' if supported else 'SCIENTIFIC_FAIL','execution_status':'EXECUTED','validity_status':'VALID','scientific_outcomes':{'ridge_regularization_screening_priority':'CANDIDATE_FOUND' if supported else 'NO_CANDIDATE'},'implementation_tests':{**dict(implementation),'gate':gate,'required_by_global_gate':True,'runner_executed_suite':True},'preregistered_spec':{'path':str(CONFIG_PATH.relative_to(REPO_ROOT)),'file_sha256':SPEC_FILE_SHA256,'declared_status':spec.get('status')},'data_evidence':{**dict(dataset.evidence),'manifest_final_holdout_accessed_post_run':False},'lambdas':{str(lam):r for lam,r in screened['records'].items()},'screen_summary':{'eligible_lambdas':screened['eligible_lambdas'],'candidate_lambda':candidate},'gate_results':[{'gate_id':'G_RIDGE_SCALE_SCREEN_CANDIDATE','passed':supported,'metrics':{'eligible_lambdas':screened['eligible_lambdas'],'candidate_lambda':candidate}}],'claim_assessments':[{'claim_id':'C_RIDGE_REGULARIZATION_SCREENING_PRIORITY','qualification_status':'NOT_APPLICABLE','inference_status':'SUPPORTED' if supported else 'GATE_NOT_MET','attribution_status':'UNRESOLVED'}],'prior_evidence_assessment':[{'claim_ref':'r12.vs_d.bounded_context_length_accessibility_screen.r6','status':'UNAFFECTED_IN_ORIGINAL_SCOPE'},{'claim_ref':'r12.vs_d.march_temporal_robustness_of_64h_linear_hint.r7','status':'UNAFFECTED_IN_ORIGINAL_SCOPE'}],'promotion_decision':{'decision':promotion,'reason':'R8 is an adaptively reused regularization screen; any candidate requires a new robustness/transfer experiment'},'provenance':{'commit_sha':q.commit_sha(),'spec_file_sha256':SPEC_FILE_SHA256,'parent_r7_run_id':34929543927,'parent_r7_artifact_digest':'sha256:32d82860cb5a970a1b5e4fc8e9e0079f066856bbd8fca507a28ef74bdd0a3491','freshness':'SCREENING_ADAPTIVELY_REUSED_JAN_FEB_AFTER_R6_R7','march_read':False,'final_holdout_accessed':False}}
    try: validate_result_against_spec(result,spec)
    except EvidenceScopeError as exc: raise ContractError(f'R8 result violates claim authority: {exc}') from exc
    return result


def render_report(result:Mapping[str,Any])->str:
    lines=['# R12 VS-D Bounded Ridge Regularization Accessibility Screen R8 — REPORT','',f"- classification: **{result['classification']}**",f"- candidate lambda: **{result.get('screen_summary',{}).get('candidate_lambda')}**",f"- promotion: **{result.get('promotion_decision',{}).get('decision','NO_PROMOTION')}**",'- role: **SCREENING — not qualification**','- data: **Jan fit / Feb adaptively reused validation; March not read**','','## Lambda screen','','| lambda | corr | shuffle median | delta | corr LCB | delta LCB | gain vs lambda=1 | gain LCB | eligible |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for key in ['0.1','1.0','10.0','100.0','1000.0']:
        r=result.get('lambdas',{}).get(key,{}); p=r.get('point',{}); b=r.get('bootstrap',{})
        lines.append(f"| {key} | {p.get('true_corr',float('nan')):.9f} | {p.get('median_shuffle_corr',float('nan')):.9f} | {p.get('true_minus_median_shuffle_corr',float('nan')):.9f} | {b.get('true_corr_lcb',float('nan')):.9f} | {b.get('true_minus_median_shuffle_corr_lcb',float('nan')):.9f} | {p.get('true_minus_lambda1_corr',float('nan')):.9f} | {b.get('true_minus_lambda1_corr_lcb',float('nan')):.9f} | {r.get('screen_eligible',False)} |")
    lines += ['','## Authority boundary','','- This screen may only prioritize a ridge-lambda candidate for a new experiment.','- February is adaptively reused and not confirmatory.','- PASS does not establish optimal regularization or historical-market qualification.','- FAIL does not exclude other probe families or identify a unique root cause.','- March and final holdout remain unopened.','']
    return '\n'.join(lines)


def failure_result(spec:Mapping[str,Any]|None,classification:str,detail:str,implementation:Mapping[str,Any]|None=None)->Dict[str,Any]:
    result={'schema':'cb16.result.v2','experiment_id':EXPERIMENT_ID,'result_command':os.environ.get('CB16_RESULT_COMMAND',RESULT_COMMAND),'commit_sha':q.commit_sha(),'classification':classification,'execution_status':'NOT_EXECUTED','validity_status':'INVALID','scientific_outcomes':{},'gate_results':[],'claim_assessments':[{'claim_id':'C_RIDGE_REGULARIZATION_SCREENING_PRIORITY','qualification_status':'NOT_APPLICABLE','inference_status':'INVALID_FOR_CLAIM','attribution_status':'NOT_APPLICABLE'}],'prior_evidence_assessment':[],'promotion_decision':{'decision':'NO_PROMOTION','reason':'formal run did not reach valid screening adjudication'},'provenance':{'commit_sha':q.commit_sha(),'spec_file_sha256':SPEC_FILE_SHA256},'error':{'type':classification,'detail':detail[:2000]}}
    if implementation is not None: result['implementation_tests']=dict(implementation)
    if spec is not None:
        try: validate_result_against_spec(result,spec)
        except EvidenceScopeError as exc: raise ContractError(f'R8 failure result violates claim authority: {exc}') from exc
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
    print(f"{RESULT_COMMAND}: classification={result['classification']} candidate={result['screen_summary']['candidate_lambda']}"); return code

if __name__=='__main__': raise SystemExit(main())
