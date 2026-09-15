"""R12 VS-D R4 bounded nonlinear accessibility diagnostic."""
from __future__ import annotations

import hashlib, json, math, os, sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from . import controlled_tasks as tasks
from . import qualification as q
from . import representation_accessibility_r3 as r3
from .contracts import ContractError

RESULT_COMMAND = "cb16.vs-d-bounded-nonlinear-accessibility-r4@v1"
EXPERIMENT_ID = "r12.vs_d.bounded_nonlinear_accessibility_diagnostic.r4"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_d_bounded_nonlinear_accessibility_r4.json"
SPEC_FILE_SHA256 = "68e3c09f605625fd89e037aac4ad1bdf9a876e330ed984b490240dd12a472659"
SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME = "experiment_spec.json", "RESULT.json", "REPORT.md"
EXIT_PASS, EXIT_SCIENTIFIC_FAIL, EXIT_EXECUTION_BLOCKED, EXIT_CONTRACT_MISMATCH = 0, 1, 3, 4


def load_spec() -> Dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPEC_FILE_SHA256:
        raise ContractError(f"R4 spec byte SHA mismatch: {digest}")
    try:
        return dict(tasks.spec_mapping(json.loads(raw), "R4 spec"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"R4 spec is invalid JSON: {exc}") from exc


def validate_spec(spec: Mapping[str, Any]) -> None:
    try:
        validate_experiment_spec(spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R4 claim-authority spec invalid: {exc}") from exc
    if spec.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("R4 experiment_id mismatch")
    data = tasks.spec_mapping(spec["data"], "data")
    names = [tasks.spec_str(x["name"], "archive.name") for x in data["allowed_archives"]]
    if names != ["BTCUSDT-1m-2020-01.zip", "BTCUSDT-1m-2020-02.zip"]:
        raise ContractError("R4 allowed archives must be exactly Jan-Feb")
    if tasks.spec_int(data["hourly_aggregation"]["expected_hourly_bars"], "hourly") != 1440:
        raise ContractError("R4 hourly count must be 1440")
    if tasks.spec_int(data["probe_fit"]["expected_decisions"], "fit") != 679:
        raise ContractError("R4 January fit count must be 679")
    if tasks.spec_int(data["probe_validation"]["expected_decisions"], "validation") != 696:
        raise ContractError("R4 February validation count must be 696")
    if tasks.spec_int(data["probe_validation"]["expected_day_blocks"], "day_blocks") != 29:
        raise ContractError("R4 requires 29 February day blocks")
    probe = tasks.spec_mapping(spec["nonlinear_probe"], "nonlinear_probe")
    expected_arch = ["Linear(active_features,64)", "Tanh", "Linear(64,64)", "Tanh", "Linear(64,1)"]
    if list(probe["architecture"]) != expected_arch or tasks.spec_int(probe["epochs"], "epochs") != 256:
        raise ContractError("R4 frozen MLP architecture/epochs mismatch")
    if tasks.spec_float(probe["learning_rate"], "lr") != 0.001 or probe["optimizer"] != "Adam":
        raise ContractError("R4 frozen optimizer mismatch")
    if tuple(int(x) for x in probe["model_seeds"]) != tuple(range(4501, 4509)):
        raise ContractError("R4 model seeds mismatch")
    control = tasks.spec_mapping(spec["negative_control"], "negative_control")
    if tuple(int(x) for x in control["control_permutation_seeds"]) != tuple(range(14501, 14509)):
        raise ContractError("R4 control seeds mismatch")
    if not bool(control["matched_model_initialization"]):
        raise ContractError("R4 requires matched model initialization")
    uncertainty = tasks.spec_mapping(spec["uncertainty_protocol"], "uncertainty_protocol")
    if tasks.spec_int(uncertainty["bootstrap_replicates"], "replicates") != 4096:
        raise ContractError("R4 freezes 4096 bootstrap replicates")
    if tasks.spec_int(uncertainty["lower_bound_order_index_zero_based"], "lcb") != 204:
        raise ContractError("R4 freezes LCB index 204")
    if tuple(int(x) for x in uncertainty["bootstrap_seeds"]) != tuple(range(15401, 15409)):
        raise ContractError("R4 bootstrap seeds mismatch")
    if tasks.spec_float(spec["linear_baseline"]["ridge_lambda"], "ridge") != 1.0:
        raise ContractError("R4 linear baseline must reproduce R3 lambda=1")


def _flatten(dataset: r3.ProbeDataset) -> Tuple[np.ndarray, np.ndarray]:
    fit = dataset.fit_market.reshape(679, 320)
    val = dataset.validation_market.reshape(696, 320)
    if not np.isfinite(fit).all() or not np.isfinite(val).all():
        raise ContractError("R4 N0 surface must be finite")
    return fit, val


def _standardized_target(target: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(target, dtype=np.float64)
    mean, std = float(y.mean()), float(y.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        raise ContractError("R4 January target variance is degenerate")
    out = (y - mean) / std
    return out.astype(np.float32), mean, std


def _build_model(width: int, seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(width,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
    return model.to(dtype=torch.float32, device="cpu")


def _train_model(model: nn.Module, x: np.ndarray, y_std: np.ndarray, spec: Mapping[str, Any]) -> None:
    p = tasks.spec_mapping(spec["nonlinear_probe"], "nonlinear_probe")
    torch.set_num_threads(tasks.spec_int(p["torch_num_threads"], "threads"))
    torch.use_deterministic_algorithms(bool(p["deterministic_algorithms"]))
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_std, dtype=np.float32)).reshape(-1,1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p["learning_rate"]), betas=tuple(float(v) for v in p["betas"]), eps=float(p["eps"]), weight_decay=float(p["weight_decay"]))
    for _ in range(tasks.spec_int(p["epochs"], "epochs")):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(xt) - yt) ** 2)
        if not bool(torch.isfinite(loss)):
            raise ContractError("R4 MLP loss became non-finite")
        loss.backward(); optimizer.step()


def _predict(model: nn.Module, x: np.ndarray, mean: float, std: float) -> np.ndarray:
    with torch.no_grad():
        pred = model(torch.from_numpy(np.asarray(x, dtype=np.float32))).reshape(-1).cpu().numpy().astype(np.float64)
    pred = mean + std * pred
    if not np.isfinite(pred).all():
        raise ContractError("R4 MLP prediction must be finite")
    return pred


def fit_seed_predictions(fit_x: np.ndarray, val_x: np.ndarray, fit_target: np.ndarray, spec: Mapping[str, Any], model_seed: int, control_seed: int) -> Dict[str, Any]:
    scaler = r3.fit_standardizer(fit_x)
    xs_fit, xs_val = scaler.transform(fit_x), scaler.transform(val_x)
    y_std, mean, std = _standardized_target(fit_target)
    true_model = _build_model(xs_fit.shape[1], model_seed)
    control_model = _build_model(xs_fit.shape[1], model_seed)
    for a,b in zip(true_model.parameters(), control_model.parameters()):
        if not torch.equal(a.detach(), b.detach()):
            raise ContractError("R4 matched models must initialize bitwise identically")
    order = np.asarray(tasks.no_fixed_point_permutation(len(y_std), seed=control_seed, generation=0), dtype=np.int64)
    shuffled = y_std[order]
    if not np.array_equal(np.sort(shuffled), np.sort(y_std)):
        raise ContractError("R4 shuffled target must preserve January target multiset")
    _train_model(true_model, xs_fit, y_std, spec)
    _train_model(control_model, xs_fit, shuffled, spec)
    return {
        "true_prediction": _predict(true_model, xs_val, mean, std),
        "control_prediction": _predict(control_model, xs_val, mean, std),
        "active_feature_count": int(scaler.active.sum()),
        "input_feature_count": int(scaler.active.size),
    }


def _lcb(values: Sequence[float], index: int) -> float:
    ordered = sorted(float(v) for v in values)
    return ordered[index]


def evaluate_seed(dataset: r3.ProbeDataset, true_pred: np.ndarray, control_pred: np.ndarray, linear_pred: np.ndarray, bootstrap_seed: int, spec: Mapping[str, Any]) -> Dict[str, Any]:
    u = tasks.spec_mapping(spec["uncertainty_protocol"], "uncertainty_protocol")
    samples = r3.bootstrap_indices(dataset.validation_day_index, replicates=int(u["bootstrap_replicates"]), seed=bootstrap_seed)
    lcb_index = int(u["lower_bound_order_index_zero_based"])
    true_vals, ctl_delta, linear_delta = [], [], []
    for idx in samples:
        y = dataset.validation_target[idx]
        t = r3.pearson_corr(true_pred[idx], y)
        c = r3.pearson_corr(control_pred[idx], y)
        l = r3.pearson_corr(linear_pred[idx], y)
        true_vals.append(t); ctl_delta.append(t-c); linear_delta.append(t-l)
    point_true = r3.pearson_corr(true_pred, dataset.validation_target)
    point_ctl = r3.pearson_corr(control_pred, dataset.validation_target)
    point_linear = r3.pearson_corr(linear_pred, dataset.validation_target)
    return {
        "point": {"mlp_true_corr": point_true, "matched_control_corr": point_ctl, "r3_linear_corr": point_linear, "true_minus_control_corr": point_true-point_ctl, "true_minus_linear_corr": point_true-point_linear},
        "bootstrap": {"mlp_true_corr_lcb": _lcb(true_vals,lcb_index), "true_minus_control_corr_lcb": _lcb(ctl_delta,lcb_index), "true_minus_linear_corr_lcb": _lcb(linear_delta,lcb_index)},
    }


def _median(records: Sequence[Mapping[str, Any]], path: tuple[str,str]) -> float:
    return float(np.median([float(r[path[0]][path[1]]) for r in records]))


def run_experiment(spec: Mapping[str, Any], dataset: r3.ProbeDataset, implementation_tests: Mapping[str, Any]) -> Dict[str, Any]:
    validate_spec(spec)
    test_gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not test_gate["passed"]:
        raise ContractError("R4 exact-commit implementation test gate is not green")
    fit_x, val_x = _flatten(dataset)
    linear = r3.fit_ridge(fit_x, dataset.fit_target, ridge_lambda=1.0)
    linear_pred = linear.predict(val_x)
    probe = spec["nonlinear_probe"]; control = spec["negative_control"]; uncertainty = spec["uncertainty_protocol"]
    records=[]
    for mseed,cseed,bseed in zip(probe["model_seeds"], control["control_permutation_seeds"], uncertainty["bootstrap_seeds"]):
        fitted = fit_seed_predictions(fit_x,val_x,dataset.fit_target,spec,int(mseed),int(cseed))
        evaluated = evaluate_seed(dataset,fitted["true_prediction"],fitted["control_prediction"],linear_pred,int(bseed),spec)
        b=evaluated["bootstrap"]
        nonlinear_pass = b["mlp_true_corr_lcb"]>0 and b["true_minus_control_corr_lcb"]>0
        gain_pass = b["true_minus_linear_corr_lcb"]>0
        records.append({"model_seed":int(mseed),"control_seed":int(cseed),"bootstrap_seed":int(bseed),"nonlinear_seed_pass":nonlinear_pass,"gain_seed_pass":gain_pass,**evaluated,"active_feature_count":fitted["active_feature_count"],"input_feature_count":fitted["input_feature_count"]})
    nonlinear_count=sum(r["nonlinear_seed_pass"] for r in records); gain_count=sum(r["gain_seed_pass"] for r in records)
    median_true=_median(records,("point","mlp_true_corr")); median_ctl=_median(records,("point","true_minus_control_corr")); median_gain=_median(records,("point","true_minus_linear_corr"))
    nonlinear_pass = nonlinear_count>=6 and median_true>0 and median_ctl>0
    gain_pass = gain_count>=6 and median_gain>0
    promotion = "PROMOTE_SUPERVISED_SENSORY_CANDIDATE_HYPOTHESIS" if nonlinear_pass and gain_pass else ("PROMOTE_BOUNDED_NONLINEAR_ACCESSIBILITY_HYPOTHESIS" if nonlinear_pass else "NO_PROMOTION")
    classification = "PASS" if (nonlinear_pass or gain_pass) else "SCIENTIFIC_FAIL"
    result={
      "schema":"cb16.result.v2","experiment_id":EXPERIMENT_ID,"result_command":os.environ.get("CB16_RESULT_COMMAND",RESULT_COMMAND),"commit_sha":q.commit_sha(),"classification":classification,"execution_status":"EXECUTED","validity_status":"VALID",
      "scientific_outcomes":{"n0_bounded_nonlinear_accessibility":"QUALIFIED" if nonlinear_pass else "NOT_QUALIFIED","nonlinear_over_linear_gain":"QUALIFIED" if gain_pass else "NOT_QUALIFIED"},
      "implementation_tests":{**dict(implementation_tests),"gate":test_gate,"required_by_global_gate":True,"runner_executed_suite":True},
      "preregistered_spec":{"path":str(CONFIG_PATH.relative_to(REPO_ROOT)),"file_sha256":SPEC_FILE_SHA256,"declared_status":spec.get("status")},
      "data_evidence":{**dict(dataset.evidence),"manifest_final_holdout_accessed_post_run":False},
      "linear_baseline":{"pearson_corr":r3.pearson_corr(linear_pred,dataset.validation_target),"active_feature_count":int(linear.standardizer.active.sum())},
      "seeds":records,
      "aggregate":{"nonlinear_seed_pass_count":nonlinear_count,"gain_seed_pass_count":gain_count,"median_mlp_true_corr":median_true,"median_true_minus_control_corr":median_ctl,"median_true_minus_linear_corr":median_gain},
      "gate_results":[
        {"gate_id":"G_NONLINEAR_SEED","passed":nonlinear_count>=6,"metrics":{"seed_pass_count":nonlinear_count}},
        {"gate_id":"G_NONLINEAR_AGGREGATE","passed":nonlinear_pass,"metrics":{"median_MLP_true_corr":median_true,"median_MLP_true_minus_control_corr":median_ctl,"seed_pass_count":nonlinear_count}},
        {"gate_id":"G_GAIN_SEED","passed":gain_count>=6,"metrics":{"seed_pass_count":gain_count}},
        {"gate_id":"G_GAIN_AGGREGATE","passed":gain_pass,"metrics":{"median_MLP_true_minus_R3_linear_corr":median_gain,"seed_pass_count":gain_count}},
      ],
      "claim_assessments":[
        {"claim_id":"C_N0_BOUNDED_NONLINEAR_ACCESSIBILITY","qualification_status":"QUALIFIED" if nonlinear_pass else "NOT_QUALIFIED","inference_status":"SUPPORTED" if nonlinear_pass else "GATE_NOT_MET","attribution_status":"UNRESOLVED"},
        {"claim_id":"C_NONLINEAR_OVER_LINEAR_GAIN","qualification_status":"QUALIFIED" if gain_pass else "NOT_QUALIFIED","inference_status":"SUPPORTED" if gain_pass else "GATE_NOT_MET","attribution_status":"UNRESOLVED"},
      ],
      "prior_evidence_assessment":[
        {"claim_ref":"r12.vs_d.historical_market_information_canary.r0","status":"UNAFFECTED_IN_ORIGINAL_SCOPE"},
        {"claim_ref":"r12.vs_d.evaluation_adapter_transfer_diagnostic.r1","status":"UNAFFECTED_IN_ORIGINAL_SCOPE"},
        {"claim_ref":"r12.vs_d.objective_balance_entropy_gradient_diagnostic.r2a","status":"UNAFFECTED_IN_ORIGINAL_SCOPE"},
        {"claim_ref":"r12.vs_d.representation_accessibility_diagnostic.r3","status":"UNAFFECTED_IN_ORIGINAL_SCOPE"},
      ],
      "promotion_decision":{"decision":promotion,"reason":"promotion is limited to the preregistered reused-development nonlinear-accessibility route"},
      "provenance":{"commit_sha":q.commit_sha(),"spec_file_sha256":SPEC_FILE_SHA256,"parent_r3_run_id":34923821507,"parent_r3_artifact_digest":"sha256:d2cc0b8d9c2ed67ba6961bcce36150a9bac7b38e857525cf7afc42cb065f62c2","freshness":"DIAGNOSTIC_REUSED_JAN_FEB_AFTER_R3_INSPECTION","march_read":False,"final_holdout_accessed":False},
    }
    try: validate_result_against_spec(result,spec)
    except EvidenceScopeError as exc: raise ContractError(f"R4 result violates claim authority: {exc}") from exc
    return result


def render_report(result: Mapping[str, Any]) -> str:
    a=result.get("aggregate",{})
    lines=["# R12 VS-D Bounded Nonlinear Accessibility R4 — REPORT","",f"- classification: **{result['classification']}**",f"- nonlinear accessibility: **{result.get('scientific_outcomes',{}).get('n0_bounded_nonlinear_accessibility','NOT_EVALUATED')}**",f"- nonlinear-over-linear gain: **{result.get('scientific_outcomes',{}).get('nonlinear_over_linear_gain','NOT_EVALUATED')}**",f"- promotion: **{result.get('promotion_decision',{}).get('decision','NO_PROMOTION')}**","- data: **January fit / February adaptively reused development; March not read**","","## Aggregate",""]
    for k,v in a.items(): lines.append(f"- {k}: `{v}`")
    if "seeds" in result:
        lines += ["","## Per-seed","","| seed | nonlinear | gain | true corr | control delta | linear delta |","| --- | --- | --- | ---: | ---: | ---: |"]
        for r in result["seeds"]:
            p=r["point"]; lines.append(f"| {r['model_seed']} | {'PASS' if r['nonlinear_seed_pass'] else 'FAIL'} | {'PASS' if r['gain_seed_pass'] else 'FAIL'} | {p['mlp_true_corr']:.9f} | {p['true_minus_control_corr']:.9f} | {p['true_minus_linear_corr']:.9f} |")
    lines += ["","## Authority boundary","","- This is adaptively reused Jan-Feb development, not fresh confirmation.","- PASS qualifies only this bounded probe relation and declared promotion.","- FAIL does not imply general nonlinear inaccessibility or absence of market information.","- March and final holdout remain unopened.",""]
    return "\n".join(lines)


def failure_result(spec: Mapping[str,Any]|None, classification: str, detail: str, implementation_tests: Mapping[str,Any]|None=None) -> Dict[str,Any]:
    claim_ids=[str(c["claim_id"]) for c in spec["claims"]] if spec else ["C_N0_BOUNDED_NONLINEAR_ACCESSIBILITY","C_NONLINEAR_OVER_LINEAR_GAIN"]
    result={"schema":"cb16.result.v2","experiment_id":EXPERIMENT_ID,"result_command":os.environ.get("CB16_RESULT_COMMAND",RESULT_COMMAND),"commit_sha":q.commit_sha(),"classification":classification,"execution_status":"NOT_EXECUTED","validity_status":"INVALID","scientific_outcomes":{},"gate_results":[],"claim_assessments":[{"claim_id":c,"qualification_status":"NOT_APPLICABLE","inference_status":"INVALID_FOR_CLAIM","attribution_status":"NOT_APPLICABLE"} for c in claim_ids],"prior_evidence_assessment":[],"promotion_decision":{"decision":"NO_PROMOTION","reason":"formal run did not reach valid scientific adjudication"},"provenance":{"commit_sha":q.commit_sha(),"spec_file_sha256":SPEC_FILE_SHA256},"error":{"type":classification,"detail":detail[:2000]}}
    if implementation_tests is not None: result["implementation_tests"]=dict(implementation_tests)
    if spec is not None:
        try: validate_result_against_spec(result,spec)
        except EvidenceScopeError as exc: raise ContractError(f"R4 failure result violates claim authority: {exc}") from exc
    return result


def write_artifacts(result_dir: Path, spec: Mapping[str,Any], result: Mapping[str,Any]) -> None:
    result_dir.mkdir(parents=True,exist_ok=True)
    (result_dir/SPEC_FILENAME).write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    (result_dir/RESULT_FILENAME).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (result_dir/REPORT_FILENAME).write_text(render_report(result))


def run_qualification(result_dir: Path) -> Tuple[Dict[str,Any],Dict[str,Any],int]:
    spec=load_spec(); validate_spec(spec)
    implementation=q.run_implementation_test_suite(); gate=q.implementation_test_gate(implementation,commit=q.commit_sha())
    if not gate["passed"]:
        classification,exit_code=q.implementation_failure_classification(implementation)
        result=failure_result(spec,classification,q.implementation_test_failure_detail(classification,implementation),{**dict(implementation),"gate":gate})
        write_artifacts(result_dir,spec,result); return spec,result,exit_code
    dataset=r3.load_probe_dataset(spec)
    result=run_experiment(spec,dataset,implementation); write_artifacts(result_dir,spec,result)
    return spec,result,(EXIT_PASS if result["classification"]=="PASS" else EXIT_SCIENTIFIC_FAIL)


def main() -> int:
    raw=(os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw or not (os.environ.get("CB16_COMMIT_SHA") or "").strip(): return EXIT_EXECUTION_BLOCKED
    result_dir=Path(raw)
    try: _s,result,code=run_qualification(result_dir)
    except r3.r0.HistoricalDataUnavailable as exc:
        try: spec=load_spec()
        except Exception: spec=None
        result=failure_result(spec,"EXECUTION_BLOCKED",f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir,spec or {},result); return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try: spec=load_spec()
        except Exception: spec=None
        result=failure_result(spec,"CONTRACT_MISMATCH",f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir,spec or {},result); return EXIT_CONTRACT_MISMATCH
    except Exception as exc:
        try: spec=load_spec()
        except Exception: spec=None
        result=failure_result(spec,"EXECUTION_BLOCKED",f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir,spec or {},result); return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} promotion={result['promotion_decision']['decision']}")
    return code

if __name__ == "__main__": raise SystemExit(main())
