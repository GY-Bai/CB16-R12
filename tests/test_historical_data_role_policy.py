"""Guard the canonical historical-data role boundary."""
from __future__ import annotations
import json, re, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'config/r12_historical_data_role_policy_v1.json'
AUTH=ROOT/'docs/authority/R12_HISTORICAL_DATA_ROLE_AND_ACCESS_POLICY.md'
INDEX=ROOT/'docs/authority/AUTHORITY_INDEX.json'
ARCHIVE_RE=re.compile(r'^(?P<symbol>[A-Z0-9]+)-1m-(?P<month>\d{4}-\d{2})\.zip$')
class HistoricalDataRolePolicyTests(unittest.TestCase):
 def setUp(self):
  self.policy=json.loads(POLICY.read_text())
  self.assigned={(x['symbol'],x['month']):x['role'] for x in self.policy['explicit_assignments']}
 def test_policy_is_canonical_and_default_deny(self):
  self.assertEqual(self.policy['schema'],'cb16.historical-data-role-policy.v1')
  self.assertEqual(self.policy['status'],'CURRENT_CANONICAL')
  p=self.policy['principles']; self.assertEqual(p['default_for_unassigned_market_data'],'UNASSIGNED_PROTECTED'); self.assertTrue(p['unassigned_protected_is_not_final_holdout']); self.assertTrue(p['role_assignment_must_precede_experiment_preregistration'])
 def test_only_current_explicit_development_assignments_are_btc_jan_mar(self):
  self.assertEqual(self.assigned,{('BTCUSDT','2020-01'):'EXPOSED_DEVELOPMENT',('BTCUSDT','2020-02'):'EXPOSED_DEVELOPMENT',('BTCUSDT','2020-03'):'EXPOSED_DEVELOPMENT'})
 def test_final_holdout_is_not_inferred(self):
  h=self.policy['protected_holdout']; self.assertEqual(h['repository_role_assignment'],'NOT_ASSIGNED'); self.assertFalse(h['access_authorized'])
 def test_current_formal_specs_do_not_self_authorize_unassigned_months(self):
  violations=[]
  for path in sorted((ROOT/'config/experiments').glob('*.json')):
   spec=json.loads(path.read_text()); data=spec.get('data')
   if not isinstance(data,dict): continue
   for raw in data.get('allowed_archives',[]):
    if not isinstance(raw,dict) or not isinstance(raw.get('name'),str): continue
    m=ARCHIVE_RE.match(raw['name'])
    if not m: continue
    key=(m.group('symbol'),m.group('month'))
    if key not in self.assigned: violations.append((path.name,raw['name']))
  self.assertEqual(violations,[])
 def test_authority_index_and_readme_list_policy(self):
  idx=json.loads(INDEX.read_text()); rel='docs/authority/R12_HISTORICAL_DATA_ROLE_AND_ACCESS_POLICY.md'; self.assertIn(rel,idx['current_canonical']); self.assertIn('R12_HISTORICAL_DATA_ROLE_AND_ACCESS_POLICY.md',(ROOT/'docs/authority/README.md').read_text()); self.assertTrue(AUTH.is_file())
if __name__=='__main__': unittest.main()
