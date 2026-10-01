"""Host checks of the independent oracle, framing, certificates and gate safety."""
import tempfile
import unittest
import json
from pathlib import Path
from _common import InvalidEvidence, atomic_report, safe, require
from generate_certificates import outputs, path_length
from verify_native import geometry, interior, parse, analyze, FULL, fixtures

class Harness(unittest.TestCase):
    def test_first_blocker_included(self):
        self.assertTrue(geometry(0,0,1<<3)&(1<<3))
        self.assertFalse(geometry(0,0,1<<3)&(1<<4))
    def test_source_occupancy_irrelevant(self):
        self.assertEqual(geometry(2,27,0),geometry(2,27,1<<27))
    def test_occupied_adjacent_endpoints(self):
        occ=(1<<0)|(1<<1)
        self.assertTrue(geometry(0,0,occ)&2)
        self.assertTrue(geometry(0,1,occ)&1)
    def test_bishop_first_blocker(self):
        self.assertTrue(geometry(1,0,1<<27)&(1<<27))
        self.assertFalse(geometry(1,0,1<<27)&(1<<36))
    def test_no_rank_wrap(self):
        self.assertFalse(geometry(0,7,0)&(1<<8))
    def test_no_self_edge(self):
        for kind in (0,1,2):
            self.assertFalse(geometry(kind,27,0)&(1<<27))
    def test_queen_union(self):
        self.assertEqual(geometry(2,36,0x55),geometry(0,36,0x55)|geometry(1,36,0x55))
    def test_interior_excludes_endpoints(self):
        self.assertEqual(interior(0,0,7),tuple(range(1,7)))
        self.assertEqual(interior(0,7,0),tuple(range(6,0,-1)))
    def test_adjacent_interior_empty(self):
        self.assertEqual(interior(1,0,9),())
    def test_unaligned_missing(self):
        self.assertIsNone(interior(1,0,7))
        self.assertIsNone(interior(0,0,9))
        self.assertIsNone(interior(0,0,0))
    def test_different_occupancy_breaks_reversal(self):
        self.assertTrue(geometry(0,0,0)&(1<<7))
        self.assertFalse(geometry(0,7,1<<3)&1)
    def test_oracle_invalid_domain(self):
        for k,s,o in [(3,0,0),(0,64,0),(0,0,-1),(0,0,1<<64)]:
            with self.assertRaises(InvalidEvidence):geometry(k,s,o)
    def test_parse_exact(self):
        self.assertEqual(parse('4294967295 4294967295\n0 1\n',2),[FULL,1])
    def test_parse_empty(self):
        self.assertEqual(parse('',0),[])
        with self.assertRaises(InvalidEvidence):parse('',1)
    def test_parse_truncation_and_extra(self):
        with self.assertRaises(InvalidEvidence):parse('0 0\n',2)
        with self.assertRaises(InvalidEvidence):parse('0 0\n0 0\n',1)
    def test_parse_reject_signed_overflow_and_framing(self):
        for line in ['-1 0','+1 0','4294967296 0','0 4294967296','0 x','0','0 0 0','nan 0','0.0 0']:
            with self.assertRaises(InvalidEvidence):parse(line,1)
    def test_safe_only_exact_success(self):
        safe({'exit_code':0,'stdout':'All terms check.\n','stderr':''})
        for r in [{'exit_code':1,'stdout':'All terms check.','stderr':''},
                  {'exit_code':0,'stdout':'All terms check.\nWARNING','stderr':''},
                  {'exit_code':0,'stdout':'','stderr':''}]:
            with self.assertRaises(InvalidEvidence):safe(r)
    def test_reports_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.json';atomic_report(p,{'gate':'PASS'});atomic_report(p,{'gate':'NOT_COMPLETED'})
            self.assertEqual(json.loads(p.read_text()),{'gate':'NOT_COMPLETED'})
    def test_mandatory_require(self):
        with self.assertRaises(InvalidEvidence):require(False,'still enforced')
    def test_certificate_counts_and_reproducibility(self):
        self.assertEqual(len(outputs()),9)
        self.assertEqual(sum(path_length(s,d) for s in range(64) for d in range(8)),1456)
        for name,text in outputs().items():
            self.assertEqual((Path(__file__).parent/name).read_text(),text)
    def test_fixtures_complete_interior_partition(self):
        p,q,c=fixtures()
        self.assertEqual((c['aligned_unordered_pairs'],c['interior_subsets_across_pairs']),(728,5322))
        self.assertEqual(c['distinct_native_queries'],len(set(q)))
        self.assertEqual(c['distinct_relation_cases'],len(set(p)))
        self.assertEqual(c['aligned_relation_requests_before_union'],5322*4*2*2)
    def test_reciprocal_is_not_correct(self):
        queries=[(0,0,1<<3),(0,7,1<<3)];pairs=[(0,0,7,1<<3)]
        wrong=[geometry(0,s,0) for _,s,_ in queries]
        r=analyze(wrong,queries,pairs)
        self.assertIsNone(r['reciprocity_mismatch']);self.assertIsNotNone(r['geometry_mismatch'])
    def test_zero_masks_reciprocal_wrong(self):
        r=analyze([0,0],[(0,0,0),(0,7,0)],[(0,0,7,0)])
        self.assertIsNone(r['reciprocity_mismatch']);self.assertIsNotNone(r['geometry_mismatch'])
    def test_analyze_requires_complete_values(self):
        with self.assertRaises(InvalidEvidence):analyze([0],[(0,0,0),(0,7,0)],[])

if __name__=='__main__':unittest.main()
