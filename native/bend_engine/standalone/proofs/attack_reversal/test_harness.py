"""Host validation tests, not extra formal theorems or native executions."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from _common import InvalidEvidence, atomic_report, safe
from verify_native import cases, geometry, geometry_mismatch, parse, reciprocity

class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=cases()[:768]
        cls.values=[geometry(r[1],r[2]) for r in cls.rows]
    def test_mask_parser(self):self.assertEqual(parse("mask 1 2\n",1),[(1<<32)|2])
    def test_truncation(self):
        with self.assertRaises(InvalidEvidence):parse("mask 1 2\n",2)
    def test_extra_row(self):
        with self.assertRaises(InvalidEvidence):parse("mask 1 2\nmask 1 2\n",1)
    def test_missing_limb(self):
        with self.assertRaises(InvalidEvidence):parse("mask 1\n",1)
    def test_wrong_tag(self):
        with self.assertRaises(InvalidEvidence):parse("table 1 2\n",1)
    def test_invalid_unsigned(self):
        for bad in ["-1","+1","4294967296","True","1.0"]:
            with self.subTest(bad=bad),self.assertRaises(InvalidEvidence):parse(f"mask {bad} 0\n",1)
    def test_zero_exit_warning(self):
        with self.assertRaises(InvalidEvidence):safe({"exit_code":0,"stdout":"All terms check.\nWARNING","stderr":""})
    def test_nonzero(self):
        with self.assertRaises(InvalidEvidence):safe({"exit_code":1,"stdout":"All terms check.","stderr":""})
    def test_safe_receipt(self):safe({"exit_code":0,"stdout":"All terms check.\n","stderr":""})
    def test_geometry_reciprocity(self):
        r=reciprocity(self.values,self.rows)
        self.assertIsNone(r["first_mismatch"])
        self.assertEqual(r["pair_comparisons"],3*4*64*64)
        self.assertEqual(r["positive_edges"],3*952)
    def test_wrong_but_reciprocal_zero(self):
        values=[0]*len(self.rows)
        self.assertIsNone(reciprocity(values,self.rows)["first_mismatch"])
        self.assertIsNotNone(geometry_mismatch(values,self.rows))
    def test_wrong_but_reciprocal_swapped_pawns(self):
        values=[geometry((5-r[1]) if r[1]>=2 else r[1],r[2]) for r in self.rows]
        self.assertIsNone(reciprocity(values,self.rows)["first_mismatch"])
        self.assertIsNotNone(geometry_mismatch(values,self.rows))
    def test_one_edge_corruption(self):
        values=list(self.values);values[0]^=1<<1
        self.assertIsNotNone(reciprocity(values,self.rows)["first_mismatch"])
    def test_source_field_count(self):
        with self.assertRaises(InvalidEvidence):geometry_mismatch(self.values[:-1],self.rows)
    def test_report_invalidated(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"report.json";p.write_text('{"native_gate":"PASS"}')
            atomic_report(p,{"native_gate":"NOT_COMPLETED"})
            self.assertEqual(json.loads(p.read_text()),{"native_gate":"NOT_COMPLETED"})

if __name__=="__main__":unittest.main()
