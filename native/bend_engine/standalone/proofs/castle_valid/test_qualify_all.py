"""Host-only guards for intended-obligation rejection classification."""

import unittest

from .qualify_all import strict


class RejectionClassifierTests(unittest.TestCase):
    def test_kind_and_quantity_failures_receive_no_credit(self):
        receipt = {"reason": "EXIT", "returncode": 1, "remaining_live_owned": 0}
        for expected, observed in (
            ("Quant", "Kind(&0)"),
            ("&0", "&2"),
            ("Type", "Data"),
            ("Data", "Type"),
            ("Kind", "Quant"),
            ("Kind(&2)", "Kind(&0)"),
        ):
            with self.subTest(expected=expected, observed=observed):
                diagnostic = f"- expected : {expected}\n- observed : {observed}\nLocation: corner\n"
                if strict(receipt, diagnostic, "corner"):
                    self.fail("kind or quantity failure received semantic credit")

    def test_only_intended_semantic_obligation_receives_credit(self):
        receipt = {"reason": "EXIT", "returncode": 1, "remaining_live_owned": 0}
        diagnostic = "- expected : {8 == 4 : U32}\n- observed : {8 == 8 : U32}\nLocation: corner\n"
        if not strict(receipt, diagnostic, "corner"):
            self.fail("intended equality failure did not receive credit")
        if strict(receipt, diagnostic, "own_bit"):
            self.fail("wrong rejection location received credit")
        for reason, code, remaining in (
            ("TIMEOUT", 1, 0),
            ("RSS_LIMIT", 1, 0),
            ("EXIT", 0, 0),
            ("EXIT", 1, 1),
        ):
            with self.subTest(reason=reason, code=code, remaining=remaining):
                if strict(
                    {
                        "reason": reason,
                        "returncode": code,
                        "remaining_live_owned": remaining,
                    },
                    diagnostic,
                    "corner",
                ):
                    self.fail("failed execution or cleanup received semantic credit")
        if strict(receipt, diagnostic + "cannot infer\n", "corner"):
            self.fail("inference failure received semantic credit")


if __name__ == "__main__":
    unittest.main()
