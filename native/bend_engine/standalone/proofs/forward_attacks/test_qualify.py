"""Host classification checks; these are not formal source laws."""
import unittest

from .qualify import manifest, semantic_rejection


class ClassificationTests(unittest.TestCase):
    def test_current_complete_manifest(self) -> None:
        manifest()

    def result(self, text: str, code: int | None = 1, timed_out: bool = False) -> dict:
        return {"raw_text": text, "exit_code": code, "timed_out": timed_out}

    def test_only_expected_semantic_location(self) -> None:
        good = "expected True{}\nobserved False{}\nLocation: CastleProjection.step"
        self.assertTrue(semantic_rejection(self.result(good), "step"))
        self.assertFalse(semantic_rejection(self.result(good), "full"))
        self.assertFalse(semantic_rejection(self.result(good, code=0), "step"))
        self.assertFalse(semantic_rejection(self.result(good, code=None, timed_out=True), "step"))

    def test_unrelated_failures_never_count(self) -> None:
        for word in ("WARNING", "unsafe", "RangeError", "Maximum call stack",
                     "more than once", "a decreasing self-call", "out of memory",
                     "a defined name", "a parameter or field scrutinee", "a filled definition"):
            text = f"expected True{{}}\nobserved False{{}}\nLocation: step\n{word}"
            self.assertFalse(semantic_rejection(self.result(text), "step"))

    def test_kind_failures_never_count(self) -> None:
        original = (
            "Error:\n"
            "- expected : Data\n"
            "- observed : Type\n"
            "Context:\n"
            "- c    : ../storage/Observe.Cells\n"
            "- b    : ../../../legal_probe/Chess.Board\n"
            "- rays : U64\n"
            "- m    : ../../../legal_probe/Chess.Ply\n"
            "- acc  : List<&2, ../../../legal_probe/Chess.Ply>\n"
            "Location: step\n"
            "108 |   match part:\n"
            "109>|     case Inl{+plain}:\n"
            "110 |       Equal.trans(List<&2,Chess.Ply>,\n"
        )
        self.assertFalse(semantic_rejection(self.result(original), "step"))
        quantity = "Error:\n- expected : -plain\n- observed : plain\nLocation: step"
        self.assertFalse(semantic_rejection(self.result(quantity), "step"))
        for observed in ("Type", "Data", "Quant", "Kind(&0)", "&2"):
            text = f"Error:\n- expected : Data\n- observed : {observed}\nLocation: step"
            self.assertFalse(semantic_rejection(self.result(text), "step"))

    def test_prefix_location_is_not_complete_name(self) -> None:
        text = "expected True{}\nobserved False{}\nLocation: step_more"
        self.assertFalse(semantic_rejection(self.result(text), "step"))


if __name__ == "__main__":
    unittest.main()
