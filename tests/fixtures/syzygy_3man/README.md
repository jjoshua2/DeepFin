# 3-man Syzygy fixture (tests only)

Ten standard Syzygy table files (the complete 3-man set), so CI (which ships no tablebases) runs the
C prober against real tables: `tests/test_uci_tb_fixture.py`.

- Origin: the standard Syzygy 3-4-5 piece set (the files the Lichess / Syzygy
  mirrors distribute), unmodified.
- Every file's md5 is listed in the vendored manifest
  `chess_anti_engine/tb/syzygy_standard_3-4-5-6.md5`; the test re-checks it.
- Total 25,824 bytes. KBvK and KNvK (draw tables, 80 bytes each) are
  there so every promotion from KPvK has a table: Fathom's root probe needs
  one for every move. Not package data: `tests*` is excluded from the wheel
  and sdist package discovery (`pyproject.toml`).

| file | bytes | md5 | sha256 |
|---|---|---|---|
| KBvK.rtbw | 80 | 99bf9b05295781611cdd7c5c3d51bf85 | bc0d8ab3560de9038460f0e8f61e3b7efd3e3d0327d19ae0387c0da7ddeb244d |
| KBvK.rtbz | 80 | 88d5f823e67448b279bb045977a80a39 | 6246c8a5c643eec9d4758d55ed843e31cc28a9b82a6f630c1c950d8b7755975b |
| KNvK.rtbw | 80 | b6781a75ffe2ab41507f91151869a418 | 9d3518b12df3d2006441df758bce31e961346e70a11920aaad75c647003bb6e7 |
| KNvK.rtbz | 80 | 42893523156bbc5d8c3c7207a7710ad7 | 0e49a0f2810a131c32fc14872e83044aa9810ce2c86df071851512077291fc25 |
| KPvK.rtbw | 7824 | 46f6ef491bd26696d7b20281e7c5b721 | 02fd2bdcd92821c869458f019f63524def1c9e7a2862ca8d2b16dc7712b16183 |
| KPvK.rtbz | 4176 | 54460894c15f087cfd16670bf1513755 | ad01af2076183c90ccee662ca6c5d74da3222ff1fee47c01b5d4e56ed01938de |
| KQvK.rtbw | 272 | f06221548404795b6b33469e247b4560 | 517667dff787162dbb1ed9d5d6484d30ee854e686ee0675c08d99ecf045d2d50 |
| KQvK.rtbz | 5392 | ac866466e16eb19a4f8c796f8e1abd2b | 71ea9444fa5bd42897d781a0c356975ea6f23e0f65a4254e470897031c161c8c |
| KRvK.rtbw | 208 | 89a27823bfa03d0b0f25728c4f0fb571 | 386fbde73308e49a4207836922c68b30b664e83c5a37f7fa37305a15cd16f2f1 |
| KRvK.rtbz | 7632 | 9cb0795fa43904a3e91bf749971964b8 | cab59f42e75c2a25da3939231850a79fa838593fff5c28d7772da92062ed965a |
