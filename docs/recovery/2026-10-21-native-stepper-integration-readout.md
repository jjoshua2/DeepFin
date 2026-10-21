# Native stepper integration qualification

The first coherent-copy component attempt on 2026-10-21 ended before any test
method ran. It copied the isolated source payload and attempted normal imports;
the finite controller then refused an uninventoried library-load request.
Native elapsed time was 4.4932757 seconds, exit status 1, with both output streams
complete. The partial target and complete failed receipts are retained.

Source inspection traces the import through the position-fingerprint helper and
Stockfish convenience imports to its import-time `libc.so.6` symbol resolution.
The controller treated that Linux library name as a relative pathname despite
already inventorying the canonical libc image. This is a source-supported cause,
not a claim that the failed receipt recorded the exact load argument.

The successor must preserve production imports and authenticate the exact mapped
library and resolved symbol address. No broad library-name allowance, model
import suppression, native stub, resource increase or automatic replay is
permitted. It will use a distinct target and receipt after independent review.

Zero native component methods have accepted credit from this attempt. Some
native imports may already have occurred; post-import facts and ECHILD were not
reached and are not inferred from the native command's exit. No GPU or ordinary
queue action was part of the controller. Ordinary productive generation retains
priority while integration continues.

Failed stdout SHA256:
`a4f900895dc92ad1ebd256663edd493d547fb4a1611927bdff440545ab6655e3`.
Failed stderr SHA256:
`06e6d44681215d23730f4cb75fd77af471f7ac82670d14a69b6ccac40d736d99`.
Independent failed-actual review:
`3844fb7f873202420ac84a1ba11974decbc1976d1d791df99876388ed6200c12`.
