# Castling producer geometry

Six universal, finite-route contracts for the four orthodox castling coordinate
choices used by the actual producer. They establish source/destination/rook-source,
the identity between king transit and rook landing, and that both transit and king
destination belong to the producer's between mask.

These laws are geometry only. The focused gate separately checks that the actual
`Chess.castle_side` source still contains the exact home/src/rook/dst/transit/right/
between/guard expressions. Native legal-probe parity is separate executable evidence.

This does not yet prove a list-membership theorem saying every flag-two member returned
by `legal_moves` satisfies these premises, nor king-safety soundness/completeness.
