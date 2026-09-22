import Base
import ../../Tables.bend as Tables
import ./Roundtrip.bend as Roundtrip

def one(a: Array<U64>, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64) ->
  {Array.get(U64,Tables.fill(1n,at,sq,bishop,mask,subset,a),at) ==
    (Tables.fill(1n,at,sq,bishop,mask,subset,a),Tables.slider(sq,bishop,subset,False{})) : Array<U64> & U64}:
  Roundtrip.write_read(a,at,Tables.slider(sq,bishop,subset,False{}))
