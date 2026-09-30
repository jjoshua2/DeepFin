import Base
import ./LAWS.bend as Laws
import ./Read.bend as Read
import ./Roundtrip.bend as Roundtrip
import ./Initialize.bend as Initialize
import ./Write.bend as Write
import ./Build.bend as Build
import ./Fill.bend as Fill

def Laws.read_preserves_storage(a,i):
  Read.read_any(a,i)

def Laws.write_read_same_location(a,i,v):
  Roundtrip.write_read(a,i,v)

def Laws.initialized_read(d,v,i):
  Initialize.read(d,v,i)

def Laws.write_preserves_shape(a,i,v):
  Write.write(a,i,v)

def Laws.fill_preserves_shape(n,at,sq,bishop,mask,subset,a):
  Build.fill(n,at,sq,bishop,mask,subset,a)

def Laws.tables_preserve_shape(n,key,at,a):
  Build.tables(n,key,at,a)

def Laws.extras_preserve_shape(n,sq,a):
  Build.extras(n,sq,a)

def Laws.fill_one_read(a,at,sq,bishop,mask,subset):
  Fill.one(a,at,sq,bishop,mask,subset)
