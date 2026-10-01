"""Recreate checked finite proof constructors; output values are not axioms."""
from pathlib import Path
import argparse

def render():
    result={}
    classes={
     'Knight':([(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)],[6,7,4,5,2,3,0,1]),
     'King': ([(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)],[1,0,3,2,7,6,5,4]),
     'WhitePawn': ([(1,1),(-1,1)],[1,0]),
     'BlackPawn': ([(1,-1),(-1,-1)],[1,0])}
    for cls,(offsets,inverse) in classes.items():
     lines=['# Finite edge certificates. The compiler checks every constructor witness.','import Base','import ./Spec.bend as S','import ../attack_geometry/Geometry.bend as G','import ../layout/Domain.bend as D','',f'def rows(+src: Nat,bound: {{Nat.is_lt(src,64n) == True{{}} : Bool}}) -> S.certificate(G.{cls}{{}},src):','  match src:']
     for src in range(64):
      proofs=[]
      for (dx,dy),inv in zip(offsets,inverse):
       x=src%8+dx; y=src//8+dy
       if not(0<=x<8 and 0<=y<8):
        term=f'e => D.impossible(S.member({src}n,G.targets(G.reverse(G.{cls}{{}}),64n)),0n,e)'
       else:
        term='Inl{{==}}'
        for i in range(inv):term='Inr{'+term+'}'
        term='e => '+term
       proofs.append(term)
      term='Unit{}'
      for p in reversed(proofs):term=f'({p},{term})'
      lines.append(f'    case {src}n: {term}')
     lines.append(f'    case 64n+p: D.impossible(S.certificate(G.{cls}{{}},64n+p),p,bound)')
     result[cls+'.bend']='\n'.join(lines)+'\n'
    return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args();s=Path(__file__).resolve().parent
    for name,text in render().items():
        if args.write:(s/name).write_text(text)
        elif (s/name).read_text()!=text:raise SystemExit("Certificate source differs: "+name)
    print("4 certificate files match")
