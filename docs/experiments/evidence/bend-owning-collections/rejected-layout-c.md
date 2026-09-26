# Rejected unboxed layout

Source commit `5e2c2a5eca3ce2b5e183bd98b93cccd1646eaf71`.

Generated C lines 1377-1379:
```c
INLINE u32 blk_at(Term a, U32 i, u32 lgs) {
  return ((u32)i & (u32)((1ull << (blk_cls(a) - lgs)) - 1)) << lgs;
}
```

Generated C lines 3720-3735:
```c
      return 0;
    }
    v_59 = o_1[0];
    v_58 = v_59;
    Term at_0 = blk_at(r_10, r_12, 4);
    u32 c_0 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 0);
    u32 c_1 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 1);
    Term c_2 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 2);
    u32 c_3 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 3);
    u32 c_4 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 4);
    u32 c_5 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 5);
    u32 c_6 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 6);
    u32 c_7 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 7);
    u32 c_8 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 8);
    u32 c_9 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 9);
    u32 c_10 = blk_read(e.mem, 1, term_loc(r_10), at_0 + 10);
```

Generated C lines 5804-5820:
```c
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_RING_SLOTS)
  {
    Term depth_0 = r0;
    WL_OPEN
    WL_SPIN
    if (depth_0 == 0) {
      Term fv_0[2];
      fv_0[0] = 0;
      fv_0[1] = 0;
      r0 = blk_new(e, 1, 0, 1, 2, fv_0);
      WL_RETN(1);
    } else {
```
