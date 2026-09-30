"""Explicit x86-64 map-test target; reject unsupported hosts before running it."""
from collections.abc import Callable
import hashlib
from pathlib import Path

TARGET_NAME = 'bmi2-popcnt'
TARGET_FLAGS = ('-march=x86-64', '-mpopcnt', '-mbmi2')
HOST_SOURCE = r'''#include <stdio.h>
#if !defined(__x86_64__)
#error This qualification requires x86-64
#endif
#if defined(__BMI2__) || defined(__POPCNT__) || defined(__AVX__)
#error Compile the host check for baseline x86-64 only
#endif
int main(void) {
  __builtin_cpu_init();
  printf("bmi2=%d popcnt=%d\n", !!__builtin_cpu_supports("bmi2"),
         !!__builtin_cpu_supports("popcnt"));
  return 0;
}
'''
TARGET_SOURCE = r'''#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#if !defined(__BMI2__) || !defined(__POPCNT__)
#error The accelerated target must enable both BMI2 and POPCNT
#endif
#if defined(__AVX__) || defined(__AVX512F__)
#error This target must not depend on automatic AVX feature selection
#endif
int main(void) {
  volatile uint64_t x = UINT64_C(0xf0f0f0f0f0f0f0f0);
  volatile uint64_t mask = UINT64_C(0xaaaaaaaaaaaaaaaa);
  if (_pext_u64(x, mask) != UINT64_C(0xcccccccc) || _mm_popcnt_u64(x) != 32)
    return 2;
  puts("bmi2-popcnt-ok");
  return 0;
}
'''


def check_host(text: str) -> None:
    if text != 'bmi2=1 popcnt=1\n':
        raise ValueError(f'BMI2/POPCNT host check failed: {text!r}; no target fallback')


def qualify(cc: str, output: Path, command: Callable[[list[str], str], str]) -> dict[str, object]:
    """The caller retains diagnostics and rejects any stderr or nonzero exit."""
    host, target = output / 'cpu-host.c', output / 'cpu-target.c'
    host.write_text(HOST_SOURCE)
    target.write_text(TARGET_SOURCE)
    host_binary, target_binary = output / 'cpu-host', output / 'cpu-target'
    # Query support using baseline code before executing any selected instructions.
    command([cc, '-std=c11', '-O2', '-march=x86-64', str(host), '-o', str(host_binary)], 'cpu-host-build')
    capability = command([str(host_binary)], 'cpu-host-run')
    check_host(capability)
    command([cc, '-std=c11', '-O2', *TARGET_FLAGS, str(target), '-o', str(target_binary)], 'cpu-target-build')
    if command([str(target_binary)], 'cpu-target-run') != 'bmi2-popcnt-ok\n':
        raise ValueError('BMI2/POPCNT instruction check failed')
    return {'name': TARGET_NAME, 'flags': list(TARGET_FLAGS), 'host': capability.strip(),
            'instruction_check': 'passed',
            'check_source_sha256': hashlib.sha256((HOST_SOURCE + TARGET_SOURCE).encode()).hexdigest()}
