# Locked upstream snapshot

- Repository: <https://github.com/r2barati/gym-invmgmt-paper>
- Branch: `main`
- Commit: `0ec993a9dad0dd220787e77e77868d10ad5ab799`
- Commit date: 2026-05-13
- License: MIT; the upstream `LICENSE` is retained under
  `vendor/gym_invmgmt_paper/`.

The vendored tree contains the environment, benchmark scenario definitions,
agents, tests, canonical per-agent caches, and the official merged result
artifact. FBiT code lives outside the vendored tree so the upstream snapshot
can be compared byte-for-byte and replaced deliberately.

The paper manifest contains a non-public generating commit for its historical
result cache. This project therefore locks the current public head above and
preserves the released result files without claiming to recreate their missing
historical commit.
