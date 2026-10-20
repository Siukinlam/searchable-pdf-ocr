# Test fixture font

`DroidSansFallback-subset.ttf` is a subset of Droid Sans Fallback (Full), used
only to draw synthetic scans in the tests. It covers printable ASCII and the
CJK characters the tests use.

- **Source:** https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/android-4.4_r1/data/fonts/DroidSansFallbackFull.ttf
- **Source SHA-256:** `97320619b0f1ce4a7cda74cf064a0ce44ef3091287bc42755597e177cccf94c2`
- **Copyright:** Copyright (c) 2005-2008, The Android Open Source Project.
- **License:** Apache License, Version 2.0. See `LICENSE` in the repository root.
- **How it was made:** subset with fontTools (`fontTools.subset`), keeping the
  name table.
