# zkir-lib

Veridise's intermediate representation for zero knowledge languages.

### Setup

This repository is already configured with a Nix flakes environment.

To build the ZKIR derivation, you can run `nix build .#zkir` (add `-L` if you
want to print the logs while building).

To launch a developer shell, run the following command:

```bash
nix develop .
```

Then the following command can be used to generate the CMake configuration:

```bash
phases=configurePhase genericBuild
```

By default, the developer shell is set up to build in debug mode. If you want to
generate a release build, append `-DCMAKE_BUILD_TYPE=Release` to `cmakeFlags`:

```
phases=configurePhase cmakeFlags="$cmakeFlags -DCMAKE_BUILD_TYPE=Release" genericBuild
```

Notes:

* Nix flakes are required for this to work.
* Nix 2.13 is assumed. Compatibility with other versions has not been checked
  yet, but they should work.

## Development workflow

Once you have generated the build configuration and are in the build folder, you
can run the following commands:

* Compile: `cmake --build .`
* Run all tests: `cmake --build . --target check`
* Run lit tests: `cmake --build . --target check-lit`
* Generate API docs (in `doc/html`): `cmake --build . --target doc`
* Run install target (requires `CMAKE_INSTALL_PREFIX` to be set):
  `cmake --build . --target install`
* Run clang-format: `clang-format -i $(find include -name '*.h' -o -name '*.td' -type f) $(find lib -name '*.cpp' -type f)`
* Run clang-tidy: `clang-tidy -p build/compile_commands.json $(find lib -name '*.cpp' -type f)`
  * Note that due to bugs in clang-tidy, this may segfault if running on all files.

The build configuration will automatically export `compile_commands.json`, so
LSP servers such as `clangd` should be able to pick up helpful IDE information
like include paths, etc.

### Tools

#TODO

### (Experimental) Python bindings

ZKIR has experimental support for MLIR's Python bindings.

Prerequisites:
* The Python packages required for MLIR's Python bindings must be installed, as
  indicated in the `mlir/python/requirements.txt` file in the LLVM monorepo's
  source tree.
* You must build and link ZKIR against a version of MLIR built with
  `MLIR_ENABLE_BINDINGS_PYTHON` set to `ON`. In the Nix setup, this can be
  accessed using the `zkirWithPython` output.
* ZKIR must be configured with `-DZKIR_ENABLE_BINDINGS_PYTHON=ON`.

## License

Copyright 2024 Veridise, Inc. All Rights Reserved.
