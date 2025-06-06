{
  inputs = {
    llzk-pkgs.url = "github:Veridise/llzk-nix-pkgs?ref=main";

    release-helpers = {
      url = "github:Veridise/open-source-release-helpers?ref=main";
      inputs = {
        nixpkgs.follows = "llzk-pkgs/nixpkgs";
        flake-utils.follows = "llzk-pkgs/flake-utils";
      };
    };

    nixpkgs = {
      url = "github:NixOS/nixpkgs";
      follows = "llzk-pkgs/nixpkgs";
    };

    flake-utils = {
      url = "github:numtide/flake-utils/v1.0.0";
      follows = "llzk-pkgs/flake-utils";
    };
  };

  # Custom colored bash prompt
  nixConfig.bash-prompt = ''\[\e[0;32m\][LLZK]\[\e[m\] \[\e[38;5;244m\]\w\[\e[m\] % '';

  outputs = { self, nixpkgs, flake-utils, llzk-pkgs, release-helpers }:
    {
      # First, we define the packages used in this repository/flake
      overlays.default = final: prev: {
        pythonCustomised = final.python3.withPackages (ps: with ps; [ numpy ]);

        mlirWithPython = (final.mlir.override {
          enablePythonBindings = true;
          python3 = final.pythonCustomised;
        }).overrideAttrs (old: {
          # Fix numpy include path for newer numpy versions
          cmakeFlags = old.cmakeFlags ++ [
            "-DPython3_NumPy_INCLUDE_DIR=${final.pythonCustomised.pkgs.numpy}/lib/python3.12/site-packages/numpy/_core/include"
          ];
        });

        llzk = final.callPackage ./nix/llzk.nix { clang = final.clang_18; };

        llzkDocs = final.llzk.overrideAttrs(attrs: {
          nativeBuildInputs = attrs.nativeBuildInputs ++ [
            final.doxygen final.graphviz
            final.git final.cacert
          ];
          buildPhase = ''
            cmake --build . --target doc -j$NIX_BUILD_CORES
          '';
          installPhase = ''
            cp -r ./doc $out
          '';
          doCheck = false;
        });

        llzkWithPython = final.callPackage ./nix/llzk.nix { 
          clang = final.clang_18;
          mlir = final.mlirWithPython;
          python3 = final.pythonCustomised;
        };

        llzkDebugClang = (final.llzk.override { stdenv = final.clangStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "DebWithSans";

          postInstall = ''
            if [ -f test/report.xml ]; then
              mkdir -p $out/artifacts
              echo "-- Copying xUnit report to $out/artifacts/clang-report.xml"
              cp test/report.xml $out/artifacts/clang-report.xml
            fi
          '';
        });

        llzkDebugClangCov = final.llzkDebugClang.overrideAttrs(attrs: {
          postCheck = ''
            MANIFEST=profiles.manifest
            PROFDATA=coverage.profdata
            BINS=bins.lst
            if [[ "$(uname)" == "Darwin" ]]; then
              find bin lib -type f | xargs file | fgrep Mach-O | grep executable | cut -f1 -d: > $BINS
            else
              find bin lib -type f | xargs file | grep ELF | grep executable | cut -f1 -d: > $BINS
            fi
            echo -n "Found profraw files:"
            find test -name "*.profraw" | tee $MANIFEST | wc -l
            cat $MANIFEST
            llvm-profdata merge -sparse -f $MANIFEST -o $PROFDATA
            OBJS=$( (head -n 1 $BINS ; tail -n +2 $BINS | sed -e "s/^/-object /") | xargs)
            # TODO HTML reports
            llvm-cov report $OBJS -instr-profile $PROFDATA > cov-summary.txt
            echo =========== COVERAGE SUMMARY =================
            cat cov-summary.txt
            echo ==============================================
            llvm-cov export -format=lcov -instr-profile $PROFDATA $OBJS > report.lcov
            rm -rf $MANIFEST $PROFDATA $BINS
          '';

          postInstall = ''
            mkdir -p $out/artifacts/
            echo "-- Copying coverage summary to $out/artifacts/cov-summary.txt"
            cp cov-summary.txt $out/artifacts/
            echo "-- Copying lcov report to $out/artifacts/report.lcov"
            cp report.lcov $out/artifacts/
            if [ -f test/report.xml ]; then
              echo "-- Copying xUnit report to $out/artifacts/clang-report.xml"
              cp test/report.xml $out/artifacts/clang-report.xml
            fi
          '';
        });

        llzkDebugGCC = (final.llzk.override { stdenv = final.gccStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "DebWithSans";

          postInstall = ''
            if [ -f test/report.xml ]; then
              mkdir -p $out/artifacts
              echo "-- Copying xUnit report to $out/artifacts/gcc-report.xml"
              cp test/report.xml $out/artifacts/gcc-report.xml
            fi
          '';
        });

        ccacheStdenv = prev.ccacheStdenv.override {
          extraConfig = ''
            export CCACHE_DIR=/tmp/ccache
            export CCACHE_UMASK=007
            export CCACHE_COMPRESS=1
          '';
        };

        # The default shell is used for LLZK development.
        # Because `nix develop` is used to set up a dev shell for a given
        # derivation, we just need to extend the llzk derivation with any
        # extra tools we need.
        devShellBase = pkgs: llzkEnv: {
          shell = llzkEnv.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              doxygen graphviz
              git

              # clang-tidy and clang-format
              clang-tools_18

              # git-clang-format
              libclang.python
            ]);

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/bin:"$PATH"

              # Add release helpers to the PATH for convenience
              export PATH="${pkgs.changelogCreator.out}/bin":"$PATH"

              # TODO: only enable if python bindings enabled
              export PYTHONPATH="$PYTHONPATH":"$PWD"/build/python

              # Needed for using mlir-tblgen inside the dev shell
              export LD_LIBRARY_PATH=${pkgs.z3.lib}/lib:$LD_LIBRARY_PATH

              # Disable container overflow checks because it can give false positives in
              # newGeneralRewritePatternSet() since LLVM itself is not built with ASan.
              # https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow#false-positives
              export ASAN_OPTIONS=detect_container_overflow=0
            '';
          });
        };

        devShellBaseWithDefault = pkgs: final.devShellBase pkgs final.llzk;
      };
    } //
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

          overlays = [
            self.overlays.default
            llzk-pkgs.overlays.default
            release-helpers.overlays.default
          ];
        };
      in
      {
        # Now, we can define the actual outputs of the flake
        packages = flake-utils.lib.flattenTree {
          # Copy the packages from the overlay.
          inherit (pkgs) llzk llzkWithPython changelogCreator;

          # For debug purposes, expose the MLIR/LLVM packages.
          inherit (pkgs) mlir mlirWithPython;
          # Prevent use of libllvm and llvm from nixpkgs, which will have different
          # versions than the mlir from llzk-pkgs.
          inherit (pkgs.llzk_llvmPackages) libllvm llvm;

          default = pkgs.llzk;
          debugClang = pkgs.llzkDebugClang;
          debugClangCov = pkgs.llzkDebugClangCov;
          debugGCC = pkgs.llzkDebugGCC;
          docs = pkgs.llzkDocs;
        };

        checks = flake-utils.lib.flattenTree {
          llzkInstallCheck = pkgs.callPackage ./nix/llzk-installcheck { };
        };

        devShells = flake-utils.lib.flattenTree {
          default = (pkgs.devShellBaseWithDefault pkgs).shell.overrideAttrs (_: {
            # Use Debug by default so assertions are enabled by default.
            cmakeBuildType = "Debug";
          });
          debugClang = (pkgs.devShellBase pkgs pkgs.llzkDebugClang).shell;
          debugGCC = (pkgs.devShellBase pkgs pkgs.llzkDebugGCC).shell;

          llvm = pkgs.mkShell {
            buildInputs = [ pkgs.llzk_llvmPackages.libllvm.dev ];
          };
        };
      }
    ));
}
