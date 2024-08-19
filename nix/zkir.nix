{
  stdenv, lib,

  # build dependencies
  cmake, ninja,
  mlir, nlohmann_json,

  # test dependencies
  gtest, python3, lit, z3, cvc5
}:

stdenv.mkDerivation {
  name = "zkir";
  version = "0.1.0";
  src =
    let
      src0 = lib.cleanSource (builtins.path {
        path = ./..;
        name = "zkir-source";
      });
    in
      lib.cleanSourceWith {
        # Ignore unnecessary files
        filter = path: type: !(lib.lists.any (x: x) [
          (path == toString (src0.origSrc + "/README.md"))
          (type == "directory" && path == toString (src0.origSrc + "/third-party"))
          (type == "directory" && path == toString (src0.origSrc + "/.github"))
          (type == "regular" && lib.strings.hasSuffix ".nix" (toString path))
          (type == "regular" && baseNameOf path == "flake.lock")
        ]);
        src = src0;
      };

  nativeBuildInputs = [ cmake ninja ];
  buildInputs = [
    mlir
  ] ++ lib.optionals mlir.hasPythonBindings [
    mlir.python
    mlir.pythonDeps
  ];

  cmakeFlags = [
    "-DZKIR_BUILD_DEVTOOLS=ON"
  ];

  doCheck = true;
  checkTarget = "check";
  checkInputs = [ gtest python3 lit z3 cvc5 ];
}
