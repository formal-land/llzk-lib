{ stdenv, lib, cmake, ninja, mlir, zkir }:

stdenv.mkDerivation {
  pname = "zkir-installcheck";
  version = "1.0.0";

  src = lib.cleanSource ./.;

  buildInputs = [ mlir zkir ];
  nativeBuildInputs = [ cmake ninja ];

  installPhase = ''touch "$out"'';
}