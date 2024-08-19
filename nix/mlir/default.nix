{ lib, stdenv, llvm_meta
, monorepoSrc, runCommand
, cmake, ninja, python3, libffi, fixDarwinDylibNames, version
, enableShared ? !stdenv.hostPlatform.isStatic
, debugVersion ? false
, enablePythonBindings ? false
, buildLlvmTools
, libxml2, libllvm
}:

let
  pythonDeps = with python3.pkgs; [
    numpy
    pybind11
    pyyaml
  ];
in
stdenv.mkDerivation rec {
  pname = "mlir";
  inherit version;

  src = runCommand "${pname}-src-${version}" {} ''
    mkdir -p "$out"
    cp -r ${monorepoSrc}/cmake "$out"
    cp -r ${monorepoSrc}/${pname} "$out"
    cp -r ${monorepoSrc}/llvm "$out"
  '';

  sourceRoot = "${src.name}/${pname}";

  nativeBuildInputs = [
    cmake ninja python3 libffi
  ] ++ lib.optionals enablePythonBindings pythonDeps # todo: this should be propagated
    ++ lib.optional stdenv.hostPlatform.isDarwin fixDarwinDylibNames;

  propagatedBuildInputs = [ libllvm ];

  cmakeFlags = [
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_BUILD_TYPE=${if debugVersion then "Debug" else "Release"}"
    "-DLLVM_ENABLE_RTTI=ON"
    "-DMLIR_INCLUDE_DOCS=ON"
    "-DMLIR_STANDALONE_BUILD=TRUE"
    "-DLLVM_TARGETS_TO_BUILD=host"
    "-DMLIR_INSTALL_PACKAGE_DIR=${placeholder "dev"}/lib/cmake/mlir"
  ] ++ lib.optionals enablePythonBindings [
    # Enable Python bindings
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
    "-DPython3_EXECUTABLE=${python3}/bin/python"
    "-DPython3_NumPy_INCLUDE_DIR=${python3.pkgs.numpy}/${python3.sitePackages}/numpy/core/include"
  # ] ++ lib.optionals enableManpages [
  #   "-DCLANG_INCLUDE_DOCS=ON"
  #   "-DLLVM_ENABLE_SPHINX=ON"
  #   "-DSPHINX_OUTPUT_MAN=ON"
  #   "-DSPHINX_OUTPUT_HTML=OFF"
  #   "-DSPHINX_WARNINGS_AS_ERRORS=OFF"
  ] ++ lib.optionals (stdenv.hostPlatform != stdenv.buildPlatform) [
    "-DLLVM_TABLEGEN_EXE=${buildLlvmTools.llvm}/bin/llvm-tblgen"
  ];

  patches = [
    ./gnu-install-dirs.patch
  ];

  outputs = [ "out" "lib" "dev" ] ++ lib.optionals enablePythonBindings [ "python" ];

  postInstall = ''
    mkdir -p $out/bin

    mkdir -p $dev/bin
    cp bin/* $dev/bin/

    # fix utility tool paths
    substituteInPlace "$dev"/lib/cmake/mlir/MLIRConfig.cmake \
      --replace '"mlir-tblgen"' \""$dev"/bin/mlir-tblgen\" \
      --replace '"mlir-pdll"' \""$dev"/bin/mlir-pdll\"

    ${lib.strings.optionalString enablePythonBindings ''
    # move mlir source code
    mkdir -p $python
    mv $out/src $python/src

    # move mlir package code
    mkdir -p $python/${python3.sitePackages}
    mv $out/python_packages/mlir_core/mlir $python/${python3.sitePackages}/mlir
    echo 'mlir' > $python/${python3.sitePackages}/mlir_core.pth

    # move Python bindings DSO to lib output, since they are searched for in there
    mv $python/${python3.sitePackages}/mlir/_mlir_libs/libMLIRPythonCAPI${stdenv.hostPlatform.extensions.sharedLibrary} $lib/lib/
    ''}
  '';

  passthru = {
    hasPythonBindings = enablePythonBindings;
    inherit pythonDeps;
  };
}
