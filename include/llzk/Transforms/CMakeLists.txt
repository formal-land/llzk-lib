include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR} ${R1CS_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/r1cs/include/)

set(LLVM_TARGET_DEFINITIONS "LLZKTransformationPasses.td")
mlir_tablegen(LLZKTransformationPasses.h.inc -gen-pass-decls -name=Transformation)
mlir_tablegen(LLZKTransformationPasses.capi.h.inc -gen-pass-capi-header --prefix LLZKTransformation)
mlir_tablegen(LLZKTransformationPasses.capi.cpp.inc -gen-pass-capi-impl --prefix LLZKTransformation)
llzk_add_mlir_doc(LLZKTransformationPassesDocGen passes/LLZKTransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKTransformationIncGen)
add_dependencies(mlir-headers LLZKTransformationIncGen)
add_dependencies(LLZKDialectHeaders LLZKTransformationIncGen)
