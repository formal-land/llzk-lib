set(ZKIR_DEP_DIALECT_LIBS
  MLIRFuncDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  PARENT_SCOPE
)
add_subdirectory(Dialect/ZKIR)

if(ZKIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(Bindings/Python)
endif()
