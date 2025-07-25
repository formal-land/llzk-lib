add_library(LLZKCAPI)
add_library(LLZK::LLZKCAPI ALIAS LLZKCAPI)
target_link_libraries(LLZKAllDialects INTERFACE LLZKCAPI)

target_sources(LLZKCAPI PRIVATE 
  Transforms.cpp 
  Validators.cpp 
  InitDialects.cpp 
  Typing.cpp
  Builder.cpp 
  Constants.cpp
  Dialect/Array.cpp 
  Dialect/Bool.cpp 
  Dialect/Cast.cpp 
  Dialect/Constrain.cpp 
  Dialect/Felt.cpp 
  Dialect/Function.cpp 
  Dialect/Global.cpp 
  Dialect/Include.cpp 
  Dialect/LLZK.cpp 
  Dialect/Polymorphic.cpp 
  Dialect/String.cpp 
  Dialect/Struct.cpp 
  Dialect/Undef.cpp)
target_link_libraries(
  LLZKCAPI PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} 
  MLIRIR
  MLIRParser 
  LLVMHeaders 
  MLIRHeaders 
  MLIRCAPIIR 
  MLIRCAPIRegisterEverything 
  LLZK::DialectRegistration
  LLZK::LLZKTransforms 
  LLZK::LLZKValidators
  LLZK::LLZKAnalysis 
)
add_dependencies(LLZKCAPI LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKCAPI)

install(TARGETS LLZKCAPI EXPORT LLZKTargets)
