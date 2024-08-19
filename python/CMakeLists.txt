##
# Top level targets
##

declare_mlir_python_sources(ZKIRPythonDialectSources)
declare_mlir_python_sources(ZKIRPythonDialectSources.Dialects
  ADD_TO_PARENT ZKIRPythonDialectSources)

# Include the tablegen files
include_directories("${ZKIR_INCLUDE_DIR}")

##
# Dialect bindings
##

# These must go under the mlir module/namespace, as the generated sources use
# relative imports from mlir.

# TODO: need to autogenerate & expose the C extensions or else attributes/types won't
# be supported.

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ZKIRPythonDialectSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ZKIROps.td
  SOURCES dialects/zkir_dialect.py
  DIALECT_NAME zkir_dialect)

##
# Python extensions
##

# All other ZKIR Python bindings go under the zkir package.
declare_mlir_python_sources(ZKIRPythonSources)

declare_mlir_python_sources(ZKIRPythonSources.Core
  ADD_TO_PARENT ZKIRPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/zkir"
  SOURCES
    __init__.py
    registration.py
)

declare_mlir_python_extension(ZKIRPythonExtension
  ADD_TO_PARENT ZKIRPythonSources
  MODULE_NAME _zkirRegistration
  ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/ZKIR/Bindings/Python"
  SOURCES
    ZKIRRegistration.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    ZKIRPythonBindings
)

##
# Final modules
##

add_mlir_python_modules(ZKIRPythonDialectModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/mlir"
  INSTALL_PREFIX python/mlir
  DECLARED_SOURCES
    ZKIRPythonDialectSources
)

# FIXME: MLIR bug?: the MLIRPythonCAPI library is not exported as a CMake target.
# Have to manually find it.
#
# TODO: this only works on macOS for now, need to fix it for Linux.
find_library(MLIRPythonCAPI_DYLIB NAMES MLIRPythonCAPI libMLIRPythonCAPI REQUIRED)
add_library(MLIRPythonCAPI SHARED IMPORTED)
set_target_properties(MLIRPythonCAPI PROPERTIES IMPORTED_LOCATION ${MLIRPythonCAPI_DYLIB})

add_mlir_python_modules(ZKIRPythonExtensionModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/zkir"
  INSTALL_PREFIX python/zkir
  DECLARED_SOURCES
    ZKIRPythonSources
  COMMON_CAPI_LINK_LIBS MLIRPythonCAPI
)
