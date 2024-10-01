#include "Dialect/ZKIR/IR/Types.h"
#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/Util/SymbolHelper.h"

namespace zkir {

StructDefOp
StructType::getDefinition(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) {
  return lookupTopLevelSymbol<StructDefOp>(symbolTable, op, getName());
}

mlir::LogicalResult
StructType::verifySymbol(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) {
  if (!getDefinition(symbolTable, op)) {
    return op->emitOpError() << "undefined component: " << *this;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult ArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType,
    uint64_t numElements
) {
  return checkValidZkirType(emitError, elementType);
}

} // namespace zkir
