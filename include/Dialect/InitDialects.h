#pragma once

#include "Dialect/ZKIR/IR/Dialect.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>

namespace zkir {
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    zkir::ZKIRDialect,
    mlir::arith::ArithDialect,
    mlir::func::FuncDialect,
    mlir::scf::SCFDialect,
    mlir::cf::ControlFlowDialect,
    mlir::memref::MemRefDialect
  >();
  // clang-format on
}
} // namespace zkir
