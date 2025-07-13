//===-- LLZKRocqPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-rocq` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DECL_ROCQPASS
#define GEN_PASS_DEF_ROCQPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

#define DEBUG_TYPE "llzk-rocq-pass"
#define AUXILIARY_FIELD_PREFIX "__llzk_rocq_pass_aux_field_"

namespace {

class RocqPass : public llzk::impl::RocqPassBase<RocqPass> {

private:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([this, &moduleOp](StructDefOp structDef) {
      FuncDefOp constrainFunc = structDef.getConstrainFuncOp();

      llvm::errs() << '"' << structDef.getName() << "\"\n";
      constrainFunc.walk([&](Operation *op) {
        llvm::errs() << '"' << op->getName() << "\"\n";
        // op->print(llvm::errs());
        llvm::errs() << "Operands: ";
        for (Value operand : op->getOperands()) {
          llvm::errs() << "operand: " << operand << "\n";
          // OpResult
          if (auto opResult = dyn_cast<OpResult>(operand)) {
            llvm::outs() << "OpResult: " << opResult.getResultNumber() << "\n";
          }
          if (auto definingOp = operand.getDefiningOp()) {
            // definingOp->dump(); // Print the op that defines this operand
        
            // If it’s a constant, extract the value
            if (auto constOp = dyn_cast<arith::ConstantOp>(definingOp)) {
              auto attr = constOp.getValue();
              llvm::outs() << "Constant: " << attr << "\n";
        
              // If it's an integer:
              if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                llvm::outs() << "Value: " << intAttr.getValue() << "\n";
              }
        
              // If it's a float:
              if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
                llvm::outs() << "Value: " << floatAttr.getValueAsDouble() << "\n";
              }
            }
          } else {
            // This operand is a block argument, not a result of another op
            if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
              llvm::outs() << "Block argument #" << blockArg.getArgNumber() << "\n";
            }
          }
        }
        llvm::errs() << '\n';
        // dyn type check
        // if (auto emitOp = dyn_cast<EmitEqualityOp>(op)) {
        //   llvm::errs() << "EmitEqualityOp\n";
        //   llvm::errs() << '"' << emitOp.getLhs() << "\"\n";
        //   llvm::errs() << '"' << emitOp.getRhs() << "\"\n";
        //   llvm::errs() << '\n';
        // }
      });
      llvm::errs() << '\n';
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createRocqPass() {
  return std::make_unique<RocqPass>();
};
