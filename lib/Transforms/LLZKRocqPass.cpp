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
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;

#define DEBUG_TYPE "llzk-rocq-pass"
#define AUXILIARY_FIELD_PREFIX "__llzk_rocq_pass_aux_field_"

namespace {

class RocqPass : public llzk::impl::RocqPassBase<RocqPass> {

private:
  std::string indent(unsigned level) {
    return std::string(level * 2, ' ');
  }

  void printType(Type type) {
    if (type.isa<FeltType>()) {
      llvm::errs() << "Felt.t";
    } else if (StructType structType = type.dyn_cast<StructType>()) {
      llvm::errs() << structType.getNameRef().getRootReference().str() << ".t";
    } else if (ArrayType arrayType = type.dyn_cast<ArrayType>()) {
      llvm::errs() << "Array.t ";
      printType(arrayType.getElementType());
    } else {
      llvm::errs() << "Unknown type";
      type.dump();
      // TODO: remove the exit at some point
      exit(1);
    }
  }

  void printTypeTuple(llvm::ArrayRef<Type> types) {
    if (types.size() == 0) {
      llvm::errs() << "unit";
    } else {
      bool isFirst = true;
      for (Type type : types) {
        if (!isFirst) {
          llvm::errs() << "* ";
        }
        isFirst = false;
        printType(type);
      }
    }
  }

  void printOperand(Value operand) {
    if (auto opResult = dyn_cast<OpResult>(operand)) {
      llvm::errs() << "result" << opResult.getResultNumber();
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      llvm::errs() << "arg" << blockArg.getArgNumber();
    } else {
      llvm::errs() << "Unknown Operand: " << operand;
      exit(1);
    }
  }

  void printOperation(Operation *operation) {
    // If it’s a constant, extract the value
    if (auto constOp = dyn_cast<arith::ConstantOp>(operation)) {
      auto attr = constOp.getValue();
      llvm::errs() << "Constant: " << attr;

      // If it's an integer:
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        llvm::errs() << "Value: " << intAttr.getValue();
      }

      // If it's a float:
      if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
        llvm::errs() << "Value: " << floatAttr.getValueAsDouble();
      }
    } else if (auto createStructOp = dyn_cast<CreateStructOp>(operation)) {
      llvm::errs() << "CreateStructOp";
    } else if (auto returnOp = dyn_cast<ReturnOp>(operation)) {
      llvm::errs() << "ReturnOp (";
      for (Value operand : returnOp.getOperands()) {
        printOperand(operand);
        llvm::errs() << ", ";
      }
      llvm::errs() << ")";
    } else {
      llvm::errs() << "Unknown Operation: " << operation->getName();
      exit(1);
    }
  }

  void printFunction(unsigned level, FuncDefOp func) {
    llvm::errs() << indent(level) << "Definition " << func.getName();
    for (auto arg : func.getArguments()) {
      llvm::errs() << " (arg" << arg.getArgNumber() << " : ";
      printType(arg.getType());
      llvm::errs() << ")";
    }
    llvm::errs() << " : ";
    llvm::ArrayRef<Type> results = func.getFunctionType().getResults();
    printTypeTuple(results);
    llvm::errs() << " :=\n";

    func.walk([&](Operation *op) {
      // Skip if this is a function definition, as it itself 
      if (isa<FuncDefOp>(op)) {
        return;
      }

      llvm::errs() << indent(level + 1) << "do! ";
      printOperation(op);
      llvm::errs() << " : ";
      auto resultTypes = op->getResultTypes();
      if (resultTypes.size() == 0) {
        llvm::errs() << "unit";
      } else {
        for (Type type : resultTypes) {
          printType(type);
        }
      }
      llvm::errs() << " in\n";
    });

    llvm::errs() << indent(level + 1) << "done.\n";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    unsigned level = 0;

    llvm::errs() << "Require Import RocqOfLLZK.RocqOfLLZK.\n";

    moduleOp.walk([this, &moduleOp, &level](StructDefOp structDef) {
      llvm::errs() << "\n";
      llvm::errs() << "Module " << structDef.getName() << ".\n";
      level++;
      printFunction(level, structDef.getComputeFuncOp());
      llvm::errs() << "\n";
      printFunction(level, structDef.getConstrainFuncOp());
      llvm::errs() << "End " << structDef.getName() << ".\n";
      level--;
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createRocqPass() {
  return std::make_unique<RocqPass>();
};
