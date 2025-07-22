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
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
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
using namespace mlir::scf;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::include;
using namespace llzk::polymorphic;

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
      for (auto dim : arrayType.getShape()) {
        llvm::errs() << " x " << dim;
      }
    } else if (auto indexType = type.dyn_cast<IndexType>()) {
      llvm::errs() << "Index.t";
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
      auto value = constOp.getValue();
      if (auto integerAttr = dyn_cast<IntegerAttr>(value)) {
        llvm::errs() << "Integer.Make " << integerAttr.getValue();
      } else {
        llvm::errs() << "Unknown constant value: " << value;
        exit(1);
      }
    } else
    // Array operations
    if (auto createArrayOp = dyn_cast<CreateArrayOp>(operation)) {
      llvm::errs() << "Array.new ";
      for (auto element : createArrayOp.getElements()) {
        printOperand(element);
        llvm::errs() << ", ";
      }
    } else if (auto readArrayOp = dyn_cast<ReadArrayOp>(operation)) {
      llvm::errs() << "Array.read ";
      printOperand(readArrayOp.getArrRef());
      llvm::errs() << "[";
      for (auto index : readArrayOp.getIndices()) {
        printOperand(index);
        llvm::errs() << ", ";
      }
      llvm::errs() << "]";
    } else if (auto extractArrayOp = dyn_cast<ExtractArrayOp>(operation)) {
      llvm::errs() << "Array.extract ";
      printOperand(extractArrayOp.getArrRef());
      llvm::errs() << "[";
      for (auto index : extractArrayOp.getIndices()) {
        printOperand(index);
      }
    } else
    // Constrain operations
    if (auto emitEqualityOp = dyn_cast<EmitEqualityOp>(operation)) {
      llvm::errs() << "EmitEqualityOp ";
      printOperand(emitEqualityOp.getLhs());
      llvm::errs() << " == ";
      printOperand(emitEqualityOp.getRhs());
    } else if (auto emitContainmentOp = dyn_cast<EmitContainmentOp>(operation)) {
      llvm::errs() << "EmitContainmentOp ";
      printOperand(emitContainmentOp.getRhs());
      llvm::errs() << " in ";
      printOperand(emitContainmentOp.getLhs());
    } else
    // Felt operations
    if (auto feltConstantOp = dyn_cast<FeltConstantOp>(operation)) {
      llvm::errs() << "Felt.const " << feltConstantOp.getValue().getValue();
    } else if (auto addFeltOp = dyn_cast<AddFeltOp>(operation)) {
      llvm::errs() << "Felt.add ";
      printOperand(addFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(addFeltOp.getRhs());
    } else if (auto subFeltOp = dyn_cast<SubFeltOp>(operation)) {
      llvm::errs() << "Felt.sub ";
      printOperand(subFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(subFeltOp.getRhs());
    } else if (auto mulFeltOp = dyn_cast<MulFeltOp>(operation)) {
      llvm::errs() << "Felt.mul ";
      printOperand(mulFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(mulFeltOp.getRhs());
    } else if (auto divFeltOp = dyn_cast<DivFeltOp>(operation)) {
      llvm::errs() << "Felt.div ";
      printOperand(divFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(divFeltOp.getRhs());
    } else if (auto modFeltOp = dyn_cast<ModFeltOp>(operation)) {
      llvm::errs() << "Felt.mod ";
      printOperand(modFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(modFeltOp.getRhs());
    } else
    // Function operations
    if (auto returnOp = dyn_cast<ReturnOp>(operation)) {
      llvm::errs() << "Return (";
      for (Value operand : returnOp.getOperands()) {
        printOperand(operand);
        llvm::errs() << ", ";
      }
      llvm::errs() << ")";
    } else if (auto callOp = dyn_cast<CallOp>(operation)) {
      llvm::errs() << "Call " << callOp.getCallee() << " ";
      llvm::errs() << "(";
      for (Value operand : callOp.getArgOperands()) {
        printOperand(operand);
        llvm::errs() << ", ";
      }
      llvm::errs() << ")";
    } else
    // Include operations
    if (auto includeOp = dyn_cast<IncludeOp>(operation)) {
      llvm::errs() << "IncludeOp " << includeOp.getSymName() << " from " << includeOp.getPath();
    } else
    // Polymorphic operations
    if (auto constReadOp = dyn_cast<ConstReadOp>(operation)) {
      llvm::errs() << "ConstReadOp " << constReadOp.getConstName();
    } else
    // Scf operations
    if (auto forOp = dyn_cast<ForOp>(operation)) {
      llvm::errs() << "ForOp ";
      printOperand(forOp.getLowerBound());
      llvm::errs() << " to ";
      printOperand(forOp.getUpperBound());
      llvm::errs() << " step ";
      printOperand(forOp.getStep());
      llvm::errs() << " initArgs: ";
      for (auto initArg : forOp.getInitArgs()) {
        printOperand(initArg);
        llvm::errs() << ", ";
      }
      llvm::errs() << " do\n";
      Region &region = forOp.getRegion();
      for (Block &block : region) {
        for (Operation &op : block) {
          printOperation(&op);
          llvm::errs() << "\n";
        }
      }
    } else if (auto yieldOp = dyn_cast<YieldOp>(operation)) {
      llvm::errs() << "YieldOp (";
      for (Value result : yieldOp.getResults()) {
        printOperand(result);
        llvm::errs() << ", ";
      }
      llvm::errs() << ")";
    } else
    // Struct operations
    if (auto fieldReadOp = dyn_cast<FieldReadOp>(operation)) {
      llvm::errs() << "FieldReadOp ";
      printOperand(fieldReadOp.getComponent());
      llvm::errs() << "." << fieldReadOp.getFieldName();
    } else if (auto fieldWriteOp = dyn_cast<FieldWriteOp>(operation)) {
      llvm::errs() << "FieldWriteOp ";
      printOperand(fieldWriteOp.getComponent());
      llvm::errs() << "." << fieldWriteOp.getFieldName() << " <- ";
      printOperand(fieldWriteOp.getVal());
    } else if (auto createStructOp = dyn_cast<CreateStructOp>(operation)) {
      llvm::errs() << "CreateStructOp";
    } else
    // Unknown operations
    {
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

  void printStructDefOp(StructDefOp structDefOp) {
    llvm::errs() << "Module " << structDefOp.getName() << ".\n";
    llvm::errs() << indent(1) << "Parameter t : Set.\n\n";
    for (auto fieldDefOp : structDefOp.getFieldDefs()) {
      llvm::errs() << indent(1) << "Field " << fieldDefOp.getSymName() << " : ";
      printType(fieldDefOp.getType());
      llvm::errs() << ".\n\n";
    }
    printFunction(1, structDefOp.getComputeFuncOp());
    llvm::errs() << "\n";
    printFunction(1, structDefOp.getConstrainFuncOp());
    llvm::errs() << "\n";
    llvm::errs() << "End " << structDefOp.getName() << ".\n";
  }

  void printTopLevelOperations(ModuleOp moduleOp) {
    for (Operation &operation : moduleOp.getBody()->getOperations()) {
      if (auto funcDefOp = dyn_cast<FuncDefOp>(operation)) {
        llvm::errs() << "\n";
        printFunction(0, funcDefOp);
      } else if (auto includeOp = dyn_cast<IncludeOp>(operation)) {
        llvm::errs() << "\n";
        llvm::errs() << "Require Import " << includeOp.getPath() << " as " << includeOp.getSymName() << ".\n";
      } else if (auto subModuleOp = dyn_cast<ModuleOp>(operation)) {
        printTopLevelOperations(subModuleOp);
      } else if (auto structDefOp = dyn_cast<StructDefOp>(operation)) {
        llvm::errs() << "\n";
        printStructDefOp(structDefOp);
      } else {
        llvm::errs() << "Unknown TopLevel Operation: " << operation.getName() << "\n";
        exit(1);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    llvm::errs() << "Require Import RocqOfLLZK.RocqOfLLZK.\n";

    printTopLevelOperations(moduleOp);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createRocqPass() {
  return std::make_unique<RocqPass>();
};
