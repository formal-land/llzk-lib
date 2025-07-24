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
using namespace mlir::detail;
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

  void printAttr(Attribute attr) {
    if (auto symAttr = attr.dyn_cast<mlir::SymbolRefAttr>()) {
      llvm::errs() << symAttr.getRootReference().str();
    } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
      llvm::errs() << intAttr.getValue();
    } else {
      llvm::errs() << "Unknown Attr: " << attr;
      exit(1);
    }
  }

  void printType(bool withParens, Type type) {
    if (type.isa<FeltType>()) {
      llvm::errs() << "Felt.t";
    } else if (StructType structType = type.dyn_cast<StructType>()) {
      auto params = structType.getParams();
      if (params && !params.empty() && withParens) {
        llvm::errs() << "(";
      }
      llvm::errs() << structType.getNameRef().getRootReference().str() << ".t";
      if (params && !params.empty()) {
        for (auto param : params) {
          llvm::errs() << " ";
          printAttr(param);
        }
        if (withParens) {
          llvm::errs() << ")";
        }
      }
    } else if (ArrayType arrayType = type.dyn_cast<ArrayType>()) {
      if (withParens) {
        llvm::errs() << "(";
      }
      llvm::errs() << "Array.t ";
      printType(true, arrayType.getElementType());
      llvm::errs() << " [";
      bool isFirst = true;
      for (auto dimSize : arrayType.getDimensionSizes()) {
        if (!isFirst) {
          llvm::errs() << "; ";
        }
        isFirst = false;
        printAttr(dimSize);
      }
      llvm::errs() << "]";
      if (withParens) {
        llvm::errs() << ")";
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

  void printTypeTuple(bool withParens, llvm::ArrayRef<Type> types) {
    if (types.size() == 0) {
      llvm::errs() << "unit";
    } else {
      bool isFirst = true;
      if (withParens && types.size() > 1) {
        llvm::errs() << "(";
      }
      for (Type type : types) {
        if (!isFirst) {
          llvm::errs() << "* ";
        }
        isFirst = false;
        printType(types.size() <= 1, type);
      }
      if (withParens && types.size() > 1) {
        llvm::errs() << ")";
      }
    }
  }

  std::string getResultNameWithoutPercent(Operation* topLevelOperation, Value value) {
    mlir::AsmState asmState(topLevelOperation);
    std::string s;
    llvm::raw_string_ostream os(s);
    value.print(os, asmState);
    os.flush();

    // Remove leading % if present
    if (!s.empty() && s[0] == '%') {
      s = s.substr(1);
    }

    // Keeping only the continuous part of the name at the beginning
    size_t pos = 0;
    while (pos < s.length() && (std::isalnum(s[pos]) || s[pos] == '_')) {
      pos++;
    }
    s = s.substr(0, pos);

    return s;
  }

  void printOperand(Operation* topLevelOperation, Value operand) {
    if (auto opResult = dyn_cast<OpResult>(operand)) {
      llvm::errs() << "var_" << getResultNameWithoutPercent(topLevelOperation, operand);
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Operation* owningOperation = blockArg.getOwner()->getParentOp();
      if (isa<FuncDefOp>(owningOperation)) {
        llvm::errs() << "arg_fun_" << blockArg.getArgNumber();
      } else if (isa<ForOp>(owningOperation)) {
        Location loc = owningOperation->getLoc();
        if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
          llvm::errs() << "arg_for_" << fileLoc.getLine() << "_" << fileLoc.getColumn();
        } else {
          llvm::errs() << "Unknown Location: " << loc;
          exit(1);
        }
      } else {
        llvm::errs() << "Unknown Owning Operation: " << owningOperation;
        exit(1);
      }
    } else {
      llvm::errs() << "Unknown Operand: " << operand;
      exit(1);
    }
  }

  void printOperation(unsigned level, Operation* topLevelOperation, Operation *operation) {
    llvm::errs() << indent(level);

    // Return
    if (auto returnOp = dyn_cast<ReturnOp>(operation)) {
      llvm::errs() << "M.Pure ";
      if (returnOp.getOperands().size() == 0) {
        llvm::errs() << "tt";
      } else {
        if (returnOp.getOperands().size() > 1) {
          llvm::errs() << "(";
        }
        bool isFirst = true;
        for (Value operand : returnOp.getOperands()) {
          if (!isFirst) {
            llvm::errs() << ", ";
          }
          isFirst = false;
          printOperand(topLevelOperation, operand);
        }
        if (returnOp.getOperands().size() > 1) {
          llvm::errs() << ")";
        }
      }
      return;
    }

    // Yield
    if (auto yieldOp = dyn_cast<YieldOp>(operation)) {
      llvm::errs() << "M.yield ";
      if (yieldOp.getResults().size() == 0) {
        llvm::errs() << "tt";
      } else {
        if (yieldOp.getResults().size() > 1) {
          llvm::errs() << "(";
        }
        bool isFirst = true;
        for (Value result : yieldOp.getResults()) {
          if (!isFirst) {
            llvm::errs() << "; ";
          }
          isFirst = false;
          printOperand(topLevelOperation, result);
        }
        if (yieldOp.getResults().size() > 1) {
          llvm::errs() << ")";
        }
      }
      return;
    }

    bool isPure =
      isa<arith::ConstantOp>(operation) ||
      isa<CreateArrayOp>(operation) ||
      isa<ReadArrayOp>(operation) ||
      isa<ExtractArrayOp>(operation) ||
      isa<FeltConstantOp>(operation) ||
      isa<AddFeltOp>(operation) ||
      isa<SubFeltOp>(operation) ||
      isa<MulFeltOp>(operation) ||
      isa<DivFeltOp>(operation) ||
      isa<ModFeltOp>(operation) ||
      isa<ConstReadOp>(operation) ||
      isa<FieldReadOp>(operation);
    llvm::errs() << (isPure ? "let " : "let* ");

    if (operation->getResults().size() == 0) {
      llvm::errs() << "_";
    } else {
      bool isFirst = true;
      if (operation->getResults().size() > 1) {
        llvm::errs() << "(";
      }
      for (mlir::Value result : operation->getResults()) {
        if (!isFirst) {
          llvm::errs() << ", ";
        }
        isFirst = false;
        llvm::errs() << "var_" << getResultNameWithoutPercent(topLevelOperation, result);
      }
      if (operation->getResults().size() > 1) {
        llvm::errs() << ")";
      }
    }
    llvm::errs() << " : ";
    auto resultTypes = operation->getResultTypes();
    if (resultTypes.size() == 0) {
      llvm::errs() << "unit";
    } else {
      bool isFirst = true;
      for (Type type : resultTypes) {
        if (!isFirst) {
          llvm::errs() << "* ";
        }
        isFirst = false;
        printType(false, type);
      }
    }
    llvm::errs() << " := ";

    // If it’s a constant, extract the value
    if (auto constOp = dyn_cast<arith::ConstantOp>(operation)) {
      auto value = constOp.getValue();
      if (auto integerAttr = dyn_cast<IntegerAttr>(value)) {
        llvm::errs() << integerAttr.getValue();
      } else {
        llvm::errs() << "Unknown constant value: " << value;
        exit(1);
      }
    } else
    // Array operations
    if (auto createArrayOp = dyn_cast<CreateArrayOp>(operation)) {
      llvm::errs() << "Array.new [";
      bool isFirst = true;
      for (auto element : createArrayOp.getElements()) {
        if (!isFirst) {
          llvm::errs() << "; ";
        }
        isFirst = false;
        printOperand(topLevelOperation, element);
      }
      llvm::errs() << "]";
    } else if (auto readArrayOp = dyn_cast<ReadArrayOp>(operation)) {
      llvm::errs() << "Array.read ";
      printOperand(topLevelOperation, readArrayOp.getArrRef());
      llvm::errs() << " (";
      for (auto index : readArrayOp.getIndices()) {
        printOperand(topLevelOperation, index);
        llvm::errs() << ", ";
      }
      llvm::errs() << "tt)";
    } else if (auto extractArrayOp = dyn_cast<ExtractArrayOp>(operation)) {
      llvm::errs() << "Array.extract ";
      unsigned resultSize = 0;
      for (Type type : resultTypes) {
        if (auto arrayType = type.dyn_cast<ArrayType>()) {
          resultSize = arrayType.getDimensionSizes().size();
        } else {
          llvm::errs() << "Expected array result type, got " << type;
          exit(1);
        }
      }
      llvm::errs() << "(Ns := [";
      for (unsigned i = 0; i < resultSize; i++) {
        llvm::errs() << "_";
        if (i < resultSize - 1) {
          llvm::errs() << "; ";
        }
      }
      llvm::errs() << "]) ";
      printOperand(topLevelOperation, extractArrayOp.getArrRef());
      llvm::errs() << " (";
      for (auto index : extractArrayOp.getIndices()) {
        printOperand(topLevelOperation, index);
        llvm::errs() << ", ";
      }
      llvm::errs() << "tt)";
    } else
    // Constrain operations
    if (auto emitEqualityOp = dyn_cast<EmitEqualityOp>(operation)) {
      llvm::errs() << "M.AssertEqual ";
      printOperand(topLevelOperation, emitEqualityOp.getLhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, emitEqualityOp.getRhs());
    } else if (auto emitContainmentOp = dyn_cast<EmitContainmentOp>(operation)) {
      llvm::errs() << "M.AssertIn ";
      printOperand(topLevelOperation, emitContainmentOp.getRhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, emitContainmentOp.getLhs());
    } else
    // Felt operations
    if (auto feltConstantOp = dyn_cast<FeltConstantOp>(operation)) {
      llvm::errs() << "UnOp.from " << feltConstantOp.getValue().getValue();
    } else if (auto addFeltOp = dyn_cast<AddFeltOp>(operation)) {
      llvm::errs() << "BinOp.add ";
      printOperand(topLevelOperation, addFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, addFeltOp.getRhs());
    } else if (auto subFeltOp = dyn_cast<SubFeltOp>(operation)) {
      llvm::errs() << "BinOp.sub ";
      printOperand(topLevelOperation, subFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, subFeltOp.getRhs());
    } else if (auto mulFeltOp = dyn_cast<MulFeltOp>(operation)) {
      llvm::errs() << "BinOp.mul ";
      printOperand(topLevelOperation, mulFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, mulFeltOp.getRhs());
    } else if (auto divFeltOp = dyn_cast<DivFeltOp>(operation)) {
      llvm::errs() << "BinOp.div ";
      printOperand(topLevelOperation, divFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, divFeltOp.getRhs());
    } else if (auto modFeltOp = dyn_cast<ModFeltOp>(operation)) {
      llvm::errs() << "BinOp.mod ";
      printOperand(topLevelOperation, modFeltOp.getLhs());
      llvm::errs() << " ";
      printOperand(topLevelOperation, modFeltOp.getRhs());
    } else
    // Function operations
    if (auto callOp = dyn_cast<CallOp>(operation)) {
      auto callee = callOp.getCallee();
      llvm::errs() << callee.getRootReference().str();
      for (auto nestedRef : callee.getNestedReferences()) {
        llvm::errs() << "." << nestedRef.getRootReference().str();
      }
      for (Value operand : callOp.getArgOperands()) {
        llvm::errs() << " ";
        printOperand(topLevelOperation, operand);
      }
    } else
    // Polymorphic operations
    if (auto constReadOp = dyn_cast<ConstReadOp>(operation)) {
      llvm::errs() << "UnOp.from (Z.of_nat " << constReadOp.getConstName() << ")";
    } else
    // Scf operations
    if (auto forOp = dyn_cast<ForOp>(operation)) {
      llvm::errs() << "M.for_ ";
      printOperand(topLevelOperation, forOp.getLowerBound());
      llvm::errs() << " (* to *) ";
      printOperand(topLevelOperation, forOp.getUpperBound());
      llvm::errs() << " (* step *) ";
      printOperand(topLevelOperation, forOp.getStep());
      llvm::errs() << " (fun";
      Region &region = forOp.getRegion();
      // We assume there is only one block
      for (Block &block : region) {
        if (block.getArguments().size() != 1) {
          llvm::errs() << "Expected 1 argument for the loop body, got ";
          llvm::errs() << block.getArguments().size();
          exit(1);
        }
        for (BlockArgument arg : block.getArguments()) {
          llvm::errs() << " (";
          printOperand(topLevelOperation, arg);
          llvm::errs() << " : ";
          printType(false, arg.getType());
          llvm::errs() << ")";
        }
        llvm::errs() << " =>\n";
        for (Operation &op : block) {
          printOperation(level + 1, topLevelOperation, &op);
        }
      }
      llvm::errs() << indent(level) << ")";
    } else
    // Struct operations
    if (auto fieldReadOp = dyn_cast<FieldReadOp>(operation)) {
      printOperand(topLevelOperation, fieldReadOp.getComponent());
      llvm::errs() << ".(";
      llvm::errs() << fieldReadOp.getComponent().getType().getNameRef().getRootReference().str();
      llvm::errs() << "." << fieldReadOp.getFieldName() << ")";
    } else if (auto fieldWriteOp = dyn_cast<FieldWriteOp>(operation)) {
      llvm::errs() << "M.FieldWrite ";
      printOperand(topLevelOperation, fieldWriteOp.getComponent());
      llvm::errs() << ".(";
      llvm::errs() << fieldWriteOp.getComponent().getType().getNameRef().getRootReference().str();
      llvm::errs() << "." << fieldWriteOp.getFieldName() << ") ";
      printOperand(topLevelOperation, fieldWriteOp.getVal());
    } else if (auto createStructOp = dyn_cast<CreateStructOp>(operation)) {
      llvm::errs() << "M.CreateStruct";
    } else
    // Unknown operations
    {
      llvm::errs() << "Unknown Operation: " << operation->getName();
      exit(1);
    }
    llvm::errs() << " in\n";
  }

  void printConstParams(StructDefOp* structDefOp, bool isImplicit) {
    auto constParams = structDefOp->getConstParamsAttr();
    if (!constParams || constParams.empty()) {
      return;
    }
    llvm::errs() << " " << (isImplicit ? "{" : "(");
    bool isFirst = true;
    for (auto constParam : constParams) {
      if (!isFirst) {
        llvm::errs() << " ";
      }
      isFirst = false;
      printAttr(constParam);
    }
    llvm::errs() << " : nat" << (isImplicit ? "}" : ")");
  }

  bool hasConstParams(StructDefOp* structDefOp) {
    auto constParams = structDefOp->getConstParamsAttr();
    return constParams && !constParams.empty();
  }

  void printFunction(
    unsigned level,
    Operation* topLevelOperation,
    StructDefOp* structDefOp,
    FuncDefOp func
  ) {
    llvm::errs() << indent(level) << "Definition " << func.getName() << " {p} `{Prime p}";
    if (structDefOp) {
      printConstParams(structDefOp, true);
    }
    unsigned argTypeIndex = 0;
    for (auto argType : func.getArgumentTypes()) {
      llvm::errs() << " (arg_fun_" << argTypeIndex << " : ";
      printType(false, argType);
      llvm::errs() << ")";
      argTypeIndex++;
    }
    llvm::errs() << " : M.t ";
    llvm::ArrayRef<Type> results = func.getFunctionType().getResults();
    printTypeTuple(true, results);
    if (func.isExternal()) {
      llvm::errs() << ".\n";
      llvm::errs() << indent(level) << "Admitted.\n";
      return;
    }
    llvm::errs() << " :=\n";

    for (Block &block : func.getBody()) {
      for (Operation &op : block) {
        printOperation(level + 1, topLevelOperation, &op);
      }
    }

    llvm::errs() << ".\n";
  }

  void printStructDefOp(unsigned level, Operation* topLevelOperation, StructDefOp* structDefOp) {
    llvm::errs() << indent(level) << "Module " << structDefOp->getName() << ".\n";
    // Special case when there are no fields
    if (structDefOp->getFieldDefs().size() == 0) {
      llvm::errs() << indent(level + 1) << "Inductive t";
      printConstParams(structDefOp, false);
      llvm::errs() << " : Set := Make.";
    } else {
      llvm::errs() << indent(level + 1) << "Record t";
      printConstParams(structDefOp, true);
      llvm::errs() << " : Set := {\n";
      for (auto fieldDefOp : structDefOp->getFieldDefs()) {
        llvm::errs() << indent(level + 2) << fieldDefOp.getSymName() << " : ";
        printType(false, fieldDefOp.getType());
        llvm::errs() << ";\n";
      }
      llvm::errs() << indent(level + 1) << "}.";
      if (hasConstParams(structDefOp)) {
        llvm::errs() << "\n";
        llvm::errs() << indent(level + 1) << "Arguments t : clear implicits.";
      }
    }
    llvm::errs() << "\n\n";
    printFunction(level + 1, topLevelOperation, structDefOp, structDefOp->getConstrainFuncOp());
    llvm::errs() << "\n";
    printFunction(level + 1, topLevelOperation, structDefOp, structDefOp->getComputeFuncOp());
    llvm::errs() << indent(level) << "End " << structDefOp->getName() << ".\n";
  }

  void printTopLevelOperations(unsigned level, Operation* topLevelOperation, ModuleOp* moduleOp) {
    for (Operation &operation : moduleOp->getBody()->getOperations()) {
      if (auto funcDefOp = dyn_cast<FuncDefOp>(operation)) {
        llvm::errs() << "\n";
        printFunction(level, topLevelOperation, nullptr, funcDefOp);
      } else if (auto includeOp = dyn_cast<IncludeOp>(operation)) {
        llvm::errs() << "\n";
        llvm::errs() << indent(level) << "(* Require Import ";
        // Substitute the "/" with "."
        std::string path = includeOp.getPath().str();
        std::replace(path.begin(), path.end(), '/', '.');
        // Remove the ".llzk" suffix
        path = path.substr(0, path.size() - 5);
        llvm::errs() << path << " as " << includeOp.getSymName() << ". *)\n";
      } else if (auto subModuleOp = dyn_cast<ModuleOp>(operation)) {
        mlir::Location loc = operation.getLoc();
        std::string moduleName = "Anonymous";
        if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
          moduleName = "Line_" +std::to_string(fileLoc.getLine());
        }
        llvm::errs() << "\n";
        llvm::errs() << indent(level) << "Module Module_" << moduleName << ".";
        printTopLevelOperations(level + 1, topLevelOperation, &subModuleOp);
        llvm::errs() << indent(level) << "End Module_" << moduleName << ".\n";
      } else if (auto structDefOp = dyn_cast<StructDefOp>(operation)) {
        llvm::errs() << "\n";
        printStructDefOp(level, topLevelOperation, &structDefOp);
      } else {
        llvm::errs() << "Unknown TopLevel Operation: " << operation.getName() << "\n";
        exit(1);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    llvm::errs() << "(* Generated *)\n";
    llvm::errs() << "Require Import Garden.LLZK.M.\n";

    printTopLevelOperations(0, moduleOp.getOperation(), &moduleOp);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createRocqPass() {
  return std::make_unique<RocqPass>();
};
