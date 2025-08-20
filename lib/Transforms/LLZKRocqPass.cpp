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
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/String/IR/Ops.h"
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
using namespace llzk::boolean;
using namespace llzk::cast;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::include;
using namespace llzk::polymorphic;
using namespace llzk::string;

#define DEBUG_TYPE "llzk-rocq-pass"
#define AUXILIARY_FIELD_PREFIX "__llzk_rocq_pass_aux_field_"

namespace {

class RocqPass : public llzk::impl::RocqPassBase<RocqPass> {

private:
  std::string indent(unsigned level) {
    return std::string(level * 2, ' ');
  }

  std::string escapeName(std::string name) {
    std::string result = name;
    size_t pos = result.find('$');
    while (pos != std::string::npos) {
      const std::string replacement = "dollar_";
      result.replace(pos, 1, replacement);
      pos = result.find('$', pos + replacement.length());
    }
    return result;
  }

  void printAttr(Attribute attr) {
    if (auto symAttr = attr.dyn_cast<mlir::SymbolRefAttr>()) {
      llvm::outs() << symAttr.getRootReference().str();
    } else if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
      int64_t value = intAttr.getValue().getSExtValue();
      if (value == std::numeric_limits<int64_t>::min()) {
        llvm::outs() << "Array.dimension_any";
      } else {
        llvm::outs() << value;
      }
    } else if (auto affineMapAttr = attr.dyn_cast<mlir::AffineMapAttr>()) {
      llvm::outs() << "Array.affine_map";
    } else if (auto typeAttr = attr.dyn_cast<mlir::TypeAttr>()) {
      printType(false, typeAttr.getValue());
    } else {
      llvm::outs() << "Unknown Attr: " << attr;
      exit(1);
    }
  }

  void printType(bool withParens, Type type) {
    if (type.isa<FeltType>()) {
      llvm::outs() << "Felt.t";
    } else if (auto intType = type.dyn_cast<IntegerType>()) {
      if (intType.getWidth() == 1) {
        llvm::outs() << "bool";
      } else {
        llvm::outs() << "Unknown integer type: " << intType;
        exit(1);
      }
    } else if (type.isa<StringType>()) {
      llvm::outs() << "string";
    } else if (StructType structType = type.dyn_cast<StructType>()) {
      auto params = structType.getParams();
      if (params && !params.empty() && withParens) {
        llvm::outs() << "(";
      }
      llvm::outs() << structType.getNameRef().getRootReference().str() << ".t";
      if (params && !params.empty()) {
        for (auto param : params) {
          llvm::outs() << " ";
          printAttr(param);
        }
        if (withParens) {
          llvm::outs() << ")";
        }
      }
    } else if (ArrayType arrayType = type.dyn_cast<ArrayType>()) {
      if (withParens) {
        llvm::outs() << "(";
      }
      llvm::outs() << "Array.t ";
      printType(true, arrayType.getElementType());
      llvm::outs() << " [";
      bool isFirst = true;
      for (auto dimSize : arrayType.getDimensionSizes()) {
        if (!isFirst) {
          llvm::outs() << "; ";
        }
        isFirst = false;
        printAttr(dimSize);
      }
      llvm::outs() << "]%nat";
      if (withParens) {
        llvm::outs() << ")";
      }
    // Index
    } else if (auto indexType = type.dyn_cast<IndexType>()) {
      llvm::outs() << "Index.t";
    // TypeVar
    } else if (auto typeVarType = type.dyn_cast<TypeVarType>()) {
      llvm::outs() << "TypeVar.t ";
      llvm::outs() << typeVarType.getNameRef();
    } else {
      llvm::outs() << "Unknown type";
      type.dump();
      // TODO: remove the exit at some point
      exit(1);
    }
  }

  void printTypeTuple(bool withParens, llvm::ArrayRef<Type> types) {
    if (types.size() == 0) {
      llvm::outs() << "unit";
    } else {
      bool isFirst = true;
      if (withParens && types.size() > 1) {
        llvm::outs() << "(";
      }
      for (Type type : types) {
        if (!isFirst) {
          llvm::outs() << "* ";
        }
        isFirst = false;
        printType(types.size() <= 1, type);
      }
      if (withParens && types.size() > 1) {
        llvm::outs() << ")";
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
      llvm::outs() << "var_" << getResultNameWithoutPercent(topLevelOperation, operand);
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Operation* owningOperation = blockArg.getOwner()->getParentOp();
      if (isa<FuncDefOp>(owningOperation)) {
        llvm::outs() << "arg_fun_" << blockArg.getArgNumber();
      } else if (isa<ForOp>(owningOperation)) {
        Location loc = owningOperation->getLoc();
        if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
          llvm::outs() << "arg_for_" << fileLoc.getLine() << "_" << fileLoc.getColumn();
        } else {
          llvm::outs() << "Unknown Location: " << loc;
          exit(1);
        }
      } else {
        llvm::outs() << "Unknown Owning Operation: " << owningOperation;
        exit(1);
      }
    } else {
      llvm::outs() << "Unknown Operand: " << operand;
      exit(1);
    }
  }

  void printOperation(unsigned level, Operation* topLevelOperation, Operation *operation) {
    llvm::outs() << indent(level);

    // Return
    if (auto returnOp = dyn_cast<ReturnOp>(operation)) {
      llvm::outs() << "M.Pure ";
      if (returnOp.getOperands().size() == 0) {
        llvm::outs() << "tt";
      } else {
        if (returnOp.getOperands().size() > 1) {
          llvm::outs() << "(";
        }
        bool isFirst = true;
        for (Value operand : returnOp.getOperands()) {
          if (!isFirst) {
            llvm::outs() << ", ";
          }
          isFirst = false;
          printOperand(topLevelOperation, operand);
        }
        if (returnOp.getOperands().size() > 1) {
          llvm::outs() << ")";
        }
      }
      return;
    }

    // Yield
    if (auto yieldOp = dyn_cast<YieldOp>(operation)) {
      llvm::outs() << "M.yield ";
      if (yieldOp.getResults().size() == 0) {
        llvm::outs() << "tt";
      } else {
        if (yieldOp.getResults().size() > 1) {
          llvm::outs() << "(";
        }
        bool isFirst = true;
        for (Value result : yieldOp.getResults()) {
          if (!isFirst) {
            llvm::outs() << "; ";
          }
          isFirst = false;
          printOperand(topLevelOperation, result);
        }
        if (yieldOp.getResults().size() > 1) {
          llvm::outs() << ")";
        }
      }
      return;
    }

    bool isPure =
      isa<arith::ConstantOp>(operation) ||
      isa<CreateArrayOp>(operation) ||
      isa<CmpOp>(operation) ||
      isa<AndBoolOp>(operation) ||
      isa<OrBoolOp>(operation) ||
      isa<XorBoolOp>(operation) ||
      isa<NotBoolOp>(operation) ||
      isa<ReadArrayOp>(operation) ||
      isa<ExtractArrayOp>(operation) ||
      isa<FeltConstantOp>(operation) ||
      isa<IntToFeltOp>(operation) ||
      isa<FeltToIndexOp>(operation) ||
      isa<AddFeltOp>(operation) ||
      isa<SubFeltOp>(operation) ||
      isa<MulFeltOp>(operation) ||
      isa<DivFeltOp>(operation) ||
      isa<ModFeltOp>(operation) ||
      isa<NegFeltOp>(operation) ||
      isa<InvFeltOp>(operation) ||
      isa<ConstReadOp>(operation) ||
      isa<FieldReadOp>(operation);
    llvm::outs() << (isPure ? "let " : "let* ");

    if (operation->getResults().size() == 0) {
      llvm::outs() << "_";
    } else {
      bool isFirst = true;
      if (operation->getResults().size() > 1) {
        llvm::outs() << "(";
      }
      for (mlir::Value result : operation->getResults()) {
        if (!isFirst) {
          llvm::outs() << ", ";
        }
        isFirst = false;
        llvm::outs() << "var_" << getResultNameWithoutPercent(topLevelOperation, result);
      }
      if (operation->getResults().size() > 1) {
        llvm::outs() << ")";
      }
    }
    llvm::outs() << " : ";
    auto resultTypes = operation->getResultTypes();
    if (resultTypes.size() == 0) {
      llvm::outs() << "unit";
    } else {
      bool isFirst = true;
      for (Type type : resultTypes) {
        if (!isFirst) {
          llvm::outs() << "* ";
        }
        isFirst = false;
        printType(false, type);
      }
    }
    llvm::outs() << " := ";

    // If it’s a constant, extract the value
    if (auto constOp = dyn_cast<arith::ConstantOp>(operation)) {
      auto value = constOp.getValue();
      if (auto integerAttr = dyn_cast<IntegerAttr>(value)) {
        llvm::outs() << integerAttr.getValue() << "%nat";
      } else {
        llvm::outs() << "Unknown constant value: " << value;
        exit(1);
      }
    } else
    // Arith operations
    if (auto selectOp = dyn_cast<arith::SelectOp>(operation)) {
      llvm::outs() << "Arith.select ";
      printOperand(topLevelOperation, selectOp.getCondition());
      llvm::outs() << " ";
      printOperand(topLevelOperation, selectOp.getTrueValue());
      llvm::outs() << " ";
      printOperand(topLevelOperation, selectOp.getFalseValue());
    } else
    // Array operations
    if (auto createArrayOp = dyn_cast<CreateArrayOp>(operation)) {
      llvm::outs() << "Array.new [";
      bool isFirst = true;
      for (auto element : createArrayOp.getElements()) {
        if (!isFirst) {
          llvm::outs() << "; ";
        }
        isFirst = false;
        printOperand(topLevelOperation, element);
      }
      llvm::outs() << "]";
    } else if (auto readArrayOp = dyn_cast<ReadArrayOp>(operation)) {
      llvm::outs() << "Array.read ";
      printOperand(topLevelOperation, readArrayOp.getArrRef());
      llvm::outs() << " (";
      for (auto index : readArrayOp.getIndices()) {
        printOperand(topLevelOperation, index);
        llvm::outs() << ", ";
      }
      llvm::outs() << "tt)";
    } else if (auto writeArrayOp = dyn_cast<WriteArrayOp>(operation)) {
      llvm::outs() << "M.ArrayWrite ";
      printOperand(topLevelOperation, writeArrayOp.getArrRef());
      llvm::outs() << " (";
      for (auto index : writeArrayOp.getIndices()) {
        printOperand(topLevelOperation, index);
        llvm::outs() << ", ";
      }
      llvm::outs() << "tt) ";
      printOperand(topLevelOperation, writeArrayOp.getRvalue());
    } else if (auto extractArrayOp = dyn_cast<ExtractArrayOp>(operation)) {
      llvm::outs() << "Array.extract ";
      unsigned resultSize = 0;
      for (Type type : resultTypes) {
        if (auto arrayType = type.dyn_cast<ArrayType>()) {
          resultSize = arrayType.getDimensionSizes().size();
        } else {
          llvm::outs() << "Expected array result type, got " << type;
          exit(1);
        }
      }
      llvm::outs() << "(Ns := [";
      for (unsigned i = 0; i < resultSize; i++) {
        llvm::outs() << "_";
        if (i < resultSize - 1) {
          llvm::outs() << "; ";
        }
      }
      llvm::outs() << "]) ";
      printOperand(topLevelOperation, extractArrayOp.getArrRef());
      llvm::outs() << " (";
      for (auto index : extractArrayOp.getIndices()) {
        printOperand(topLevelOperation, index);
        llvm::outs() << ", ";
      }
      llvm::outs() << "tt)";
    } else if (auto insertArrayOp = dyn_cast<InsertArrayOp>(operation)) {
      llvm::outs() << "Array.insert ";
      printOperand(topLevelOperation, insertArrayOp.getArrRef());
      llvm::outs() << " (";
      for (auto index : insertArrayOp.getIndices()) {
        printOperand(topLevelOperation, index);
        llvm::outs() << ", ";
      }
      llvm::outs() << "tt) ";
      printOperand(topLevelOperation, insertArrayOp.getRvalue());
    } else if (auto lenArrayOp = dyn_cast<ArrayLengthOp>(operation)) {
      llvm::outs() << "Array.len ";
      printOperand(topLevelOperation, lenArrayOp.getArrRef());
      llvm::outs() << " ";
      printOperand(topLevelOperation, lenArrayOp.getDim());
    } else
    // Bool operations
    if (auto cmpOp = dyn_cast<boolean::CmpOp>(operation)) {
      llvm::outs() << "Bool.cmp ";
      FeltCmpPredicate predicate = cmpOp.getPredicate();
      switch (predicate) {
        case FeltCmpPredicate::EQ:
          llvm::outs() << "BoolCmp.Eq";
          break;
        case FeltCmpPredicate::NE:
          llvm::outs() << "BoolCmp.Ne";
          break;
        case FeltCmpPredicate::LT:
          llvm::outs() << "BoolCmp.Lt";
          break;
        case FeltCmpPredicate::LE:
          llvm::outs() << "BoolCmp.Le";
          break;
        case FeltCmpPredicate::GT:
          llvm::outs() << "BoolCmp.Gt";
          break;
        case FeltCmpPredicate::GE:
          llvm::outs() << "BoolCmp.Ge";
          break;
        default:
          llvm::outs() << "Unknown predicate: " << predicate;
          exit(1);
      }
      llvm::outs() << " ";
      printOperand(topLevelOperation, cmpOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, cmpOp.getRhs());
    } else if (auto andBoolOp = dyn_cast<AndBoolOp>(operation)) {
      llvm::outs() << "Bool.and ";
      printOperand(topLevelOperation, andBoolOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, andBoolOp.getRhs());
    } else if (auto orBoolOp = dyn_cast<OrBoolOp>(operation)) {
      llvm::outs() << "Bool.or ";
      printOperand(topLevelOperation, orBoolOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, orBoolOp.getRhs());
    } else if (auto xorBoolOp = dyn_cast<XorBoolOp>(operation)) {
      llvm::outs() << "Bool.xor ";
      printOperand(topLevelOperation, xorBoolOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, xorBoolOp.getRhs());
    } else if (auto notBoolOp = dyn_cast<NotBoolOp>(operation)) {
      llvm::outs() << "Bool.not ";
      printOperand(topLevelOperation, notBoolOp.getOperand());
    } else if (auto assertOp = dyn_cast<AssertOp>(operation)) {
      llvm::outs() << "M.AssertBool ";
      printOperand(topLevelOperation, assertOp.getCondition());
    } else
    // Cast operations
    if (auto intToFeltOp = dyn_cast<IntToFeltOp>(operation)) {
      llvm::outs() << "M.cast_to_felt ";
      printOperand(topLevelOperation, intToFeltOp.getOperand());
    } else if (auto feltToIndexOp = dyn_cast<FeltToIndexOp>(operation)) {
      llvm::outs() << "M.cast_to_index ";
      printOperand(topLevelOperation, feltToIndexOp.getOperand());
    } else
    // Constrain operations
    if (auto emitEqualityOp = dyn_cast<EmitEqualityOp>(operation)) {
      llvm::outs() << "M.AssertEqual ";
      printOperand(topLevelOperation, emitEqualityOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, emitEqualityOp.getRhs());
    } else if (auto emitContainmentOp = dyn_cast<EmitContainmentOp>(operation)) {
      llvm::outs() << "M.AssertIn ";
      printOperand(topLevelOperation, emitContainmentOp.getRhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, emitContainmentOp.getLhs());
    } else
    // Felt operations
    if (auto feltConstantOp = dyn_cast<FeltConstantOp>(operation)) {
      llvm::outs() << "UnOp.from " << feltConstantOp.getValue().getValue();
    } else if (auto addFeltOp = dyn_cast<AddFeltOp>(operation)) {
      llvm::outs() << "BinOp.add ";
      printOperand(topLevelOperation, addFeltOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, addFeltOp.getRhs());
    } else if (auto subFeltOp = dyn_cast<SubFeltOp>(operation)) {
      llvm::outs() << "BinOp.sub ";
      printOperand(topLevelOperation, subFeltOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, subFeltOp.getRhs());
    } else if (auto mulFeltOp = dyn_cast<MulFeltOp>(operation)) {
      llvm::outs() << "BinOp.mul ";
      printOperand(topLevelOperation, mulFeltOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, mulFeltOp.getRhs());
    } else if (auto divFeltOp = dyn_cast<DivFeltOp>(operation)) {
      llvm::outs() << "BinOp.div ";
      printOperand(topLevelOperation, divFeltOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, divFeltOp.getRhs());
    } else if (auto modFeltOp = dyn_cast<ModFeltOp>(operation)) {
      llvm::outs() << "BinOp.mod ";
      printOperand(topLevelOperation, modFeltOp.getLhs());
      llvm::outs() << " ";
      printOperand(topLevelOperation, modFeltOp.getRhs());
    } else if (auto negFeltOp = dyn_cast<NegFeltOp>(operation)) {
      llvm::outs() << "UnOp.neg ";
      printOperand(topLevelOperation, negFeltOp.getOperand());
    } else if (auto invFeltOp = dyn_cast<InvFeltOp>(operation)) {
      llvm::outs() << "UnOp.inv ";
      printOperand(topLevelOperation, invFeltOp.getOperand());
    } else
    // Function operations
    if (auto callOp = dyn_cast<CallOp>(operation)) {
      auto callee = callOp.getCallee();
      llvm::outs() << escapeName(callee.getRootReference().str());
      for (auto nestedRef : callee.getNestedReferences()) {
        llvm::outs() << "." << escapeName(nestedRef.getRootReference().str());
      }
      for (Value operand : callOp.getArgOperands()) {
        llvm::outs() << " ";
        printOperand(topLevelOperation, operand);
      }
    } else
    // Polymorphic operations
    if (auto constReadOp = dyn_cast<ConstReadOp>(operation)) {
      llvm::outs() << "UnOp.from (Z.of_nat " << constReadOp.getConstName() << ")";
    } else if (auto unifiableCastOp = dyn_cast<UnifiableCastOp>(operation)) {
      llvm::outs() << "UnOp.unifiable_cast ";
      printOperand(topLevelOperation, unifiableCastOp.getInput());
    } else
    // Scf operations
    if (auto ifOp = dyn_cast<IfOp>(operation)) {
      llvm::outs() << "M.if_ ";
      printOperand(topLevelOperation, ifOp.getCondition());
      llvm::outs() << " then\n";
      for (Operation &op : *ifOp.thenBlock()) {
        printOperation(level + 1, topLevelOperation, &op);
      }
      llvm::outs() << " else\n";
      auto elseBlock = ifOp.elseBlock();
      if (elseBlock) {
        for (Operation &op : *elseBlock) {
          printOperation(level + 1, topLevelOperation, &op);
        }
      }
    } else if (auto forOp = dyn_cast<ForOp>(operation)) {
      llvm::outs() << "M.for_ ";
      printOperand(topLevelOperation, forOp.getLowerBound());
      llvm::outs() << " (* to *) ";
      printOperand(topLevelOperation, forOp.getUpperBound());
      llvm::outs() << " (* step *) ";
      printOperand(topLevelOperation, forOp.getStep());
      llvm::outs() << " (fun";
      Region &region = forOp.getRegion();
      // We assume there is only one block
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          llvm::outs() << " (";
          printOperand(topLevelOperation, arg);
          llvm::outs() << " : ";
          printType(false, arg.getType());
          llvm::outs() << ")";
        }
        llvm::outs() << " =>\n";
        for (Operation &op : block) {
          printOperation(level + 1, topLevelOperation, &op);
        }
        llvm::outs() << "\n";
      }
      llvm::outs() << indent(level) << ")";
    } else
    // String operations
    if (auto litStringOp = dyn_cast<LitStringOp>(operation)) {
      llvm::outs() << "String.new " << litStringOp.getValue();
    } else
    // Struct operations
    if (auto fieldReadOp = dyn_cast<FieldReadOp>(operation)) {
      printOperand(topLevelOperation, fieldReadOp.getComponent());
      llvm::outs() << ".(";
      llvm::outs() << fieldReadOp.getComponent().getType().getNameRef().getRootReference().str();
      llvm::outs() << "." << escapeName(fieldReadOp.getFieldName().str()) << ")";
    } else if (auto fieldWriteOp = dyn_cast<FieldWriteOp>(operation)) {
      llvm::outs() << "M.FieldWrite ";
      printOperand(topLevelOperation, fieldWriteOp.getComponent());
      llvm::outs() << ".(";
      llvm::outs() << fieldWriteOp.getComponent().getType().getNameRef().getRootReference().str();
      llvm::outs() << "." << escapeName(fieldWriteOp.getFieldName().str()) << ") ";
      printOperand(topLevelOperation, fieldWriteOp.getVal());
    } else if (auto createStructOp = dyn_cast<CreateStructOp>(operation)) {
      llvm::outs() << "M.CreateStruct";
    } else
    // Unknown operations
    {
      llvm::outs() << "Unknown Operation: " << operation->getName();
      exit(1);
    }
    llvm::outs() << " in\n";
  }

  void printConstParams(StructDefOp* structDefOp, bool withParens) {
    auto constParams = structDefOp->getConstParamsAttr();
    if (!constParams || constParams.empty()) {
      return;
    }
    llvm::outs() << " ";
    if (withParens) {
      llvm::outs() << "{";
    }
    bool isFirst = true;
    for (auto constParam : constParams) {
      if (!isFirst) {
        llvm::outs() << " ";
      }
      isFirst = false;
      printAttr(constParam);
    }
    if (withParens) {
      llvm::outs() << " : nat";
      llvm::outs() << "}";
    }
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
    llvm::outs() << indent(level) << "Definition " << escapeName(func.getName().str()) << " {p} `{Prime p}";
    if (structDefOp) {
      printConstParams(structDefOp, true);
    }
    unsigned argTypeIndex = 0;
    for (auto argType : func.getArgumentTypes()) {
      llvm::outs() << " (arg_fun_" << argTypeIndex << " : ";
      printType(false, argType);
      llvm::outs() << ")";
      argTypeIndex++;
    }
    llvm::outs() << " : M.t ";
    llvm::ArrayRef<Type> results = func.getFunctionType().getResults();
    printTypeTuple(true, results);
    if (func.isExternal()) {
      llvm::outs() << ".\n";
      llvm::outs() << indent(level) << "Admitted.\n";
      return;
    }
    llvm::outs() << " :=\n";

    for (Block &block : func.getBody()) {
      for (Operation &op : block) {
        printOperation(level + 1, topLevelOperation, &op);
      }
    }

    llvm::outs() << ".\n";
  }

  void printStructDefOp(unsigned level, Operation* topLevelOperation, StructDefOp* structDefOp) {
    llvm::outs() << indent(level) << "Module " << structDefOp->getName() << ".\n";
    // Special case when there are no fields
    if (structDefOp->getFieldDefs().size() == 0) {
      llvm::outs() << indent(level + 1) << "Inductive t";
      printConstParams(structDefOp, true);
      llvm::outs() << " : Set := Make.";
    } else {
      llvm::outs() << indent(level + 1) << "Record t";
      printConstParams(structDefOp, true);
      llvm::outs() << " : Set := {\n";
      for (auto fieldDefOp : structDefOp->getFieldDefs()) {
        llvm::outs() << indent(level + 2) << escapeName(fieldDefOp.getSymName().str()) << " : ";
        printType(false, fieldDefOp.getType());
        llvm::outs() << ";\n";
      }
      llvm::outs() << indent(level + 1) << "}.";
    }
    // No implicit arguments for the type itself
    if (hasConstParams(structDefOp)) {
      llvm::outs() << "\n";
      llvm::outs() << indent(level + 1) << "Arguments t : clear implicits.";
    }
    llvm::outs() << "\n\n";
    // Add the `map_mod` operator. We also do it for types with no fields, as the function might be
    // called in types containing it.
    llvm::outs() << indent(level + 1) << "Global Instance IsMapMop {ρ} `{Prime ρ}";
    printConstParams(structDefOp, true);
    llvm::outs() << " : MapMod ";
    if (hasConstParams(structDefOp)) {
      llvm::outs() << "(t";
      printConstParams(structDefOp, false);
      llvm::outs() << ")";
    } else {
      llvm::outs() << "t";
    }
    llvm::outs() << " := {\n";
    if (structDefOp->getFieldDefs().size() == 0) {
      llvm::outs() << indent(level + 2) << "map_mod α := α;\n";
    } else {
      llvm::outs() << indent(level + 2) << "map_mod α := {|\n";
      for (auto fieldDefOp : structDefOp->getFieldDefs()) {
        llvm::outs() << indent(level + 3) << escapeName(fieldDefOp.getSymName().str()) << " := map_mod α.(";
        llvm::outs() << escapeName(fieldDefOp.getSymName().str()) << ");\n";
      }
      llvm::outs() << indent(level + 2) << "|};\n";
    }
    llvm::outs() << indent(level + 1) << "}.";
    llvm::outs() << "\n\n";
    printFunction(level + 1, topLevelOperation, structDefOp, structDefOp->getConstrainFuncOp());
    llvm::outs() << "\n";
    printFunction(level + 1, topLevelOperation, structDefOp, structDefOp->getComputeFuncOp());
    llvm::outs() << indent(level) << "End " << structDefOp->getName() << ".\n";
  }

  void printTopLevelOperations(unsigned level, Operation* topLevelOperation, ModuleOp* moduleOp) {
    // Empty module
    if (moduleOp->getBody()->getOperations().size() == 0) {
      llvm::outs() << "\n";
      llvm::outs() << indent(level) << "(* Empty module *)\n";
      return;
    }
    for (Operation &operation : moduleOp->getBody()->getOperations()) {
      // Function
      if (auto funcDefOp = dyn_cast<FuncDefOp>(operation)) {
        llvm::outs() << "\n";
        printFunction(level, topLevelOperation, nullptr, funcDefOp);
      // Include
      } else if (auto includeOp = dyn_cast<IncludeOp>(operation)) {
        llvm::outs() << "\n";
        llvm::outs() << indent(level) << "(* Require Import ";
        // Substitute the "/" with "."
        std::string path = includeOp.getPath().str();
        std::replace(path.begin(), path.end(), '/', '.');
        // Remove the ".llzk" suffix
        path = path.substr(0, path.size() - 5);
        llvm::outs() << path << " as " << includeOp.getSymName() << ". *)\n";
      // Module
      } else if (auto subModuleOp = dyn_cast<ModuleOp>(operation)) {
        mlir::Location loc = operation.getLoc();
        std::string moduleName = "Anonymous";
        if (auto fileLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
          moduleName = "Line_" +std::to_string(fileLoc.getLine());
        }
        llvm::outs() << "\n";
        llvm::outs() << indent(level) << "Module Module_" << moduleName << ".";
        printTopLevelOperations(level + 1, topLevelOperation, &subModuleOp);
        llvm::outs() << indent(level) << "End Module_" << moduleName << ".\n";
      // Struct
      } else if (auto structDefOp = dyn_cast<StructDefOp>(operation)) {
        llvm::outs() << "\n";
        printStructDefOp(level, topLevelOperation, &structDefOp);
      } else {
        llvm::outs() << "Unknown TopLevel Operation: " << operation.getName() << "\n";
        exit(1);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    llvm::outs() << "(* Generated *)\n";
    llvm::outs() << "Require Import Garden.LLZK.M.\n";

    printTopLevelOperations(0, moduleOp.getOperation(), &moduleOp);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createRocqPass() {
  return std::make_unique<RocqPass>();
};
