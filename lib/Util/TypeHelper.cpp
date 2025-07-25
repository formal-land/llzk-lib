//===-- TypeHelper.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace felt;
using namespace polymorphic;
using namespace string;

void BuildShortTypeString::appendSymName(StringRef str) {
  if (str.empty()) {
    ss << '?';
  } else {
    ss << '@' << str;
  }
}

void BuildShortTypeString::appendSymRef(SymbolRefAttr sa) {
  appendSymName(sa.getRootReference().getValue());
  for (FlatSymbolRefAttr nestedRef : sa.getNestedReferences()) {
    ss << "::";
    appendSymName(nestedRef.getValue());
  }
}

BuildShortTypeString &BuildShortTypeString::append(Type type) {
  size_t position = ret.size();
  // Cases must be consistent with isValidTypeImpl() below.
  if (type.isSignlessInteger(1)) {
    ss << 'b';
  } else if (llvm::isa<IndexType>(type)) {
    ss << 'i';
  } else if (llvm::isa<FeltType>(type)) {
    ss << 'f';
  } else if (llvm::isa<StringType>(type)) {
    ss << 's';
  } else if (llvm::isa<TypeVarType>(type)) {
    ss << "!t<";
    appendSymName(llvm::cast<TypeVarType>(type).getRefName());
    ss << '>';
  } else if (llvm::isa<ArrayType>(type)) {
    ArrayType at = llvm::cast<ArrayType>(type);
    ss << "!a<";
    append(at.getElementType());
    ss << ':';
    append(at.getDimensionSizes());
    ss << '>';
  } else if (llvm::isa<StructType>(type)) {
    StructType st = llvm::cast<StructType>(type);
    ss << "!s<";
    appendSymRef(st.getNameRef());
    if (ArrayAttr params = st.getParams()) {
      ss << '_';
      append(params.getValue());
    }
    ss << '>';
  } else {
    ss << "!INVALID";
  }
  assert(
      ret.find(PLACEHOLDER, position) == std::string::npos &&
      "formatting a Type should not produce the 'PLACEHOLDER' char"
  );
  return *this;
}

BuildShortTypeString &BuildShortTypeString::append(Attribute a) {
  // Special case for inserting the `PLACEHOLDER`
  if (a == nullptr) {
    ss << PLACEHOLDER;
    return *this;
  }

  size_t position = ret.size();
  // Adapted from AsmPrinter::Impl::printAttributeImpl()
  if (llvm::isa<IntegerAttr>(a)) {
    IntegerAttr ia = llvm::cast<IntegerAttr>(a);
    Type ty = ia.getType();
    bool isUnsigned = ty.isUnsignedInteger() || ty.isSignlessInteger(1);
    ia.getValue().print(ss, !isUnsigned);
  } else if (llvm::isa<SymbolRefAttr>(a)) {
    appendSymRef(llvm::cast<SymbolRefAttr>(a));
  } else if (llvm::isa<TypeAttr>(a)) {
    append(llvm::cast<TypeAttr>(a).getValue());
  } else if (llvm::isa<AffineMapAttr>(a)) {
    ss << "!m<";
    // Filter to remove spaces from the affine_map representation
    filtered_raw_ostream fs(ss, [](char c) { return c == ' '; });
    llvm::cast<AffineMapAttr>(a).getValue().print(fs);
    fs.flush();
    ss << '>';
  } else if (llvm::isa<ArrayAttr>(a)) {
    append(llvm::cast<ArrayAttr>(a).getValue());
  } else {
    // All valid/legal cases must be covered above
    assertValidAttrForParamOfType(a);
  }
  assert(
      ret.find(PLACEHOLDER, position) == std::string::npos &&
      "formatting a non-null Attribute should not produce the 'PLACEHOLDER' char"
  );
  return *this;
}

BuildShortTypeString &BuildShortTypeString::append(ArrayRef<Attribute> attrs) {
  llvm::interleave(attrs, ss, [this](Attribute a) { append(a); }, "_");
  return *this;
}

std::string BuildShortTypeString::from(const std::string &base, ArrayRef<Attribute> attrs) {
  BuildShortTypeString bldr;

  bldr.ret.reserve(base.size() + attrs.size()); // reserve minimum space required

  // First handle replacements of PLACEHOLDER
  auto END = attrs.end();
  auto IT = attrs.begin();
  {
    size_t start = 0;
    for (size_t pos; (pos = base.find(PLACEHOLDER, start)) != std::string::npos; start = pos + 1) {
      // Append original up to the PLACEHOLDER
      bldr.ret.append(base, start, pos - start);
      // Append the formatted Attribute
      assert(IT != END && "must have an Attribute for every 'PLACEHOLDER' char");
      bldr.append(*IT++);
    }
    // Append remaining suffix of the original
    bldr.ret.append(base, start, base.size() - start);
  }

  // Append any remaining Attributes
  if (IT != END) {
    bldr.ss << '_';
    bldr.append(ArrayRef(IT, END));
  }

  return bldr.ret;
}

namespace {

template <typename... Types> class TypeList {

  /// Helper class that handles appending the 'Types' names to some kind of stream
  template <typename StreamType> struct Appender {

    // single
    template <typename Ty> static inline void append(StreamType &stream) {
      stream << '\'' << Ty::name << '\'';
    }

    // multiple
    template <typename First, typename Second, typename... Rest>
    static void append(StreamType &stream) {
      append<First>(stream);
      stream << ", ";
      append<Second, Rest...>(stream);
    }

    // full list with wrapping brackets
    static inline void append(StreamType &stream) {
      stream << '[';
      append<Types...>(stream);
      stream << ']';
    }
  };

public:
  // Checks if the provided value is an instance of any of `Types`
  template <typename T> static inline bool matches(const T &value) {
    return llvm::isa_and_present<Types...>(value);
  }

  static void reportInvalid(EmitErrorFn emitError, const Twine &foundName, const char *aspect) {
    InFlightDiagnostic diag = emitError().append(aspect, " must be one of ");
    Appender<InFlightDiagnostic>::append(diag);
    diag.append(" but found '", foundName, '\'').report();
  }

  static inline void reportInvalid(EmitErrorFn emitError, Attribute found, const char *aspect) {
    if (emitError) {
      reportInvalid(emitError, found ? found.getAbstractAttribute().getName() : "nullptr", aspect);
    }
  }

  // Returns a comma-separated list formatted string of the names of `Types`
  static inline std::string getNames() {
    return buildStringViaCallback(Appender<llvm::raw_string_ostream>::append);
  }
};

/// Helpers to compute the union of multiple TypeList without repetition.
/// Use as: TypeListUnion<TypeList<...>, TypeList<...>, ...>
template <class... Ts> struct make_unique {
  using type = TypeList<Ts...>;
};

template <class... Ts> struct make_unique<TypeList<>, Ts...> : make_unique<Ts...> {};

template <class U, class... Us, class... Ts>
struct make_unique<TypeList<U, Us...>, Ts...>
    : std::conditional_t<
          (std::is_same_v<U, Us> || ...) || (std::is_same_v<U, Ts> || ...),
          make_unique<TypeList<Us...>, Ts...>, make_unique<TypeList<Us...>, Ts..., U>> {};

template <class... Ts> using TypeListUnion = typename make_unique<Ts...>::type;

// Dimensions in the ArrayType must be one of the following:
//  - Integer constants
//  - SymbolRef (flat ref for struct params, non-flat for global constants from another module)
//  - AffineMap (for array created within a loop where size depends on loop variable)
using ArrayDimensionTypes = TypeList<IntegerAttr, SymbolRefAttr, AffineMapAttr>;

// Parameters in the StructType must be one of the following:
//  - Integer constants
//  - SymbolRef (flat ref for struct params, non-flat for global constants from another module)
//  - Type
//  - AffineMap (for array of non-homogeneous structs)
using StructParamTypes = TypeList<IntegerAttr, SymbolRefAttr, TypeAttr, AffineMapAttr>;

class AllowedTypes {
  struct ColumnCheckData {
    SymbolTableCollection *symbolTable = nullptr;
    Operation *op = nullptr;
  };

  bool no_felt : 1 = false;
  bool no_string : 1 = false;
  bool no_non_signal_struct : 1 = false;
  bool no_signal_struct : 1 = false;
  bool no_array : 1 = false;
  bool no_var : 1 = false;
  bool no_int : 1 = false;
  bool no_struct_params : 1 = false;
  bool must_be_column : 1 = false;

  ColumnCheckData columnCheck;

  /// Validates that, if columns are a requirement, the struct type has columns.
  /// If columns are not a requirement returns true early since the pointers required
  /// for lookup may be null.
  bool validColumns(StructType s) {
    if (!must_be_column) {
      return true;
    }
    assert(columnCheck.symbolTable);
    assert(columnCheck.op);
    return succeeded(s.hasColumns(*columnCheck.symbolTable, columnCheck.op));
  }

public:
  constexpr AllowedTypes &noFelt() {
    no_felt = true;
    return *this;
  }

  constexpr AllowedTypes &noString() {
    no_string = true;
    return *this;
  }

  constexpr AllowedTypes &noStruct() {
    no_non_signal_struct = true;
    no_signal_struct = true;
    return *this;
  }

  constexpr AllowedTypes &noStructExceptSignal() {
    no_non_signal_struct = true;
    no_signal_struct = false;
    return *this;
  }

  constexpr AllowedTypes &noArray() {
    no_array = true;
    return *this;
  }

  constexpr AllowedTypes &noVar() {
    no_var = true;
    return *this;
  }

  constexpr AllowedTypes &noInt() {
    no_int = true;
    return *this;
  }

  constexpr AllowedTypes &noStructParams(bool noStructParams = true) {
    no_struct_params = noStructParams;
    return *this;
  }

  constexpr AllowedTypes &onlyInt() {
    no_int = false;
    return noFelt().noString().noStruct().noArray().noVar();
  }

  constexpr AllowedTypes &mustBeColumn(SymbolTableCollection &symbolTable, Operation *op) {
    must_be_column = true;
    columnCheck.symbolTable = &symbolTable;
    columnCheck.op = op;
    return *this;
  }

  // This is the main check for allowed types.
  bool isValidTypeImpl(Type type);

  bool areValidArrayDimSizes(ArrayRef<Attribute> dimensionSizes, EmitErrorFn emitError = nullptr) {
    // In LLZK, the number of array dimensions must always be known, i.e., `hasRank()==true`
    if (dimensionSizes.empty()) {
      if (emitError) {
        emitError().append("array must have at least one dimension").report();
      }
      return false;
    }
    // Rather than immediately returning on failure, we check all dimensions and aggregate to
    // provide as many errors are possible in a single verifier run.
    bool success = true;
    for (Attribute a : dimensionSizes) {
      if (!ArrayDimensionTypes::matches(a)) {
        ArrayDimensionTypes::reportInvalid(emitError, a, "Array dimension");
        success = false;
      } else if (no_var && !llvm::isa_and_present<IntegerAttr>(a)) {
        TypeList<IntegerAttr>::reportInvalid(emitError, a, "Concrete array dimension");
        success = false;
      } else if (failed(verifyAffineMapAttrType(emitError, a))) {
        success = false;
      } else if (failed(verifyIntAttrType(emitError, a))) {
        success = false;
      }
    }
    return success;
  }

  bool isValidArrayElemTypeImpl(Type type) {
    // ArrayType element can be any valid type sans ArrayType itself.
    return !llvm::isa<ArrayType>(type) && isValidTypeImpl(type);
  }

  bool isValidArrayTypeImpl(
      Type elementType, ArrayRef<Attribute> dimensionSizes, EmitErrorFn emitError = nullptr
  ) {
    if (!areValidArrayDimSizes(dimensionSizes, emitError)) {
      return false;
    }

    // Ensure array element type is valid
    if (!isValidArrayElemTypeImpl(elementType)) {
      if (emitError) {
        // Print proper message if `elementType` is not a valid LLZK type or
        //  if it's simply not the right kind of type for an array element.
        if (succeeded(checkValidType(emitError, elementType))) {
          emitError()
              .append(
                  '\'', ArrayType::name, "' element type cannot be '",
                  elementType.getAbstractType().getName(), '\''
              )
              .report();
        }
      }
      return false;
    }
    return true;
  }

  bool isValidArrayTypeImpl(Type type) {
    if (ArrayType arrTy = llvm::dyn_cast<ArrayType>(type)) {
      return isValidArrayTypeImpl(arrTy.getElementType(), arrTy.getDimensionSizes());
    }
    return false;
  }

  // Note: The `no*` flags here refer to Types nested within a TypeAttr parameter (if any) except
  // for the `no_struct_params` flag which requires that `params` is null or empty.
  bool areValidStructTypeParams(ArrayAttr params, EmitErrorFn emitError = nullptr) {
    if (isNullOrEmpty(params)) {
      return true;
    }
    if (no_struct_params) {
      return false;
    }
    bool success = true;
    for (Attribute p : params) {
      if (!StructParamTypes::matches(p)) {
        StructParamTypes::reportInvalid(emitError, p, "Struct parameter");
        success = false;
      } else if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(p)) {
        if (!isValidTypeImpl(tyAttr.getValue())) {
          if (emitError) {
            emitError().append("expected a valid LLZK type but found ", tyAttr.getValue()).report();
          }
          success = false;
        }
      } else if (no_var && !llvm::isa<IntegerAttr>(p)) {
        TypeList<IntegerAttr>::reportInvalid(emitError, p, "Concrete struct parameter");
        success = false;
      } else if (failed(verifyAffineMapAttrType(emitError, p))) {
        success = false;
      } else if (failed(verifyIntAttrType(emitError, p))) {
        success = false;
      }
    }

    return success;
  }

  // Note: The `no*` flags here refer to Types nested within a TypeAttr parameter.
  bool isValidStructTypeImpl(Type type, bool allowSignalStruct, bool allowNonSignalStruct) {
    if (!allowSignalStruct && !allowNonSignalStruct) {
      return false;
    }
    if (StructType sType = llvm::dyn_cast<StructType>(type); sType && validColumns(sType)) {
      return (allowSignalStruct && isSignalType(sType)) ||
             (allowNonSignalStruct && areValidStructTypeParams(sType.getParams()));
    }
    return false;
  }
};

bool AllowedTypes::isValidTypeImpl(Type type) {
  assert(
      !(no_int && no_felt && no_string && no_var && no_non_signal_struct && no_signal_struct &&
        no_array) &&
      "All types have been deactivated"
  );
  return (!no_int && type.isSignlessInteger(1)) || (!no_int && llvm::isa<IndexType>(type)) ||
         (!no_felt && llvm::isa<FeltType>(type)) || (!no_string && llvm::isa<StringType>(type)) ||
         (!no_var && llvm::isa<TypeVarType>(type)) || (!no_array && isValidArrayTypeImpl(type)) ||
         isValidStructTypeImpl(type, !no_signal_struct, !no_non_signal_struct);
}

} // namespace

bool isValidType(Type type) { return AllowedTypes().isValidTypeImpl(type); }

bool isValidColumnType(Type type, SymbolTableCollection &symbolTable, Operation *op) {
  return AllowedTypes().noString().noInt().mustBeColumn(symbolTable, op).isValidTypeImpl(type);
}

bool isValidGlobalType(Type type) { return AllowedTypes().noVar().isValidTypeImpl(type); }

bool isValidEmitEqType(Type type) {
  return AllowedTypes().noString().noStructExceptSignal().isValidTypeImpl(type);
}

// Allowed types must align with StructParamTypes (defined below)
bool isValidConstReadType(Type type) {
  return AllowedTypes().noString().noStruct().noArray().isValidTypeImpl(type);
}

bool isValidArrayElemType(Type type) { return AllowedTypes().isValidArrayElemTypeImpl(type); }

bool isValidArrayType(Type type) { return AllowedTypes().isValidArrayTypeImpl(type); }

bool isConcreteType(Type type, bool allowStructParams) {
  return AllowedTypes().noVar().noStructParams(!allowStructParams).isValidTypeImpl(type);
}

bool isSignalType(Type type) {
  if (auto structParamTy = llvm::dyn_cast<StructType>(type)) {
    return isSignalType(structParamTy);
  }
  return false;
}

bool isSignalType(StructType sType) {
  // Only check the leaf part of the reference (i.e., just the struct name itself) to allow cases
  // where the `COMPONENT_NAME_SIGNAL` struct may be placed within some nesting of modules, as
  // happens when it's imported via an IncludeOp.
  return sType.getNameRef().getLeafReference() == COMPONENT_NAME_SIGNAL;
}

bool hasAffineMapAttr(Type type) {
  bool encountered = false;
  type.walk([&](AffineMapAttr a) {
    encountered = true;
    return WalkResult::interrupt();
  });
  return encountered;
}

bool isDynamic(IntegerAttr intAttr) { return ShapedType::isDynamic(fromAPInt(intAttr.getValue())); }

namespace {

/// Optional result from type unifications. Maps `AffineMapAttr` appearing in one type to the
/// associated `IntegerAttr` from the other type at the same nested position. The `Side` enum in the
/// key indicates which input expression the `AffineMapAttr` is from. Additionally, if a conflict is
/// found (i.e., multiple occurrences of a specific `AffineMapAttr` on the same side map to
/// different `IntegerAttr` from the other side), the mapped value will be `nullptr`.
///
/// This map is for tracking replacement of `AffineMapAttr` with integer constant values to
/// determine if a type unification is due to a concrete integer instantiation of `AffineMapAttr`.
using AffineInstantiations = DenseMap<std::pair<AffineMapAttr, Side>, IntegerAttr>;

struct UnifierImpl {
  ArrayRef<StringRef> rhsRevPrefix;
  UnificationMap *unifications;
  AffineInstantiations *affineToIntTracker;
  // This optional function can be used to provide an exception to the standard unification
  // rules and return a true/success result when it otherwise may not.
  llvm::function_ref<bool(Type oldTy, Type newTy)> overrideSuccess;

  UnifierImpl(UnificationMap *unificationMap, ArrayRef<StringRef> rhsReversePrefix = {})
      : rhsRevPrefix(rhsReversePrefix), unifications(unificationMap), affineToIntTracker(nullptr),
        overrideSuccess(nullptr) {}

  bool typeParamsUnify(
      const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams,
      bool unifyDynamicSize = false
  ) {
    auto pred = [this, unifyDynamicSize](auto lhsAttr, auto rhsAttr) {
      return paramAttrUnify(lhsAttr, rhsAttr, unifyDynamicSize);
    };
    return (lhsParams.size() == rhsParams.size()) &&
           std::equal(lhsParams.begin(), lhsParams.end(), rhsParams.begin(), pred);
  }

  UnifierImpl &trackAffineToInt(AffineInstantiations *tracker) {
    this->affineToIntTracker = tracker;
    return *this;
  }

  UnifierImpl &withOverrides(llvm::function_ref<bool(Type oldTy, Type newTy)> overrides) {
    this->overrideSuccess = overrides;
    return *this;
  }

  /// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
  /// are equivalent or could be equivalent after full instantiation of struct parameters.
  bool typeParamsUnify(
      const ArrayAttr &lhsParams, const ArrayAttr &rhsParams, bool unifyDynamicSize = false
  ) {
    if (lhsParams && rhsParams) {
      return typeParamsUnify(lhsParams.getValue(), rhsParams.getValue(), unifyDynamicSize);
    }
    // When one or the other is null, they're only equivalent if both are null
    return !lhsParams && !rhsParams;
  }

  bool arrayTypesUnify(ArrayType lhs, ArrayType rhs) {
    // Check if the element types of the two arrays can unify
    if (!typesUnify(lhs.getElementType(), rhs.getElementType())) {
      return false;
    }
    // Check if the dimension size attributes unify between the LHS and RHS
    return typeParamsUnify(
        lhs.getDimensionSizes(), rhs.getDimensionSizes(), /*unifyDynamicSize=*/true
    );
  }

  bool structTypesUnify(StructType lhs, StructType rhs) {
    // Check if it references the same StructDefOp, considering the additional RHS path prefix.
    SmallVector<StringRef> rhsNames = getNames(rhs.getNameRef());
    rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
    if (rhsNames != getNames(lhs.getNameRef())) {
      return false;
    }
    // Check if the parameters unify between the LHS and RHS
    return typeParamsUnify(lhs.getParams(), rhs.getParams());
  }

  bool typesUnify(Type lhs, Type rhs) {
    if (lhs == rhs) {
      return true;
    }
    if (overrideSuccess && overrideSuccess(lhs, rhs)) {
      return true;
    }
    // A type variable can be any type, thus it unifies with anything.
    if (TypeVarType lhsTvar = llvm::dyn_cast<TypeVarType>(lhs)) {
      track(Side::LHS, lhsTvar.getNameRef(), rhs);
      return true;
    }
    if (TypeVarType rhsTvar = llvm::dyn_cast<TypeVarType>(rhs)) {
      track(Side::RHS, rhsTvar.getNameRef(), lhs);
      return true;
    }
    if (llvm::isa<StructType>(lhs) && llvm::isa<StructType>(rhs)) {
      return structTypesUnify(llvm::cast<StructType>(lhs), llvm::cast<StructType>(rhs));
    }
    if (llvm::isa<ArrayType>(lhs) && llvm::isa<ArrayType>(rhs)) {
      return arrayTypesUnify(llvm::cast<ArrayType>(lhs), llvm::cast<ArrayType>(rhs));
    }
    return false;
  }

private:
  template <typename Tracker, typename Key, typename Val>
  inline void track(Tracker &tracker, Side side, Key keyHead, Val val) {
    auto key = std::make_pair(keyHead, side);
    auto it = tracker.find(key);
    if (it == tracker.end()) {
      tracker.try_emplace(key, val);
    } else if (it->getSecond() != val) {
      it->second = nullptr;
    }
  }

  void track(Side side, SymbolRefAttr symRef, Type ty) {
    if (unifications) {
      Attribute attr;
      if (TypeVarType tvar = dyn_cast<TypeVarType>(ty)) {
        // If 'ty' is TypeVarType<@S>, just map to @S directly.
        attr = tvar.getNameRef();
      } else {
        // Otherwise wrap as a TypeAttr.
        attr = TypeAttr::get(ty);
      }
      assert(symRef);
      assert(attr);
      track(*unifications, side, symRef, attr);
    }
  }

  void track(Side side, SymbolRefAttr symRef, Attribute attr) {
    if (unifications) {
      // If 'attr' is TypeAttr<TypeVarType<@S>>, just map to @S directly.
      if (TypeAttr tyAttr = dyn_cast<TypeAttr>(attr)) {
        if (TypeVarType tvar = dyn_cast<TypeVarType>(tyAttr.getValue())) {
          attr = tvar.getNameRef();
        }
      }
      assert(symRef);
      assert(attr);
      // If 'attr' is a SymbolRefAttr, map in both directions for the correctness of
      // `isMoreConcreteUnification()` which relies on RHS check while other external
      // checks on the UnificationMap may do LHS checks, and in the case of both being
      // SymbolRefAttr, unification in either direction is possible.
      if (SymbolRefAttr otherSymAttr = dyn_cast<SymbolRefAttr>(attr)) {
        track(*unifications, reverse(side), otherSymAttr, symRef);
      }
      track(*unifications, side, symRef, attr);
    }
  }

  void track(Side side, AffineMapAttr affineAttr, IntegerAttr intAttr) {
    if (affineToIntTracker) {
      assert(affineAttr);
      assert(intAttr);
      assert(!isDynamic(intAttr));
      track(*affineToIntTracker, side, affineAttr, intAttr);
    }
  }

  bool paramAttrUnify(Attribute lhsAttr, Attribute rhsAttr, bool unifyDynamicSize = false) {
    assertValidAttrForParamOfType(lhsAttr);
    assertValidAttrForParamOfType(rhsAttr);
    // Straightforward equality check.
    if (lhsAttr == rhsAttr) {
      return true;
    }
    // AffineMapAttr can unify with IntegerAttr (other than kDynamic) because struct parameter
    // instantiation will result in conversion of AffineMapAttr to IntegerAttr.
    if (AffineMapAttr lhsAffine = llvm::dyn_cast<AffineMapAttr>(lhsAttr)) {
      if (IntegerAttr rhsInt = llvm::dyn_cast<IntegerAttr>(rhsAttr)) {
        if (!isDynamic(rhsInt)) {
          track(Side::LHS, lhsAffine, rhsInt);
          return true;
        }
      }
    }
    if (AffineMapAttr rhsAffine = llvm::dyn_cast<AffineMapAttr>(rhsAttr)) {
      if (IntegerAttr lhsInt = llvm::dyn_cast<IntegerAttr>(lhsAttr)) {
        if (!isDynamic(lhsInt)) {
          track(Side::RHS, rhsAffine, lhsInt);
          return true;
        }
      }
    }
    // If either side is a SymbolRefAttr, assume they unify because either flattening or a pass with
    // a more involved value analysis is required to check if they are actually the same value.
    if (SymbolRefAttr lhsSymRef = llvm::dyn_cast<SymbolRefAttr>(lhsAttr)) {
      track(Side::LHS, lhsSymRef, rhsAttr);
      return true;
    }
    if (SymbolRefAttr rhsSymRef = llvm::dyn_cast<SymbolRefAttr>(rhsAttr)) {
      track(Side::RHS, rhsSymRef, lhsAttr);
      return true;
    }
    // If either side is ShapedType::kDynamic then, similarly to Symbols, assume they unify.
    auto dyn_cast_if_dynamic = [](Attribute attr) -> IntegerAttr {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
        if (isDynamic(intAttr)) {
          return intAttr;
        }
      }
      return nullptr;
    };
    auto isa_const = [](Attribute attr) {
      return llvm::isa_and_present<IntegerAttr, SymbolRefAttr, AffineMapAttr>(attr);
    };
    if (auto lhsIntAttr = dyn_cast_if_dynamic(lhsAttr)) {
      if (isa_const(rhsAttr)) {
        return true;
      }
    }
    if (auto rhsIntAttr = dyn_cast_if_dynamic(rhsAttr)) {
      if (isa_const(lhsAttr)) {
        return true;
      }
    }
    // If both are type refs, check for unification of the types.
    if (TypeAttr lhsTy = llvm::dyn_cast<TypeAttr>(lhsAttr)) {
      if (TypeAttr rhsTy = llvm::dyn_cast<TypeAttr>(rhsAttr)) {
        return typesUnify(lhsTy.getValue(), rhsTy.getValue());
      }
    }
    // Otherwise, they do not unify.
    return false;
  }
};

} // namespace

bool typeParamsUnify(
    const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams,
    UnificationMap *unifications
) {
  return UnifierImpl(unifications).typeParamsUnify(lhsParams, rhsParams);
}

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const ArrayAttr &lhsParams, const ArrayAttr &rhsParams, UnificationMap *unifications
) {
  return UnifierImpl(unifications).typeParamsUnify(lhsParams, rhsParams);
}

bool arrayTypesUnify(
    ArrayType lhs, ArrayType rhs, ArrayRef<StringRef> rhsReversePrefix, UnificationMap *unifications
) {
  return UnifierImpl(unifications, rhsReversePrefix).arrayTypesUnify(lhs, rhs);
}

bool structTypesUnify(
    StructType lhs, StructType rhs, ArrayRef<StringRef> rhsReversePrefix,
    UnificationMap *unifications
) {
  return UnifierImpl(unifications, rhsReversePrefix).structTypesUnify(lhs, rhs);
}

bool typesUnify(
    Type lhs, Type rhs, ArrayRef<StringRef> rhsReversePrefix, UnificationMap *unifications
) {
  return UnifierImpl(unifications, rhsReversePrefix).typesUnify(lhs, rhs);
}

bool isMoreConcreteUnification(
    Type oldTy, Type newTy, llvm::function_ref<bool(Type oldTy, Type newTy)> knownOldToNew
) {
  UnificationMap unifications;
  AffineInstantiations affineInstantiations;
  // Run type unification with the addition that affine map can become integer in the new type.
  if (!UnifierImpl(&unifications)
           .trackAffineToInt(&affineInstantiations)
           .withOverrides(knownOldToNew)
           .typesUnify(oldTy, newTy)) {
    return false;
  }

  // If either map contains RHS-keyed mappings then the old type is "more concrete" than the new.
  // In the UnificationMap, a RHS key would indicate that the new type contains a SymbolRef (i.e.
  // the "least concrete" attribute kind) where the old type contained any other attribute. In the
  // AffineInstantiations map, a RHS key would indicate that the new type contains an AffineMapAttr
  // where the old type contains an IntegerAttr.
  auto entryIsRHS = [](const auto &entry) { return entry.first.second == Side::RHS; };
  return !llvm::any_of(unifications, entryIsRHS) && !llvm::any_of(affineInstantiations, entryIsRHS);
}

IntegerAttr forceIntType(IntegerAttr attr) {
  if (AllowedTypes().onlyInt().isValidTypeImpl(attr.getType())) {
    return attr;
  }
  return IntegerAttr::get(IndexType::get(attr.getContext()), attr.getValue());
}

Attribute forceIntAttrType(Attribute attr) {
  if (IntegerAttr intAttr = llvm::dyn_cast_if_present<IntegerAttr>(attr)) {
    return forceIntType(intAttr);
  }
  return attr;
}

SmallVector<Attribute> forceIntAttrTypes(ArrayRef<Attribute> attrList) {
  return llvm::map_to_vector(attrList, forceIntAttrType);
}

LogicalResult verifyIntAttrType(EmitErrorFn emitError, Attribute in) {
  if (IntegerAttr intAttr = llvm::dyn_cast_if_present<IntegerAttr>(in)) {
    Type attrTy = intAttr.getType();
    if (!AllowedTypes().onlyInt().isValidTypeImpl(attrTy)) {
      if (emitError) {
        emitError()
            .append("IntegerAttr must have type 'index' or 'i1' but found '", attrTy, '\'')
            .report();
      }
      return failure();
    }
  }
  return success();
}

LogicalResult verifyAffineMapAttrType(EmitErrorFn emitError, Attribute in) {
  if (AffineMapAttr affineAttr = llvm::dyn_cast_if_present<AffineMapAttr>(in)) {
    AffineMap map = affineAttr.getValue();
    if (map.getNumResults() != 1) {
      if (emitError) {
        emitError()
            .append(
                "AffineMapAttr must yield a single result, but found ", map.getNumResults(),
                " results"
            )
            .report();
      }
      return failure();
    }
  }
  return success();
}

LogicalResult verifyStructTypeParams(EmitErrorFn emitError, ArrayAttr params) {
  return success(AllowedTypes().areValidStructTypeParams(params, emitError));
}

LogicalResult verifyArrayDimSizes(EmitErrorFn emitError, ArrayRef<Attribute> dimensionSizes) {
  return success(AllowedTypes().areValidArrayDimSizes(dimensionSizes, emitError));
}

LogicalResult
verifyArrayType(EmitErrorFn emitError, Type elementType, ArrayRef<Attribute> dimensionSizes) {
  return success(AllowedTypes().isValidArrayTypeImpl(elementType, dimensionSizes, emitError));
}

void assertValidAttrForParamOfType(Attribute attr) {
  // Must be the union of valid attribute types within ArrayType, StructType, and TypeVarType.
  using TypeVarAttrs = TypeList<SymbolRefAttr>; // per ODS spec of TypeVarType
  if (!TypeListUnion<ArrayDimensionTypes, StructParamTypes, TypeVarAttrs>::matches(attr)) {
    llvm::report_fatal_error(
        "Legal type parameters are inconsistent. Encountered " +
        attr.getAbstractAttribute().getName()
    );
  }
}

} // namespace llzk
