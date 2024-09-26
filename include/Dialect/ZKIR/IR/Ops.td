#ifndef ZKIR_OPS
#define ZKIR_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "Dialect/ZKIR/IR/Dialect.td"
include "Dialect/ZKIR/IR/Attrs.td"
include "Dialect/ZKIR/IR/Types.td"

class ZKIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<ZKIRDialect, mnemonic, traits>;

// -----
// Basic constructs
// -----

def ZKIR_StructDefOp : ZKIR_Op<"struct", [
  HasParent<"mlir::ModuleOp">,
  Symbol,
  SymbolTable,
  IsolatedFromAbove,
  GraphRegionNoTerminator,
]> {
  let summary = "Circuit struct/component definition";
  let description = [{
      Describes a struct/component in a circuit.

      Example:

      ```zkir
      zkir.struct @ComponentA {
        func.func public @compute(%a: !zkir.felt) {
          return
        }

        func.func public @constrain(%a: !zkir.felt) {
          return
        }
      }
      ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<FlatSymbolRefArrayAttr>:$const_params
  );

  let regions = (region SizedRegion<1> : $body);

  let assemblyFormat = [{
    $sym_name (`<` $const_params^ `>`)? $body attr-dict
  }];

  let hasRegionVerifier = 1;
}

def ZKIR_FieldDefOp : ZKIR_Op<"field", [HasParent<"StructDefOp">, Symbol]> {
  let summary = "Struct/component field";
  let description = [{
    Describes a field in a struct/component.

    Example:

    ```zkir
    zkir.field @f : !zkir.felt
    ```
  }];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<TypedArrayAttrBase<ZKIR_TypeModiferAttr, "type modifiers">>:$modifiers,
      TypeAttrOf<AnyZKIRType>:$type
    );

  let assemblyFormat = [{
    $sym_name `:` ($modifiers^)? $type attr-dict
  }];
}

def ZKIR_RefFieldOp : ZKIR_Op<"fref"> {
  let summary = "Reference a struct/component field.";
  let description = [{
    This operation references a named field from a struct/component.
  }];

  //TODO: I think $component should be "TypedStrAttr<ZKIR_StructType>" but 
  //  that gives "invalid kind of attribute specified" and maybe requires some
  //  other work on the symbol table or type system.
  let arguments = (ins SymbolNameAttr:$component, SymbolNameAttr:$field_name);
  let results = (outs AnyZKIRType:$field);

  let assemblyFormat = [{
    $component `[` $field_name `]` attr-dict `:` type($field)
  }];
}

def ZKIR_FeltConstantOp : ZKIR_Op<"feltconst", [
  ConstantLike, Pure,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "field element constant";
  let description = [{
    The `zkir.feltconst` operation produces a felt-typed SSA value equal to
    some felt-typed integer constant.

    Example:

    ```zkir
    %0 = zkir.feltconst 42
    ```
  }];

  let arguments = (ins ZKIR_FeltConstAttr:$value);
  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def ZKIR_CreateArrayOp : ZKIR_Op<"new_array", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    TypesMatchWith<"operand types match result type",
                   "result", "elements", "::llvm::SmallVector<::mlir::Type, 2>("
                   "::llvm::cast<::zkir::ArrayType>($_self).getNumElements(), "
                   "::llvm::cast<::zkir::ArrayType>($_self).getElementType())">
  ]> {
  let summary = "creates an array";
  let description = [{
    This operation creates a new array with a given elements.

    Example:

    ```zkir
    %0 = zkir.new_array %a, %b, %c : !zkir.array<3 x !zkir.felt>
    ```
  }];

  let arguments = (ins Variadic<AnyZKIRType>:$elements);
  let results = (outs ZKIR_ArrayType:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";
}

// -----
// Field element operators
// -----

class ZKIR_BinaryFeltOp<string mnemonic, list<Trait> traits = []>
    : ZKIR_Op<mnemonic, traits # [SameOperandsAndResultType, Pure]> {

  let arguments = (ins ZKIR_FeltType:$lhs, ZKIR_FeltType:$rhs);
  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}


class ZKIR_UnaryFeltOp<string mnemonic, list<Trait> traits = []>
    : ZKIR_Op<mnemonic, traits # [SameOperandsAndResultType, Pure]> {

  let arguments = (ins ZKIR_FeltType:$operand);
  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "$operand attr-dict";
}

def ZKIR_AddFeltOp : ZKIR_BinaryFeltOp<"add", [Commutative]> {
  let summary = "Addition operator for field elements";
  let description = [{}];
}

def ZKIR_SubFeltOp : ZKIR_BinaryFeltOp<"sub"> {
  let summary = "Subtraction operator for field elements";
  let description = [{}];
}

def ZKIR_MulFeltOp : ZKIR_BinaryFeltOp<"mul", [Commutative]> {
  let summary = "Multiplication operator for field elements";
  let description = [{}];
}

def ZKIR_DivFeltOp : ZKIR_BinaryFeltOp<"div"> {
  let summary = "Division operator for field elements";
  let description = [{}];
}

def ZKIR_ModFeltOp : ZKIR_BinaryFeltOp<"mod"> {
  let summary = "Modulus/remainder operator for field elements";
  let description = [{}];
}

def ZKIR_NegFeltOp : ZKIR_UnaryFeltOp<"neg"> {
  let summary = "Negation operator for field elements";
  let description = [{}];
}

def ZKIR_AndFeltOp : ZKIR_BinaryFeltOp<"and", [Commutative]> {
  let summary = "Bitwise AND operator for field elements";
  let description = [{}];
}

def ZKIR_OrFeltOp : ZKIR_BinaryFeltOp<"or", [Commutative]> {
  let summary = "Bitwise OR operator for field elements";
  let description = [{}];
}

def ZKIR_XorFeltOp : ZKIR_BinaryFeltOp<"xor", [Commutative]> {
  let summary = "Bitwise XOR operator for field elements";
  let description = [{}];
}

def ZKIR_NotFeltOp : ZKIR_UnaryFeltOp<"not"> {
  let summary = "Bit flip operator for field elements";
  let description = [{}];
}

def ZKIR_ShlFeltOp : ZKIR_BinaryFeltOp<"shl"> {
  let summary = "Left shift operator for field elements";
  let description = [{}];
}

def ZKIR_ShrFeltOp : ZKIR_BinaryFeltOp<"shr"> {
  let summary = "Right shift operator for field elements";
  let description = [{}];
}

// Match format of Index comparisons (for now)
def ZKIR_CmpOp : ZKIR_Op<"cmp", [Pure]> {
  let summary = "field element comparison operation";
  let description = [{
    The `zkir.cmp` operation takes two field element values and compares them
    according to the comparison predicate and returns an `i1`. The following
    comparisons are supported:

    -   `eq`:  equal
    -   `ne`:  not equal
    -   `lt`: less than
    -   `le`: less than or equal
    -   `gt`: greater than
    -   `ge`: greater than or equal

    The result is `1` if the comparison is true and `0` otherwise.

    Example:

    ```zkir
    // Less than comparison.
    %0 = zkir.cmp<lt>(%a, %b)

    // Greater than or equal comparison.
    %1 = zkir.cmp<ge>(%a, %b)

    // Not equal comparison.
    %2 = zkir.cmp<ne>(%a, %b)
    ```
  }];

  let arguments = (ins ZKIR_CmpPredicateAttr:$pred, ZKIR_FeltType:$lhs, ZKIR_FeltType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "`` $pred `(` $lhs `,` $rhs `)` attr-dict";
}

#endif // ZKIR_OPS
