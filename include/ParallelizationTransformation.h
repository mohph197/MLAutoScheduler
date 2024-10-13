
//===----------------------- ParallelizationTransformation.h ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the ParallelizationTransformation class, which
/// contains the declartion of the Parallelization transformation
///
//===----------------------------------------------------------------------===//

#ifndef MLSCEDULER_PARALLELIZATION_TRANSFORMATION_H_
#define MLSCEDULER_PARALLELIZATION_TRANSFORMATION_H_

#include "Transformation.h"
#include "MLIRCodeIR.h"
#include "Node.h"
#include "Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"

#include <iostream>
#include <random>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

class Parallelization : public Transformation
{
private:
    mlir::TilingInterface *op;
    int OperationStage;
    mlir::MLIRContext *context;
    llvm::SmallVector<int64_t, 4> tileSizes;

public:
    Parallelization();

    /// Constructor for Tiling that allows specifying the tile size.
    Parallelization(mlir::TilingInterface *op, int OperationStage, llvm::SmallVector<int64_t, 4> tileSizes, mlir::MLIRContext *context);

    /// Applies the tiling transformation to the given CodeIR object.
    /// Overrides the applyTransformation() method from the base class Transformation.
    void applyTransformation(CodeIR CodeIr) override;
    std::string printTransformation() override;
    std::string getType() override;
    /// Creates a list of tiling transformation candidates for the given CodeIR object.
    /// Overrides the createCandidates() method from the base class Transformation.
    static SmallVector<Node *, 2> createParallelizationCandidates(Node *node, mlir::MLIRContext *context,
                                                                  int CurrentStage,
                                                                  std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> LinalgOpStages);

    llvm::SmallVector<int64_t, 4> getTileSizes();
    int getOperationStage();
    void setOperationStage(int stage);
};

#endif // MLSCEDULER_PARALLELIZATION_TRANSFORMATION_H_