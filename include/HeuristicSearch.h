#ifndef MLSCEDULER_HEURISTIC_SEARCH_H_
#define MLSCEDULER_HEURISTIC_SEARCH_H_

#include "SearchMethod.h"
#include "Node.h"
#include "EvaluationByExecution.h"
#include "TilingTransformation.h"
#include "InterchangeTransformation.h"
#include "ParallelizationTransformation.h"
#include "VectorizationTransformation.h"

#include <queue>

using namespace mlir;
class HeuristicSearch : public SearchMethod{
    private:
        mlir::MLIRContext *context;
        std::string functionName;

    public:
        HeuristicSearch(mlir::MLIRContext *context, std::string functionName);
        Node * runSearchMethod(Node * root) override;

};

#endif // MLSCEDULER_HEURISTIC_SEARCH_H_