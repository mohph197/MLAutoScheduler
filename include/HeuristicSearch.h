//! Old code, not used anymore
// #ifndef MLSCEDULER_HEURISTIC_SEARCH_H_
// #define MLSCEDULER_HEURISTIC_SEARCH_H_

// #include "SearchMethod.h"
// #include "Node.h"
// #include "EvaluationByExecution.h"
// #include "TilingTransformation.h"
// #include "InterchangeTransformation.h"
// #include "ParallelizationTransformation.h"
// #include "VectorizationTransformation.h"

// #include <queue>

// using namespace mlir;
// class HeuristicSearch : public SearchMethod{
//     private:
//         mlir::MLIRContext *context;
//         std::string functionName;

//     public:
//         HeuristicSearch(mlir::MLIRContext *context, std::string functionName);
//         Node * runSearchMethod(Node * root) override;

// };

// #endif // MLSCEDULER_HEURISTIC_SEARCH_H_


// ! An old version of the HeuristicSearch.h file
// //===---------------------- HeuristicSearch.h - Heuristic Search Header ----------------------===//
// //
// // This file declares the HeuristicSearch class and HeuristicNode class for heuristic-based tree search.
// //
// //===-----------------------------------------------------------------------------------------===//

// #ifndef HEURISTICSEARCH_H
// #define HEURISTICSEARCH_H

// #include <vector>
// #include <unordered_map>
// #include <string>
// #include <random>
// #include <algorithm>
// #include <limits>
// #include <cmath>
// #include <iostream>  // For debugging
// #include <mlir/IR/MLIRContext.h>
// #include <mlir/Dialect/Linalg/IR/Linalg.h>

// #include "Node.h"
// #include "Transformation.h"
// #include "CodeIR.h"
// #include "ParallelizationTransformation.h"
// #include "InterchangeTransformation.h"
// #include "VectorizationTransformation.h"

// // HeuristicNode class representing a node in the search tree
// class HeuristicNode {
// public:
//     HeuristicNode(HeuristicNode* parent, int level, int index);

//     // Getters and Setters
//     HeuristicNode* getParent() const;
//     void setParent(HeuristicNode* parent);

//     int getVisitCount() const;
//     void incrementVisitCount();

//     double getBestValue() const;
//     void setBestValue(double value);

//     bool getIsFullyExpanded() const;
//     void setIsFullyExpanded(bool value);

//     int getLevel() const;
//     void setLevel(int level);

//     int getIndex() const;
//     void setIndex(int index);

//     std::vector<HeuristicNode*>& getChildrenNodes();
//     void addChild(HeuristicNode* child);
//     void clearChildren();

//     // Get the best child node based on a heuristic
//     HeuristicNode* bestChild();

//     // Evaluation function (could be simulation result or any other heuristic value)
//     std::string getEvaluation() const;  // Assuming this returns a string in your implementation

// private:
//     HeuristicNode* parent;
//     std::vector<HeuristicNode*> children;
//     int level;
//     int index;
//     int visitCount;
//     bool isFullyExpanded;
//     double bestValue;
// };

// // HeuristicSearch class that manages the search algorithm
// class HeuristicSearch {
// public:
//     // Constructor
//     HeuristicSearch(mlir::MLIRContext* context, std::string functionName, double explorationFactor = 1.414);

//     // Main search method
//     HeuristicNode* runSearchMethod(HeuristicNode* root, 
//         std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages, 
//         int iterations);

//     // Select the best node for expansion
//     HeuristicNode* select(HeuristicNode* node);

//     // Expand the selected node
//     std::vector<HeuristicNode*> expand(HeuristicNode* node, int level, int stage, 
//         std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages);

//     // Evaluate the heuristic value of the node
//     double evaluate(HeuristicNode* node);

//     // Backpropagate the result through the tree
//     void backpropagate(HeuristicNode* node, double result);

// private:
//     mlir::MLIRContext* context;
//     std::string functionName;
//     double explorationFactor;  // Factor to balance exploration vs exploitation

//     std::random_device random_device;
//     std::mt19937 random_engine{random_device()};
// };

// #endif // HEURISTICSEARCH_H


//===---------------------- HeuristicSearch.h - Heuristic Search Header ----------------------===//
//
// This file defines the HeuristicSearch class and the Node class for the heuristic-based tree search.
//
//===----------------------------------------------------------------------------------------===//

#ifndef HEURISTIC_SEARCH_H
#define HEURISTIC_SEARCH_H

#include <vector>
#include <random>
#include <unordered_map>
#include <string>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h> // For LinalgOps
#include <mlir/Dialect/Linalg/Transforms/Transforms.h> // For transformation-related utilities
#include "CodeIR.h" // Assuming CodeIR is defined elsewhere and provides IR handling
#include "Node.h" // Assuming Node class is defined in a separate file

class HeuristicSearch {
public:
    // Constructor
    HeuristicSearch(mlir::MLIRContext *context, std::string functionName, double explorationFactor);

    // Heuristic search methods
    Node* select(Node* node); // Selection phase
    std::vector<Node*> expand(Node* node, int level, int stage,
        std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages); // Expansion phase
    double evaluate(Node* node); // Evaluation phase
    void backpropagate(Node* node, double result); // Backpropagation phase
    Node* runSearchMethod(Node* root, 
        std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages, 
        int iterations); // Search loop

private:
    mlir::MLIRContext *context; // MLIR context for IR transformations
    std::string functionName; // Name of the function being optimized
    double explorationFactor; // UCB exploration factor

    std::mt19937 random_engine; // Random number engine for sampling

};

#endif // HEURISTIC_SEARCH_H
