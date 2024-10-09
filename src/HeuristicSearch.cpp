#include "HeuristicSearch.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <functional>

// Custom comparator for priority queue (min-heap) based on evaluation score
struct CompareEvaluation {
    bool operator()(Node* a, Node* b) {
        // We assume lower evaluation score is better (i.e., faster execution time)
        return std::stod(a->getEvaluation()) > std::stod(b->getEvaluation());
    }
};

HeuristicSearch::HeuristicSearch(mlir::MLIRContext *context, std::string functionName)
{
    this->context = context;
    this->functionName = functionName;
}

//! Run the heuristic-based search method
Node* HeuristicSearch::runSearchMethod(Node* root)
{
    // Initialize the priority queue (min-heap)
    std::priority_queue<Node*, std::vector<Node*>, CompareEvaluation> exploration_queue;

    // Clone the root's MLIR code for evaluation
    MLIRCodeIR* CodeIr = (MLIRCodeIR*)root->getTransformedCodeIr();
    MLIRCodeIR* ClonedCode = (MLIRCodeIR*)CodeIr->cloneIr();
    Node* clone = new Node(ClonedCode, root->getCurrentStage());
    Node* BestNode = clone;

    // Create an evaluator for transformation evaluations
    EvaluationByExecution evaluator = EvaluationByExecution(this->functionName + "_logs_heuristic_search_now.txt");

    // Insert the root node with its initial evaluation
    root->setEvaluation(evaluator.evaluateTransformation(root));
    exploration_queue.push(root);

    int level = 0;
    int maxLevels = 3;  // You can make this configurable or dynamic

    while (!exploration_queue.empty() && level < maxLevels)
    {
        std::cout << "########### Heuristic Search - Level " << level << " ###########\n";

        // Get the current best node (lowest evaluation score) from the priority queue
        Node* currentNode = exploration_queue.top();
        exploration_queue.pop();

        // Create candidates based on the current transformation
        SmallVector<Node*, 2> candidates;

        switch (level) {
            case 0:
                // Apply parallelization transformation
                // candidates = Parallelization::createParallelizationCandidates(currentNode, this->context);
                break;
            case 1:
                // Apply tiling transformation
                // candidates = Tiling::createTilingCandidates(currentNode, this->context);
                break;
            case 2:
                // Apply vectorization transformation
                candidates = Vectorization::createVectorizationCandidates(currentNode, this->context);
                break;
        }

        // Evaluate each transformation candidate
        for (auto child : candidates) {
            std::string eval = evaluator.evaluateTransformation(child);
            child->setEvaluation(eval);
        }

        // Sort the candidates by evaluation value (lowest is better)
        std::sort(candidates.begin(), candidates.end(), [](Node* a, Node* b) {
            return std::stod(a->getEvaluation()) < std::stod(b->getEvaluation());
        });

        // Set children for tree representation and track the best node if at level 0
        currentNode->setChildrenNodes(candidates);
        if (level == 0) BestNode = currentNode;

        // Add the best candidate nodes (sorted by evaluation) to the priority queue
        for (Node* child : candidates) {
            exploration_queue.push(child);
        }

        // Increment the level
        level++;
    }

    return BestNode;  // Return the best node found
}
