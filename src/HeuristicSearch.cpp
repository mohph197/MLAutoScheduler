//===---------------------- HeuristicSearch.cpp - Heuristic Search Implementation ----------------------===//
//
// This file implements the HeuristicSearch class and adapts Node for heuristic-based tree search.
//
//===--------------------------------------------------------------------------------------------------===//

#include "HeuristicSearch.h"

// Constructor for Node
Node::Node(Node* parent, int level, int index)
    : parent(parent), level(level), index(index), visitCount(0), isFullyExpanded(false), bestValue(0.0) {}

// Getters and setters for Node
Node* Node::getParent() const { return parent; }
void Node::setParent(Node* parent) { this->parent = parent; }

int Node::getVisitCount() const { return visitCount; }
void Node::incrementVisitCount() { visitCount++; }

double Node::getBestValue() const { return bestValue; }
void Node::setBestValue(double value) { bestValue = value; }

bool Node::getIsFullyExpanded() const { return isFullyExpanded; }
void Node::setIsFullyExpanded(bool value) { isFullyExpanded = value; }

int Node::getLevel() const { return level; }
void Node::setLevel(int level) { this->level = level; }

int Node::getIndex() const { return index; }
void Node::setIndex(int index) { this->index = index; }

// Get the list of children nodes
std::vector<Node*>& Node::getChildrenNodes() { return children; }

// Add a child node
void Node::addChild(Node* child) {
    children.push_back(child);
}

// Clear all child nodes
void Node::clearChildren() {
    children.clear();
}

// Get the best child node based on the heuristic score
Node* Node::bestChild() {
    Node* bestNode = nullptr;
    double bestValue = std::numeric_limits<double>::infinity();

    for (Node* child : children) {
        double value = child->getBestValue() / std::max(1, child->getVisitCount());
        if (value < bestValue) {
            bestValue = value;
            bestNode = child;
        }
    }
    return bestNode;
}

// Constructor for HeuristicSearch
HeuristicSearch::HeuristicSearch(mlir::MLIRContext *context, std::string functionName, double explorationFactor)
    : context(context), functionName(functionName), explorationFactor(explorationFactor) {}

// Select method - selects the best node to expand using a heuristic (e.g., exploration vs exploitation)
Node* HeuristicSearch::select(Node* node) {
    Node* selectedNode = node;
    while (selectedNode->getIsFullyExpanded() && selectedNode->getChildrenNodes().size() > 0) {
        double bestValue = std::numeric_limits<double>::infinity();
        Node* bestNode = nullptr;

        for (Node* child : selectedNode->getChildrenNodes()) {
            double exploitation = child->getBestValue() / std::max(1, child->getVisitCount());
            double exploration = explorationFactor * std::sqrt(std::log(selectedNode->getVisitCount()) / std::max(1, child->getVisitCount()));
            double heuristicValue = exploitation - exploration;

            if (heuristicValue < bestValue) {
                bestValue = heuristicValue;
                bestNode = child;
            }
        }

        if (bestNode) {
            selectedNode = bestNode;
        }
    }
    return selectedNode;
}

std::vector<Node*> HeuristicSearch::expand(Node* node, int level, int stage,
    std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> LinalgOpStages) {

    std::vector<Node*> children;
    SmallVector<Node*, 2> candidates;
    SmallVector<Node*, 2> sampled_candidates;
    
    std::cerr << "level: " << level << "  stage: " << stage << std::endl;

    if (!node->getIsFullyExpanded() && level <= 3) {
        // Generate candidate transformations based on the level
        switch (level) {
            case 0:
                candidates = Tiling::createTilingCandidates(node, this->context, stage, LinalgOpStages);
                break;
            case 1:
                candidates = Parallelization::createParallelizationCandidates(node, this->context, stage, LinalgOpStages);
                break;
            case 2:
                candidates = Interchange::createInterchangeCandidates(node, this->context);
                break;
            case 3:
                candidates = Vectorization::createVectorizationCandidates(node, this->context);
                break;
        }

        // Limit to 25 candidates if needed
        if (candidates.size() > 25) {
            sampled_candidates.resize(25);
            std::sample(candidates.begin(), candidates.end(), sampled_candidates.begin(), 25, random_engine);
            candidates = sampled_candidates;
        }

        // Create Node children from candidates
        int index = 0;
        for (Node* candidate : candidates) {
            CodeIR* codeir = candidate->getTransformedCodeIr();
            std::vector<Transformation*> transformationList = candidate->getTransformationList();
            Transformation* transformationApplied = candidate->getTransformation();

            Node* child = new Node(node, level + 1, index);
            children.push_back(child);
            node->addChild(child);
            index++;
        }

        node->setIsFullyExpanded(true);
        std::cerr << children.size() << ": size of the children\n";
        return children;
    } else {
        node->setIsFullyExpanded(true);
        return {};
    }
}


// Evaluate method - evaluate the heuristic value of the node (e.g., execution time, memory usage)
double HeuristicSearch::evaluate(Node* node) {
    // Simulate evaluation with a heuristic function (adjust this)
    // For example, using execution time, code size, etc.:
    return std::stod(node->getEvaluation());
}

// Backpropagate method - propagate the result up the tree
void HeuristicSearch::backpropagate(Node* node, double result) {
    while (node != nullptr) {
        node->incrementVisitCount();
        node->setBestValue(node->getBestValue() + result);
        node = node->getParent();
    }
}

// Run search method - main search loop that iterates over a number of iterations
Node* HeuristicSearch::runSearchMethod(Node* root, 
    std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages, 
    int iterations) {

    std::random_device rd;
    std::mt19937 gen(rd());

    Node* selectedNode;
    std::vector<Node*> children;

    for (int i = 0; i < iterations; i++) {
        std::cerr << "Iteration: " << i << std::endl;

        // Selection
        selectedNode = this->select(root);
        std::cerr << "Selection Done\n";

        // Expansion
        int level = selectedNode->getLevel();
        if (level < 4) {
            children = this->expand(selectedNode, level, root->getIndex(), LinalgOpStages);
        }
        std::cerr << "Expansion Done\n";

        // Simulation/Evaluation
        double simulationResult = this->evaluate(selectedNode);
        std::cerr << "Evaluation Done: " << simulationResult << std::endl;

        // Backpropagation
        this->backpropagate(selectedNode, simulationResult);
        std::cerr << "Backpropagation Done\n";
    }

    // Find the best result from the search
    Node* result = root;
    selectedNode = root;
    while (!selectedNode->getChildrenNodes().empty()) {
        selectedNode = selectedNode->bestChild();
        if (std::stod(selectedNode->getEvaluation()) < std::stod(result->getEvaluation())) {
            result = selectedNode;
        }
    }

    result->setParent(nullptr);
    result->clearChildren();

    std::cerr << "Best Evaluation: " << result->getEvaluation() << std::endl;
    return result;
}
