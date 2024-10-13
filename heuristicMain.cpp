#include <iostream>
#include <unordered_map>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h> // For LinalgOps
#include <mlir/Dialect/Linalg/Transforms/Transforms.h> // For transformation-related utilities

#include "HeuristicSearch.h"
#include "CodeIR.h"
#include "Node.h" // Assuming Node is used internally

int main(int argc, char** argv) {
    // Check if input arguments are provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input file>" << std::endl;
        return 1;
    }

    // Step 1: Initialize MLIR context and the CodeIR object
    mlir::MLIRContext context;
    CodeIR* initialCodeIR = new CodeIR();

    // Step 2: Parse the input file and create the MLIR module
    std::string inputFilename = argv[1];
    mlir::OwningOpRef<mlir::ModuleOp> module = initialCodeIR->parseInputFile(inputFilename, context);

    if (!module) {
        std::cerr << "Failed to load the MLIR module from the input file." << std::endl;
        delete initialCodeIR;
        return 1;
    }

    // Step 3: Create the LinalgOpStages map (mapping operations to their transformation stages)
    std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages;
    LinalgOpStages = initialCodeIR->getLinalgOpsWithStages(module.get());

    if (LinalgOpStages.empty()) {
        std::cerr << "No LinalgOps found in the module." << std::endl;
        delete initialCodeIR;
        return 1;
    }

    // Step 4: Create the root node at level 0, stage 0
    int initialStage = LinalgOpStages.size() - 1;
    Node* rootNode = new Node(nullptr, 0, 0);

    // Step 5: Create the HeuristicSearch object
    HeuristicSearch heuristicSearch;

    // Step 6: Run the heuristic search for a set number of iterations
    int iterations = 10;  // Adjust the number of iterations as needed
    Node* bestNode = heuristicSearch.runSearchMethod(rootNode, LinalgOpStages, iterations);

    // Step 7: Log the best result after the search
    if (bestNode) {
        std::cout << "Best Node Evaluation: " << bestNode->getEvaluation() << std::endl;
    } else {
        std::cout << "No best node found." << std::endl;
    }

    // Clean up dynamically allocated objects
    delete rootNode;
    delete initialCodeIR;

    return 0;
}
