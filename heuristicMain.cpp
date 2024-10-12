#include <iostream>
#include <unordered_map>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h> // For LinalgOps
#include <mlir/Dialect/Linalg/Transforms/Transforms.h> // For transformation-related utilities

#include "HeuristicSearch.h"
#include "CodeIR.h"
#include "Node.h" // Assuming Node is used internally in HeuristicNode

//! An old version
// int main(int argc, char** argv) {
//     // Check if input arguments are provided
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <input file>" << std::endl;
//         return 1;
//     }

//     // Step 1: Initialize MLIR context and the CodeIR object
//     mlir::MLIRContext context;
//     CodeIR* initialCodeIR = new CodeIR(); // Assuming CodeIR handles your IR and MLIR parsing

//     // Step 2: Parse the input file and create the MLIR module
//     std::string inputFilename = argv[1];
//     mlir::OwningOpRef<mlir::ModuleOp> module = initialCodeIR->parseInputFile(inputFilename, context);

//     if (!module) {
//         std::cerr << "Failed to load the MLIR module from the input file." << std::endl;
//         return 1;
//     }

//     // Step 3: Create the LinalgOpStages map (mapping operations to their transformation stages)
//     std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, std::string>> LinalgOpStages;
    
//     // Assuming you have a method to extract LinalgOps from the module and classify them into stages
//     LinalgOpStages = initialCodeIR->getLinalgOpsWithStages(module.get());

//     if (LinalgOpStages.empty()) {
//         std::cerr << "No LinalgOps found in the module." << std::endl;
//         return 1;
//     }

//     // Step 4: Create the root node at level 0, stage 0
//     int initialStage = LinalgOpStages.size() - 1; // Assuming stages based on the number of LinalgOps
//     HeuristicNode* rootNode = new HeuristicNode(nullptr, 0, 0); // Root node with depth 0 and initial evaluation

//     // Step 5: Create the HeuristicSearch object
//     HeuristicSearch heuristicSearch;

//     // Step 6: Expand the root node at stage 0
//     std::vector<HeuristicNode*> expandedNodes = heuristicSearch.expand(rootNode, 0, initialStage, LinalgOpStages);

//     // Print out initial results
//     std::cout << "Expanded nodes at root level: " << expandedNodes.size() << std::endl;
    
//     // Further expansion loop (for demonstration purposes, you may expand multiple levels)
//     for (auto* node : expandedNodes) {
//         std::vector<HeuristicNode*> furtherExpansion = heuristicSearch.expand(node, 1, initialStage, LinalgOpStages);
//         std::cout << "Further expansion at level 1 produced " << furtherExpansion.size() << " nodes." << std::endl;

//         // You can continue expanding as needed by calling heuristicSearch.expand() further down the tree
//     }

//     // Optionally: You may want to evaluate the expanded nodes based on heuristic criteria
//     for (auto* node : expandedNodes) {
//         std::string evaluation = heuristicSearch.evaluateNode(node, initialCodeIR, LinalgOpStages);
//         std::cout << "Node evaluation: " << evaluation << std::endl;
//     }

//     // Step 7: Clean up dynamically allocated memory (depending on your design)
//     delete rootNode;
//     delete initialCodeIR;
//     // Ensure that other nodes are also cleaned up if dynamically allocated

//     return 0;
// }

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
    HeuristicNode* rootNode = new HeuristicNode(nullptr, 0, 0);

    // Step 5: Create the HeuristicSearch object
    HeuristicSearch heuristicSearch;

    // Step 6: Run the heuristic search for a set number of iterations
    int iterations = 10;  // Adjust the number of iterations as needed
    HeuristicNode* bestNode = heuristicSearch.runSearchMethod(rootNode, LinalgOpStages, iterations);

    // Step 7: Log the best result after the search
    if (bestNode) {
        std::cout << "Best Node Evaluation: " << bestNode->getEvaluation() << std::endl;
    } else {
        std::cout << "No best node found." << std::endl;
    }

    // Clean up dynamically allocated objects
    delete rootNode;
    delete initialCodeIR;
    // Optionally, free other nodes if necessary, depending on how memory is managed in runSearchMethod

    return 0;
}
