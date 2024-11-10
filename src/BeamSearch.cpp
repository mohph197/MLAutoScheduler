//===------------------------- BeamSearch.cpp - BeamSearch  ----------------===//
//
///===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the BeamSearch class, which contains
///  the implmentation of the beam search method
///
//===----------------------------------------------------------------------===//

#include "BeamSearch.h"

BeamSearch::BeamSearch(int beamSize, mlir::MLIRContext *context, std::string functionName)
{
    this->beamSize = beamSize;
    this->context = context;
    this->functionName = functionName;
}

Node *BeamSearch::runSearchMethod(Node *root)
{
    // Clone the root's MLIR code for evaluation
    MLIRCodeIR *CodeIr = (MLIRCodeIR *)root->getTransformedCodeIr();
    Operation *Target = (Operation *)CodeIr->getIr();

    std::unordered_map<std::string, std::pair<mlir::linalg::LinalgOp, LinalgMappingClassification>> linalgOps = getLinalgOps(Target);
    std::cerr << "##### Linalg Ops Size = " << linalgOps.size() << " #####\n";

    // Create an evaluator for transformation evaluations
    EvaluationByExecution evaluator = EvaluationByExecution(this->functionName, "_beam_search_gen.txt");
    double rootEval = evaluator.evaluateTransformation(root);
    root->setEvaluation(rootEval);
    Node *BestNode = root;
    int currentOp = linalgOps.size() - 1;
    BestNode->setCurrentStage(currentOp);

    while (currentOp >= 0)
    {
        std::cerr << "################# Current Op = " << currentOp << " ###############\n";

        // Initialize the exploration queue and level counter
        std::queue<Node *> exploration_queue;
        exploration_queue.push(BestNode);
        int level = 0;

        while (!exploration_queue.empty() && level < 3)
        {
            std::cerr << "################# Level = " << level << " ###############\n";

            // Create a list to store schedule nodes at the current level
            SmallVector<Node *, 2> level_schedules;

            // Iterate through nodes in the exploration queue at the current level
            while (!exploration_queue.empty())
            {
                Node *node = exploration_queue.front();
                exploration_queue.pop();

                mlir::Operation *currentTarget = (mlir::Operation *)((MLIRCodeIR *)node->getTransformedCodeIr())->getIr();
                linalgOps = getLinalgOps(currentTarget);

                // Generate transformation candidates based on the current level.
                SmallVector<Node *, 2> candidates;
                switch (level)
                {
                case 0:
                    // candidates = Parallelization::createParallelizationCandidates(node, this->context, currentOp, linalgOps);
                    // candidates = Tiling::createTilingCandidates(node, this->context, currentOp, linalgOps);
                    SmallVector<Node *, 2> PCandidates = Parallelization::createParallelizationCandidates(node, this->context, currentOp, linalgOps);
                    SmallVector<Node *, 2> TCandidates = Tiling::createTilingCandidates(node, this->context, currentOp, linalgOps);
                    candidates.insert(candidates.end(), PCandidates.begin(), PCandidates.end());
                    candidates.insert(candidates.end(), TCandidates.begin(), TCandidates.end());
                    break;
                case 1:
                    candidates = Interchange::createInterchangeCandidates(node, this->context, currentOp, linalgOps);
                    break;
                case 2:
                    Node *vectNode = Vectorization::createVectorizationNode(node, currentOp, this->context);
                    candidates.push_back(vectNode);
                    break;
                }

                // Evaluate each transformation candidate and store their evaluation results
                for (auto ChildNode : candidates)
                {
                    ChildNode->setCurrentStage(currentOp);
                    double childEval = evaluator.evaluateTransformation(ChildNode);
                    ChildNode->setEvaluation(childEval);
                }

                // Insert the parent node as a candidate
                MLIRCodeIR *ToCloneCodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
                MLIRCodeIR *ClonedCode = (MLIRCodeIR *)ToCloneCodeIr->cloneIr();
                Node *ClonedNode = new Node(ClonedCode, node->getCurrentStage());
                ClonedNode->setTransformationList(node->getTransformationList());
                ClonedNode->setEvaluation(node->getEvaluation());

                candidates.insert(candidates.begin(), ClonedNode);

                // parent_nodes.insert(parent_nodes.begin(),ClonedNode );

                // Sort the candidates based on their evaluation scores
                std::sort(candidates.begin(), candidates.end(), [](Node *a, Node *b)
                        { return a->getEvaluation() < b->getEvaluation(); });

                // Set the children nodes of the current node (for printing the tree)
                node->setChildrenNodes(candidates);

                level_schedules.insert(level_schedules.end(), candidates.begin(), candidates.end());
            }

            // Sort the level's schedule nodes from smallest to largest evaluation
            std::sort(level_schedules.begin(), level_schedules.end(), [](Node *a, Node *b) {
                return a->getEvaluation() < b->getEvaluation();
            });

            /* // Forcing beam search to take one of the parent nodes in the next level
            std::sort(parent_nodes.begin(), parent_nodes.end(), [](Node *a, Node *b) {
                return std::stod(a->getEvaluation()) < std::stod(b->getEvaluation());
            });
            parent_nodes.resize(std::min(1, (int)parent_nodes.size()));
            level_schedules.insert(level_schedules.begin(), parent_nodes.begin(), parent_nodes.end());*/

            // Add the top 'beam_size' children to the exploration queue for the next level
            for (int i; i < std::min(this->beamSize, (int)level_schedules.size()); i++)
            {
                exploration_queue.push(level_schedules[i]);
            }

            level++;
        }

        Node *opBestNode = exploration_queue.front();

        // If the best node in the current op has a better evaluation than the current best node
        if (opBestNode->getEvaluation() < BestNode->getEvaluation())
        {
            BestNode = opBestNode;
        }

        if (currentOp >= 0)
            BestNode->setCurrentStage(--currentOp);
    }


    return BestNode;
}
