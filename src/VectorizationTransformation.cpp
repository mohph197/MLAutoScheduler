//===------------ VectorizationTransformation.cpp VectorizationTransformation -----------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implmentation of the VectorizationTransformation class, which
/// contains the declartion of the Vectorization transformation
///
//===----------------------------------------------------------------------===//
#include "VectorizationTransformation.h"

using namespace mlir;

mlir::Operation *DecomposeConv2dOp(mlir::Operation *Target)
{

  // TODO: TYPE OF CONV
  std::string transformDialectString = "module attributes {transform.with_named_sequence} { \n transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})  { \n   %conv = transform.structured.match ops{[\"linalg.conv_2d_nhwc_hwcf\"]} in %variant_op : (!transform.any_op) -> !transform.any_op %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op %pool = transform.structured.match ops{[\"linalg.pooling_nchw_max\"]} in %variant_op : (!transform.any_op) -> !transform.any_op %decomposed_pool = transform.structured.decompose %pool: (!transform.any_op) -> !transform.any_op transform.yield}}";
  mlir::transform::TransformOptions options1;
  mlir::OwningOpRef<mlir::ModuleOp> moduleFromFile = parseSourceString<mlir::ModuleOp>(transformDialectString, Target->getContext());
  llvm::StringRef entryPoint = "__transform_main";
  mlir::Operation *transformEntryPoint = transform::detail::findTransformEntryPoint(Target, *moduleFromFile, entryPoint);

  transform::applyTransformNamedSequence(
      Target, transformEntryPoint, *moduleFromFile,
      options1.enableExpensiveChecks(false));
  return Target;
  /*mlir::PassManager pm((Target)->getName());

  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(createTransformDialectInterpreterPass(transformDialectString));
  if (!mlir::failed(pm.run((Target))))
  {
    return Target;
  }*/
}

Vectorization::Vectorization(mlir::linalg::LinalgOp *op,
                             mlir::MLIRContext *context)
{
  this->op = op;
  this->context = context;
}

std::string Vectorization::getType()
{
  return "Vectorization";
}

std::string Vectorization::printTransformation()
{

  std::string result = "V( ";
  result += " )";

  return result;
}
void Vectorization::applyTransformation(CodeIR CodeIr)
{
}

Node* Vectorization::createVectorizationNode(
  Node *node,
  int operationStage,
  mlir::MLIRContext *context
) {
  IRRewriter rewriter(context);

  MLIRCodeIR *CodeIr = (MLIRCodeIR *)node->getTransformedCodeIr();
  MLIRCodeIR *ClonedCode = (MLIRCodeIR *)CodeIr->cloneIr();
  Operation *ClonedTarget = (Operation *)ClonedCode->getIr();
  Node *VectNode = new Node(ClonedCode, node->getCurrentStage());

  linalgOps = getLinalgOps(ClonedTarget);
  linalg::LinalgOp linalgOp = linalgOps["operation" + std::to_string(operationStage)].first

  std::vector<Transformation *> TransList = node->getTransformationList();
  VectNode->setTransformationList(TransList);

  Vectorization *vectorization = new Vectorization(&linalgOp, context);
  VectNode->setTransformation(vectorization);
  VectNode->addTransformation(vectorization);

  bool ToDecompose = false;

  if (mlir::TilingInterface ClonedTileableOp = dyn_cast<mlir::TilingInterface>(linalgOp))
  {
    //if ((op->getName().getStringRef()).str() == "linalg.pooling_nchw_max" || (op->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw")
    if ((linalgOp->getName().getStringRef()).str() == "linalg.pooling_nchw_max"
        || (linalgOp->getName().getStringRef()).str() == "linalg.pooling_nchw_sum"
        || (linalgOp->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw")
    {
      llvm::SmallVector<int64_t, 4> tilingSizes;
      OpBuilder builder(context);
      SmallVector<Range> iterationDomain = ClonedTileableOp.getIterationDomain(builder);
      for (size_t i = 0; i < iterationDomain.size(); ++i)
      {
        if (i == 2)
        {
          tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the conv2D
        }
        else if (((linalgOp->getName().getStringRef()).str() == "linalg.pooling_nchw_max"
        || (linalgOp->getName().getStringRef()).str() == "linalg.pooling_nchw_sum")
        && i == 4)
        {
          tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the pooling
          break;
        }
        else if ((linalgOp->getName().getStringRef()).str() == "linalg.conv_2d_nchw_fchw"
        && i == 5)
        {
          tilingSizes.push_back(1); // DEPENDS on the 'h' and the type of the conv2D
          break;
        }
        else
        {
          tilingSizes.push_back(0);
        }
      }
      scf::SCFTilingOptions options;

      SmallVector<OpFoldResult> mixedSizes = getMixedSizes(tilingSizes, context);
      options.setTileSizes(mixedSizes);
      std::cerr << "Modified tilingSizes " << (op->getName().getStringRef()).str() << " : [";
      for (size_t i = 0; i < tilingSizes.size(); ++i)
      {
        std::cerr << tilingSizes[i];
        if (i < tilingSizes.size() - 1)
        {
          std::cerr << ", ";
        }
      }
      std::cerr << "]\n";

      std::cerr << "TRYING TO TILE CONV2D\n";

      ToDecompose = true;

      Tiling *tiling =
          new Tiling(&ClonedTileableOp,
                    VectNode->getCurrentStage(),
                    options,
                    tilingSizes,
                    context);

      node->setTransformation(tiling);

      node->addTransformation(tiling);

      FailureOr<scf::SCFTilingResult> maybeTiled =
          scf::tileUsingSCF(rewriter, ClonedTileableOp, options);
      std::cerr << "END OF TILE CONV2D" << std::endl;

      if (!failed(maybeTiled))
        rewriter.replaceOp(ClonedTileableOp, maybeTiled->loops.front()->getResults());
    }
  }

  if (ToDecompose)
  {
    std::cout << "START DECOMPOSE\n";
    mlir::Operation *DecomposedTarget = DecomposeConv2dOp(ClonedTarget);
    MLIRCodeIR *DecomposedCodeIr = (MLIRCodeIR *)CodeIr->setMLIRIR(DecomposedTarget);
    node->setTransformedCodeIr(DecomposedCodeIr);
    std::cout << "END DECOMPOSE\n";
    DecomposedTarget->dump();
  }

  llvm::ArrayRef<int64_t> emptyArrayRef;
  llvm::ArrayRef<bool> boolArrayRef;
  mlir::LogicalResult vectorized = linalg::vectorize(rewriter, linalgOp, emptyArrayRef,
                                                            boolArrayRef, false);

  std::cerr << "VECTORIZATION SUCCEEDED: " << vectorized.succeeded() << std::endl;

  RewritePatternSet patterns(context);

  // Add vectorization canonicalization and lowering patterns to the set
  // mlir::transform::detail::VectorizeOpGenericAdaptorBase::Properties props;

  // if (!props.getDisableTransferPermutationMapLoweringPatterns())
  mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  // if (!props.getDisableMultiReductionToContractPatterns())
  vector::populateVectorReductionToContractPatterns(patterns);

  vector::populateSinkVectorBroadcastPatterns(patterns);

  // Add additional vectorization patterns for specific operations
  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                linalg::LinalgCopyVTWForwardingPattern>(context, 2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
  tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

  patterns.add<linalg::CopyVectorizationPattern>(context);

  if (failed(applyPatternsAndFoldGreedily(ClonedTarget, std::move(patterns)))) {
    std::cerr << "VECT PATTERNS FAILED" << std::endl;
  }

  return VectNode;
}

/*
std::string removeVectExtraModuleTagCreated(std::string input) // TODO: Figure out why Transform Dialect Interpreter introduces an extra module
{
  std::string s = "module {";
  std::string s1 = "}";
  std::string::size_type i = input.find(s);
  if (i != std::string::npos)
    input.erase(i, s.length());
  std::string::size_type i1 = input.rfind(s1);
  if (i1 != std::string::npos)
    input.erase(i1, s1.length());
  return input;
}
pid_t popenvect(const char *command, int *infp, int *outfp)
{
  int p_stdin[2], p_stdout[2];
  pid_t pid;

  if (pipe(p_stdin) != 0 || pipe(p_stdout) != 0)
    return -1;

  pid = fork();

  if (pid < 0)
  {
    close(p_stdin[READ]);
    close(p_stdin[WRITE]);
    close(p_stdout[READ]);
    close(p_stdout[WRITE]);

    return pid;
  }
  else if (pid == 0)
  {
    close(p_stdin[WRITE]);
    dup2(p_stdin[READ], READ);
    close(p_stdout[READ]);
    dup2(p_stdout[WRITE], WRITE);
    dup2(p_stdout[WRITE], STDERR_FILENO);
    if (std::getenv("LLVM_PATH") != nullptr)
    {
      std::string llvm_path = std::getenv("LLVM_PATH");
      std::string opt = llvm_path + "/build/bin/mlir-opt";
      execl(opt.c_str(),
            "mlir-opt", "--test-transform-dialect-interpreter", "--test-transform-dialect-erase-schedule",
            NULL);
    }

    perror("execl");
    exit(1);
  }

  // Parent process
  close(p_stdin[READ]);
  close(p_stdout[WRITE]);
  if (infp == NULL)
    close(p_stdin[WRITE]);
  else
    *infp = p_stdin[WRITE];

  if (outfp == NULL)
    close(p_stdout[READ]);
  else
    *outfp = p_stdout[READ];

  return pid;
}
std::string getVectorizedCode(std::string inputCode, std::string transfromDialectString)
{
  int in_fd, out_fd;
  pid_t pid;

  // std::string str2 = str1 + "transform.sequence failures(propagate) {^bb0(%arg1: !transform.any_op): \n   %func = transform.structured.match ops{[\"func.func\"]} in %arg1: (!transform.any_op) -> !transform.any_op \n  %func_0 = transform.structured.vectorize %func {vectorize_padding} : (!transform.any_op) -> (!transform.any_op) \n %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 : (!transform.any_op) -> (!transform.any_op) \n  transform.structured.hoist_redundant_tensor_subsets %func_01 : (!transform.any_op) -> ()}";
  std::string str = inputCode + transfromDialectString;

  //  Call popen2 to execute the command and get the input and output file descriptors
  pid = popenvect("", &in_fd, &out_fd);

  if (pid < 0)
  {
    perror("Failed to execute command");
    exit(EXIT_FAILURE);
  }
  // Measure the start time
  write(in_fd, str.c_str(), str.size());

  close(in_fd);
  // Read the output of the executed command
  const int max_output_size = INT_MAX;
  std::vector<char> output_data(max_output_size); // Using a dynamic buffer

  ssize_t total_bytes_read = 0;

  while (true)
  {
    ssize_t bytes_read = read(out_fd, output_data.data() + total_bytes_read, output_data.size() - total_bytes_read);

    if (bytes_read > 0)
    {
      total_bytes_read += bytes_read;

      // Check if the buffer is full (you can adjust this condition based on your needs)
      if (total_bytes_read == max_output_size)
      {
        break; // Exit the loop to avoid buffer overflow
      }
    }
    else if (bytes_read == 0)
    {
      // No more data available to read
      break;
    }
    else
    {
      // Error occurred while reading
      perror("Error while reading output");
      break;
    }
  }

  close(out_fd); // Close the output file descriptor

  // Wait for the child process to finish
  int status;
  waitpid(pid, &status, 0);

  // Check if the child process exited normally
  if (WIFEXITED(status))
  {
    int exit_status = WEXITSTATUS(status);
    printf("OPT Child process exited with status: %d\n", exit_status);
    return output_data.data();
  }
  else
  {
    printf("OPT process did not exit normally.\n");
    return "process did not exit normally";
  }
}*/