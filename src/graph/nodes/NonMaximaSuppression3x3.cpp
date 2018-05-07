#include "arm_compute/graph/nodes/FlattenLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::graph;

std::unique_ptr<arm_compute::IFunction> FlattenLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);

    _target_hint              = ctx.hints().target_hint();
    arm_compute::ITensor *in  = input->tensor();
    arm_compute::ITensor *out = output->tensor();

    // Auto configure output
    TensorShape tensor_shape = in->info()->tensor_shape();
    tensor_shape.collapse(in->info()->num_dimensions());
    arm_compute::auto_init_if_empty(*out->info(), tensor_shape, 1, in->info()->data_type(), in->info()->fixed_point_position());

    // Create node context
    NodeContext node_ctx(OperationType::FlattenLayer);
    node_ctx.set_target(_target_hint);
    node_ctx.add_input(in);
    node_ctx.add_output(out);

    // Get function
    return OperationRegistry::get().find_operation(OperationType::FlattenLayer, _target_hint)->configure(node_ctx);
}


