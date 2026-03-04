#!/usr/bin/env python3
"""Diagnostic: verify that scaling-mode backward propagates through ALL transformer layers.

This test hooks into each TransformerLayer and checks whether backward gradients
reach every layer. It directly tests the autograd graph integrity.

Usage:
    cd /research/d1/gds/ytyang/yichengfeng/fork_megatron/Megatron-LM
    python tests/unit/test_backward_layer_count.py
"""

import os
import sys
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def trace_grad_fn_chain(tensor, max_depth=200):
    """Walk the grad_fn chain and print each node."""
    node = tensor.grad_fn
    depth = 0
    nodes = []
    while node is not None and depth < max_depth:
        nodes.append((depth, type(node).__name__))
        # Follow the first input in the graph
        next_fns = node.next_functions
        node = None
        for fn, _ in next_fns:
            if fn is not None:
                node = fn
                break
        depth += 1
    return nodes


def count_node_types(tensor, max_depth=2000):
    """BFS through the entire autograd graph and count node types."""
    from collections import Counter, deque
    visited = set()
    queue = deque()
    counter = Counter()

    root = tensor.grad_fn
    if root is None:
        return counter
    queue.append(root)
    visited.add(id(root))

    while queue:
        node = queue.popleft()
        name = type(node).__name__
        counter[name] += 1

        for fn, _ in node.next_functions:
            if fn is not None and id(fn) not in visited:
                visited.add(id(fn))
                queue.append(fn)

    return counter


def test_simple_autograd_chain():
    """Test that a simple chain of linear layers preserves autograd."""
    print("=" * 60)
    print("TEST 1: Simple chain of linear layers")
    print("=" * 60)

    num_layers = 4
    layers = [torch.nn.Linear(64, 64).cuda() for _ in range(num_layers)]

    x = torch.randn(8, 64, device='cuda', requires_grad=True)

    # Forward
    h = x
    for i, layer in enumerate(layers):
        h = layer(h)
        h = torch.relu(h)

    loss = h.sum()

    # Check autograd graph
    nodes = count_node_types(loss)
    print(f"Autograd graph node types: {dict(nodes)}")

    # Add backward hooks
    layer_bwd_called = {i: False for i in range(num_layers)}
    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, grad_input, grad_output):
                layer_bwd_called[idx] = True
                print(f"  Layer {idx} backward: grad_output shape = {grad_output[0].shape if grad_output[0] is not None else None}")
            return hook
        h = layer.register_full_backward_hook(make_hook(i))
        hooks.append(h)

    # Backward
    loss.backward()

    for i in range(num_layers):
        status = "OK" if layer_bwd_called[i] else "MISSING"
        print(f"  Layer {i}: backward called = {layer_bwd_called[i]} [{status}]")

    assert all(layer_bwd_called.values()), "Not all layers received backward gradients!"
    print("PASSED: All layers received backward gradients.\n")


def test_deallocate_backward():
    """Test that deallocate_output_tensor + custom_backward preserves full autograd."""
    from torch.autograd import Variable

    print("=" * 60)
    print("TEST 2: deallocate + custom_backward chain")
    print("=" * 60)

    num_layers = 4
    layers = [torch.nn.Linear(64, 64).cuda() for _ in range(num_layers)]

    x = torch.randn(8, 64, device='cuda', requires_grad=True)

    # Forward
    h = x
    for layer in layers:
        h = layer(h)
        h = torch.relu(h)

    loss = h.sum()

    # Deallocate output (mimic deallocate_output_tensor)
    loss.data = torch.empty((1,), device=loss.device, dtype=loss.dtype)

    # Backward via C++ engine (mimic custom_backward)
    grad_output = torch.ones_like(loss, memory_format=torch.preserve_format)

    layer_bwd_called = {i: False for i in range(num_layers)}
    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(idx):
            def hook(module, grad_input, grad_output):
                layer_bwd_called[idx] = True
            return hook
        h = layer.register_full_backward_hook(make_hook(i))
        hooks.append(h)

    Variable._execution_engine.run_backward(
        tensors=(loss,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

    for i in range(num_layers):
        status = "OK" if layer_bwd_called[i] else "MISSING"
        print(f"  Layer {i}: backward called = {layer_bwd_called[i]} [{status}]")

    assert all(layer_bwd_called.values()), "Not all layers received backward after deallocate!"
    print("PASSED: custom_backward propagated through all layers.\n")


def test_index_copy_autograd():
    """Test that index_copy_ (used in unpermute) preserves autograd."""
    print("=" * 60)
    print("TEST 3: index_copy_ autograd preservation")
    print("=" * 60)

    x = torch.randn(6, 32, device='cuda', requires_grad=True)
    indices = torch.tensor([3, 1, 5, 0, 4, 2], device='cuda')

    # Forward: index_copy_ (simulates unpermute)
    out = torch.zeros_like(x)
    out.index_copy_(0, indices, x)

    loss = out.sum()
    loss.backward()

    has_grad = x.grad is not None and (x.grad != 0).any()
    print(f"  x.grad exists and nonzero: {has_grad}")
    assert has_grad, "index_copy_ broke autograd!"
    print("PASSED: index_copy_ preserves autograd.\n")


def test_alltoall_scaling_path():
    """Test the _AllToAll autograd Function with scaling-mode path."""
    print("=" * 60)
    print("TEST 4: _AllToAll autograd Function (scaling mode path)")
    print("=" * 60)

    # We need torch.distributed initialized for _AllToAll
    # Instead, test the raw logic: new_zeros + copy_ inside an autograd.Function
    class FakeAllToAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_, output_rows):
            ctx.input_rows = input_.shape[0]
            ctx.output_rows = output_rows
            # Mimic scaling mode _profiled_all_to_all_single
            output = input_.new_zeros(output_rows, input_.shape[1])
            rows_to_copy = min(output_rows, input_.shape[0])
            if rows_to_copy > 0:
                output[:rows_to_copy].copy_(input_[:rows_to_copy])
            return output

        @staticmethod
        def backward(ctx, grad_output):
            # Mimic the reverse all-to-all
            output = grad_output.new_zeros(ctx.input_rows, grad_output.shape[1])
            rows_to_copy = min(ctx.input_rows, grad_output.shape[0])
            if rows_to_copy > 0:
                output[:rows_to_copy].copy_(grad_output[:rows_to_copy])
            return output, None

    x = torch.randn(10, 32, device='cuda', requires_grad=True)
    y = FakeAllToAll.apply(x, 8)  # simulate output with fewer rows

    loss = y.sum()
    loss.backward()

    has_grad = x.grad is not None
    print(f"  x.grad exists: {has_grad}")
    if has_grad:
        nonzero = (x.grad != 0).sum().item()
        print(f"  x.grad nonzero elements: {nonzero} / {x.grad.numel()}")
    assert has_grad, "_AllToAll scaling path broke autograd!"
    print("PASSED: _AllToAll preserves autograd in scaling mode.\n")


def test_chain_with_alltoall():
    """Test that a chain of layers with interleaved all-to-all ops preserves autograd."""
    print("=" * 60)
    print("TEST 5: Layer chain with interleaved all-to-all (simulated MoE)")
    print("=" * 60)

    class FakeAllToAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_):
            output = input_.new_zeros(input_.shape)
            output.copy_(input_)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.new_zeros(grad_output.shape)
            output.copy_(grad_output)
            return output

    num_layers = 4
    attn_layers = [torch.nn.Linear(64, 64).cuda() for _ in range(num_layers)]
    ffn_layers = [torch.nn.Linear(64, 64).cuda() for _ in range(num_layers)]

    x = torch.randn(8, 64, device='cuda', requires_grad=True)

    h = x
    for i in range(num_layers):
        # Attention
        h = attn_layers[i](h) + h  # residual
        # MoE FFN with simulated all-to-all
        permuted = FakeAllToAll.apply(h)
        ffn_out = ffn_layers[i](permuted)
        unpermuted = FakeAllToAll.apply(ffn_out)
        h = unpermuted + h  # residual

    loss = h.sum()

    # Add hooks
    attn_bwd = {i: False for i in range(num_layers)}
    ffn_bwd = {i: False for i in range(num_layers)}
    for i in range(num_layers):
        def make_attn_hook(idx):
            def hook(m, gi, go): attn_bwd[idx] = True
            return hook
        def make_ffn_hook(idx):
            def hook(m, gi, go): ffn_bwd[idx] = True
            return hook
        attn_layers[i].register_full_backward_hook(make_attn_hook(i))
        ffn_layers[i].register_full_backward_hook(make_ffn_hook(i))

    loss.backward()

    all_ok = True
    for i in range(num_layers):
        a_status = "OK" if attn_bwd[i] else "MISSING"
        f_status = "OK" if ffn_bwd[i] else "MISSING"
        print(f"  Layer {i}: attn_bwd={attn_bwd[i]} [{a_status}], ffn_bwd={ffn_bwd[i]} [{f_status}]")
        if not attn_bwd[i] or not ffn_bwd[i]:
            all_ok = False

    assert all_ok, "Not all layers received backward in chain with all-to-all!"
    print("PASSED: All layers received backward through chain with all-to-all.\n")


def test_graph_depth():
    """Count autograd graph depth for a 4-layer chain to establish baseline."""
    print("=" * 60)
    print("TEST 6: Autograd graph depth analysis")
    print("=" * 60)

    num_layers = 4
    layers = [torch.nn.Linear(64, 64).cuda() for _ in range(num_layers)]

    x = torch.randn(8, 64, device='cuda', requires_grad=True)
    h = x
    for layer in layers:
        h = layer(h)
        h = torch.relu(h)

    loss = h.sum()

    # Count total nodes in autograd graph
    nodes = count_node_types(loss)
    total = sum(nodes.values())
    print(f"  Total autograd nodes: {total}")
    print(f"  Key node types: {dict(nodes.most_common(10))}")

    # Walk the DFS chain
    chain = trace_grad_fn_chain(loss)
    print(f"  DFS chain depth: {len(chain)}")
    for d, name in chain[:20]:
        print(f"    [{d}] {name}")
    if len(chain) > 20:
        print(f"    ... ({len(chain) - 20} more)")
    print()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        sys.exit(0)

    torch.manual_seed(42)

    test_simple_autograd_chain()
    test_deallocate_backward()
    test_index_copy_autograd()
    test_alltoall_scaling_path()
    test_chain_with_alltoall()
    test_graph_depth()

    print("=" * 60)
    print("ALL BASIC AUTOGRAD TESTS PASSED")
    print("=" * 60)
    print("\nNote: The above tests verify basic autograd mechanics.")
    print("The actual scaling mode bug may be in more complex interactions.")
    print("Next step: Run the actual model with layer-level backward hooks.")
