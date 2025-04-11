#include <metal_stdlib>
using namespace metal;

// Optimized LSTM kernel with temporal unrolling and threadgroup optimizations
kernel void lstm_optimized(
    device const float* x [[buffer(0)]],
    device const float* h_in [[buffer(1)]],
    device const float* cell [[buffer(2)]],
    device float* hidden_state [[buffer(3)]],
    device float* cell_state [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& time_step [[buffer(6)]],
    constant uint& num_time_steps [[buffer(7)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    // Threadgroup configuration: 32x8x1
    const uint b = tgid.x;
    const uint d = hidden_size * 4;
    const uint unroll_factor = 4;
    
    // Shared memory for intermediate results
    threadgroup float gates[32][4]; // [threads][unroll steps]
    
    // Process 4 time steps per thread
    for (uint t = 0; t < unroll_factor; ++t) {
        uint current_step = time_step + t;
        if (current_step >= num_time_steps) break;
        
        uint elem = b * d + tid.y;
        uint x_index = b * num_time_steps * d + current_step * d + elem;
        
        // Compute gates with fused operations
        float val = h_in[elem] + x[x_index];
        gates[tid.y][t] = 1.0f / (1.0f + exp(-fabs(val))) * (val < 0 ? -1.0f : 1.0f) + 1.0f;
    }
    
    // Synchronize threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write results
    for (uint t = 0; t < unroll_factor; ++t) {
        uint current_step = time_step + t;
        if (current_step >= num_time_steps) break;
        
        uint elem = b * d + tid.y;
        float i = gates[tid.y][t];
        float f = gates[tid.y + hidden_size][t];
        float g = tanh(h_in[elem + 2*hidden_size] + x[b * num_time_steps * d + current_step * d + elem + 2*hidden_size]);
        float o = gates[tid.y + 3*hidden_size][t];
        
        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * tanh(cell_state[elem]);
    }
}
