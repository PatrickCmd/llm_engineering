
#include <iostream>
#include <chrono>
#include <iomanip>

// This function calculates an approximation of pi/4 using a Leibniz-like series.
// The implementation is heavily optimized for single-threaded performance, as
// required by the provided compilation command which does not include parallelization flags.
//
// Optimizations used:
// 1. Loop Unrolling: The main loop is unrolled by a factor of 8. This reduces
//    loop overhead and increases the amount of work done per iteration.
// 2. Multiple Accumulators: Eight separate accumulator variables (sum0 to sum7)
//    are used. This breaks the dependency chain present in a single accumulator,
//    allowing the CPU's out-of-order execution engine to process multiple
//    calculations simultaneously and better utilize its pipelines.
// 3. Formula Simplification: The original Python code performs two divisions per loop.
//    The expression `(1/(4i+1)) - (1/(4i-1))` is algebraically simplified to
//    `-2 / (16i^2 - 1)`. This reduces the number of expensive division operations
//    to one per original iteration. The `-ffast-math` compiler flag permits such
//    re-association of floating-point math, so this is a safe and effective optimization.
// 4. Constant Hoisting: All loop-invariant calculations are performed once before the loop begins.
double calculate(long long iterations, int param1, int param2) {
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    double sum4 = 0.0, sum5 = 0.0, sum6 = 0.0, sum7 = 0.0;

    const double p1d = static_cast<double>(param1);
    const double p2d = static_cast<double>(param2);
    
    const double term_num = -2.0 * p2d;
    const double p1d_sq = p1d * p1d;
    const double p2d_sq = p2d * p2d;

    const long long limit = iterations - (iterations % 8);
    long long i = 1;

    for (; i <= limit; i += 8) {
        double d0 = static_cast<double>(i + 0);
        double d1 = static_cast<double>(i + 1);
        double d2 = static_cast<double>(i + 2);
        double d3 = static_cast<double>(i + 3);
        double d4 = static_cast<double>(i + 4);
        double d5 = static_cast<double>(i + 5);
        double d6 = static_cast<double>(i + 6);
        double d7 = static_cast<double>(i + 7);

        sum0 += term_num / (p1d_sq * d0 * d0 - p2d_sq);
        sum1 += term_num / (p1d_sq * d1 * d1 - p2d_sq);
        sum2 += term_num / (p1d_sq * d2 * d2 - p2d_sq);
        sum3 += term_num / (p1d_sq * d3 * d3 - p2d_sq);
        sum4 += term_num / (p1d_sq * d4 * d4 - p2d_sq);
        sum5 += term_num / (p1d_sq * d5 * d5 - p2d_sq);
        sum6 += term_num / (p1d_sq * d6 * d6 - p2d_sq);
        sum7 += term_num / (p1d_sq * d7 * d7 - p2d_sq);
    }
    
    // Combine the partial sums.
    double total_sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    // Handle the remaining iterations that are not a multiple of 8.
    for (; i <= iterations; ++i) {
        double i_d = static_cast<double>(i);
        total_sum += term_num / (p1d_sq * i_d * i_d - p2d_sq);
    }

    // Add the initial 1.0 from the Python code.
    return 1.0 + total_sum;
}


int main() {
    // Use fast I/O by decoupling C++ streams from C stdio.
    std::ios_base::sync_with_stdio(false);

    constexpr long long iterations = 200'000'000;
    constexpr int param1 = 4;
    constexpr int param2 = 1;

    auto start_time = std::chrono::high_resolution_clock::now();

    double result = calculate(iterations, param1, param2) * 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print results with specified precision, using '\n' for speed.
    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << '\n';
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds\n";

    return 0;
}
