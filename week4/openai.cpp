#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    constexpr uint64_t iterations = 200000000ULL;

    auto start = chrono::high_resolution_clock::now();

    // Compute: result = 1.0 + sum_{i=1..N} (-2) / ((4i-1)*(4i+1)) = 1.0 - sum 2/(16*i*i - 1)
    // Unrolled for speed, minimizing divisions (1 per i).
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;

    uint64_t i = 1;
    const uint64_t unroll = 8;
    const uint64_t limit = iterations - (iterations % unroll);

    for (; i < limit + 1; i += unroll) {
        uint64_t q0 = (i << 2);
        uint64_t q1 = q0 + 4;
        uint64_t q2 = q0 + 8;
        uint64_t q3 = q0 + 12;
        uint64_t q4 = q0 + 16;
        uint64_t q5 = q0 + 20;
        uint64_t q6 = q0 + 24;
        uint64_t q7 = q0 + 28;

        double d0 = double(q0 * q0 - 1ULL);
        double d1 = double(q1 * q1 - 1ULL);
        double d2 = double(q2 * q2 - 1ULL);
        double d3 = double(q3 * q3 - 1ULL);
        double d4 = double(q4 * q4 - 1ULL);
        double d5 = double(q5 * q5 - 1ULL);
        double d6 = double(q6 * q6 - 1ULL);
        double d7 = double(q7 * q7 - 1ULL);

        s0 -= 2.0 / d0;
        s1 -= 2.0 / d1;
        s2 -= 2.0 / d2;
        s3 -= 2.0 / d3;
        s4 -= 2.0 / d4;
        s5 -= 2.0 / d5;
        s6 -= 2.0 / d6;
        s7 -= 2.0 / d7;
    }

    for (; i <= iterations; ++i) {
        uint64_t q = (i << 2);
        double d = double(q * q - 1ULL);
        s0 -= 2.0 / d;
    }

    double result = 1.0 + (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7);
    result *= 4.0;

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    cout.setf(std::ios::fixed);
    cout << setprecision(12) << "Result: " << result << "\n";
    cout << setprecision(6) << "Execution Time: " << elapsed << " seconds\n";
    return 0;
}