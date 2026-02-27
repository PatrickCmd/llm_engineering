


#include <cstdio>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    constexpr int iterations = 200000000;
    constexpr double param1 = 4.0;
    constexpr double param2 = 1.0;
    
    double result0 = 0.0;
    double result1 = 0.0;
    double result2 = 0.0;
    double result3 = 0.0;
    
    for (int i = 1; i <= iterations; i += 4) {
        double j0m = i * param1 - param2;
        double j0p = i * param1 + param2;
        double j1m = (i+1) * param1 - param2;
        double j1p = (i+1) * param1 + param2;
        double j2m = (i+2) * param1 - param2;
        double j2p = (i+2) * param1 + param2;
        double j3m = (i+3) * param1 - param2;
        double j3p = (i+3) * param1 + param2;
        
        result0 += (1.0/j0p - 1.0/j0m);
        result1 += (1.0/j1p - 1.0/j1m);
        result2 += (1.0/j2p - 1.0/j2m);
        result3 += (1.0/j3p - 1.0/j3m);
    }
    
    double result = (1.0 + result0 + result1 + result2 + result3) * 4.0;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    printf("Result: %.12f\n", result);
    printf("Execution Time: %.6f seconds\n", elapsed);
    
    return 0;
}
