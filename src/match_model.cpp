// Checks the C++ forward pass against the Python reference.
// Reads <stem>.meta (+ its .bin) and <stem>.check (raw features + reference logit),
// runs the C++ forward pass on each row, and reports the max abs difference.
// Exits non-zero if any row exceeds the tolerance.
// Usage: match_model <model-stem>   e.g. match_model data/model
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "model.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <model-stem>\n";
        return 2;
    }
    const std::string stem = argv[1];
    const double tol = 1e-5;

    Model model;
    try {
        model = Model::load(stem + ".meta");
    } catch (const std::exception& e) {
        std::cerr << "load error: " << e.what() << "\n";
        return 2;
    }

    std::ifstream cf(stem + ".check");
    if (!cf) {
        std::cerr << "cannot open check file: " << stem << ".check\n";
        return 2;
    }

    int n = 0, d = 0;
    cf >> n >> d;
    if (d != model.input_dim()) {
        std::cerr << "check dim " << d << " != model input_dim "
                  << model.input_dim() << "\n";
        return 2;
    }

    double max_err = 0.0;
    int worst = -1;
    int checked = 0;
    for (int r = 0; r < n; ++r) {
        std::vector<float> feats(d);
        for (int i = 0; i < d; ++i) cf >> feats[i];
        double ref = 0.0;
        cf >> ref;
        if (!cf) {
            std::cerr << "check file truncated at row " << r << "\n";
            return 2;
        }

        const double got = model.forward(feats);
        const double err = std::fabs(got - ref);
        if (err > max_err) {
            max_err = err;
            worst = r;
        }
        ++checked;
    }

    std::cout << "checked " << checked << " rows\n";
    std::cout << "max |c++ - reference| = " << max_err
              << " (worst row " << worst << ")\n";
    std::cout << "tolerance = " << tol << "\n";

    if (max_err < tol) {
        std::cout << "MATCH\n";
        return 0;
    }
    std::cout << "MISMATCH\n";
    return 1;
}
