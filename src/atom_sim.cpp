#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

/**
 * This program is not a "perfect and real" atom simulation (that is not
 * computationally feasible in full detail), but it is a physically grounded
 * quantum simulation of a hydrogen atom's 1s electron cloud.
 *
 * Model summary:
 * - The electron probability density for hydrogen 1s is:
 *     p(r) = 4 r^2 / a0^3 * exp(-2r/a0), r >= 0
 * - This is a Gamma distribution with shape k=3 and scale theta=a0/2.
 * - We sample many electron positions, then estimate observable quantities.
 */
class Hydrogen1SSimulator {
 public:
  explicit Hydrogen1SSimulator(unsigned int seed = std::random_device{}())
      : rng_(seed), unit_normal_(0.0, 1.0) {}

  struct Results {
    double mean_radius_m = 0.0;
    double rms_radius_m = 0.0;
    double mean_potential_energy_J = 0.0;
  };

  Results Run(std::size_t samples) {
    Results out;

    double sum_r = 0.0;
    double sum_r2 = 0.0;
    double sum_pe = 0.0;

    for (std::size_t i = 0; i < samples; ++i) {
      const double r = SampleRadius1s();
      sum_r += r;
      sum_r2 += r * r;
      sum_pe += PotentialEnergy(r);
    }

    out.mean_radius_m = sum_r / static_cast<double>(samples);
    out.rms_radius_m = std::sqrt(sum_r2 / static_cast<double>(samples));
    out.mean_potential_energy_J = sum_pe / static_cast<double>(samples);
    return out;
  }

  static constexpr double BohrRadius() { return kBohrRadius; }

 private:
  // Physical constants in SI.
  static constexpr double kElectronCharge = 1.602176634e-19; // C
  static constexpr double kCoulombConstant = 8.9875517923e9; // N m^2 C^-2
  static constexpr double kBohrRadius = 5.29177210903e-11;   // m

  std::mt19937_64 rng_;
  std::normal_distribution<double> unit_normal_;

  static double PotentialEnergy(double r) {
    // Coulomb potential for electron-proton pair: U = -k e^2 / r
    return -kCoulombConstant * kElectronCharge * kElectronCharge / r;
  }

  double SampleRadius1s() {
    // If X_i ~ Exponential(scale=a0/2), then X1+X2+X3 ~ Gamma(k=3, theta=a0/2).
    // Equivalent and efficient: sample Gamma directly.
    std::gamma_distribution<double> gamma_dist(/*alpha=*/3.0,
                                               /*beta=*/kBohrRadius / 2.0);
    return gamma_dist(rng_);
  }
};

int main(int argc, char** argv) {
  std::size_t samples = 1'000'000;
  if (argc > 1) {
    try {
      samples = static_cast<std::size_t>(std::stoull(argv[1]));
      if (samples == 0) {
        std::cerr << "Sample count must be positive.\n";
        return 1;
      }
    } catch (...) {
      std::cerr << "Invalid sample count. Usage: ./atom_sim [samples]\n";
      return 1;
    }
  }

  Hydrogen1SSimulator sim;
  const auto results = sim.Run(samples);

  const double a0 = Hydrogen1SSimulator::BohrRadius();
  const double analytic_mean_r = 1.5 * a0;           // <r> for 1s
  const double analytic_rms_r = std::sqrt(3.0) * a0; // sqrt(<r^2>) for 1s

  std::cout << std::setprecision(8) << std::scientific;
  std::cout << "Hydrogen 1s Monte Carlo simulation\n";
  std::cout << "Samples: " << samples << "\n\n";

  std::cout << "Estimated <r>      = " << results.mean_radius_m << " m\n";
  std::cout << "Analytic <r>       = " << analytic_mean_r << " m\n";
  std::cout << "Estimated RMS(r)   = " << results.rms_radius_m << " m\n";
  std::cout << "Analytic RMS(r)    = " << analytic_rms_r << " m\n";
  std::cout << "Estimated <U>      = " << results.mean_potential_energy_J
            << " J\n";

  std::cout << "\nNote: A truly perfect and fully real atom simulation would require\n"
            << "many-body quantum electrodynamics and is computationally intractable\n"
            << "for exact treatment. This program is a realistic quantum model for\n"
            << "the hydrogen ground-state electron cloud.\n";

  return 0;
}
