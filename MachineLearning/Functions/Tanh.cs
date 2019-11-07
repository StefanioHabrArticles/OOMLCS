using Newtonsoft.Json;
using System;
using static System.Math;

namespace MachineLearning.Functions
{
    /// <summary>
    /// Гиперболический тангенс
    /// </summary>
    class Tanh : ActivationFunction, IActivationFunction
    {
        [JsonProperty("beta")]
        private double beta;

        /// <summary>
        /// Удобные значения параметров:
        /// alpha = 1.7159 ;
        /// beta = 2 / 3 .
        /// </summary>
        public Tanh(double alpha, double beta) : base(alpha) => this.beta = beta;

        public double Activate(double x) => alpha * Tanh(beta * x);

        public double Derivate(double x)
        {
            double f = Activate(x);
            double c = beta / alpha;
            return c * (alpha * alpha - f * f);
        }
    }
}
