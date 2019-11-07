using System;
using static System.Math;

namespace MachineLearning.Functions
{
    /// <summary>
    /// Сигмоида
    /// </summary>
    class Sigmoid : ActivationFunction, IActivationFunction
    {
        public Sigmoid(double alpha) : base(alpha) { }

        public double Activate(double x) => 1 / (1 + Exp(alpha * -x));

        public double Derivate(double x)
        {
            double f = Activate(x);
            return alpha * f * (1 - f);
        }
    }
}
