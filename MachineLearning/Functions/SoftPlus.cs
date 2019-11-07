using static System.Math;

namespace MachineLearning.Functions
{
    public class SoftPlus : ActivationFunction, IActivationFunction
    {
        public SoftPlus(double alpha) : base(alpha) { }

        public double Activate(double x) => Log(1 + Exp(alpha * x));

        public double Derivate(double x) => alpha / (1 + Exp(-x * alpha));
    }
}
