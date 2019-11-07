namespace MachineLearning.Functions
{
    /// <summary>
    /// Текучий выпрямитель
    /// </summary>
    class LeakyReLU : ActivationFunction, IActivationFunction
    {
        public LeakyReLU(double alpha) : base(alpha) { }

        public double Activate(double x) => x < 0 ? alpha * x : x;

        public double Derivate(double x) => x < 0 ? -alpha : 1;
    }
}
