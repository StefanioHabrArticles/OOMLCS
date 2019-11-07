using MachineLearning.LinearAlgebra;

namespace MachineLearning.Perceptron
{
    /// <summary>
    /// Нейрон
    /// </summary>
    internal class Neuron
    {
        private Vector inputs;

        /// <summary>
        /// Конструктор нейрона по весам
        /// </summary>
        /// <param name="weights">Вектор весов</param>
        /// <param name="bias">Смещение - нулевой вес</param>
        /// <param name="af">Интерфейс функции активации</param>
        public Neuron(Vector weights, double bias, IActivationFunction af)
        {
            Weights = weights;
            Bias = bias;
            activationFunction = af;
        }

        /// <summary>
        /// Вектор входных данных
        /// </summary>
        public Vector Inputs
        {
            get => inputs;
            set
            {
                inputs = value;
                Activator();
            }
        }
        
        /// <summary>
        /// Вектор весов нейрона
        /// </summary>
        public Vector Weights { get; set; }

        /// <summary>
        /// Смещение нейрона
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// Функция активации
        /// </summary>
        private readonly IActivationFunction activationFunction;

        /// <summary>
        /// Выход нейрона - его "возбуждённое" состояние
        /// </summary>
        public double Output { get; private set; }

        /// <summary>
        /// Производная выхода нейрона
        /// </summary>
        public double Derivative { get; private set; }

        /// <summary>
        /// "Возбуждение" нейрона - вычисление состояния
        /// </summary>
        private void Activator()
        {
            double sum = Bias + (Inputs * Weights);
            Output = activationFunction.Activate(sum);
            Derivative = activationFunction.Derivate(sum);
        }
    }
}
