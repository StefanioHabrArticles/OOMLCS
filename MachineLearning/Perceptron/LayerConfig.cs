namespace MachineLearning.Perceptron
{
    /// <summary>
    /// Конфигурация слоя перцептрона
    /// </summary>
    public class LayerConfig
    {
        /// <summary>
        /// Число нейронов текущего слоя
        /// </summary>
        public int NumOfNeurons { get; set; }

        /// <summary>
        /// Число нейронов предыдущего слоя
        /// </summary>
        public int NumOfPrevNeurons { get; set; }

        /// <summary>
        /// Скорость обучения
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Момент инерции
        /// </summary>
        public double Momentum { get; set; }

        /// <summary>
        /// Функция активации нейронов текущего слоя
        /// </summary>
        public IActivationFunction ActivationFunction { get; set; }
    }
}
