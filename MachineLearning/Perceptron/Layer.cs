using MachineLearning.LinearAlgebra;

namespace MachineLearning.Perceptron
{
    /// <summary>
    /// Слой перцептрона
    /// </summary>
    internal class Layer
    {
        /// <summary>
        /// Матрица весов на предыдущей итерации обучения
        /// </summary>
        private double[][] lastDeltaWeights;

        /// <summary>
        /// Нейроны этого слоя
        /// </summary>
        private Neuron[] neurons;

        /// <summary>
        /// Конфигурация текущего слоя
        /// </summary>
        public LayerConfig Config { get; }

        /// <summary>
        /// Конструктор слоя
        /// </summary>
        /// <param name="config">Объект конфигурации слоя</param>
        public Layer(LayerConfig config)
        {
            Config = config;
            lastDeltaWeights = new double[config.NumOfNeurons][];
            neurons = new Neuron[config.NumOfNeurons];
            int t = config.NumOfNeurons + config.NumOfPrevNeurons;
            for (int i = 0; i < config.NumOfNeurons; i++)
            {
                var W = Vector.RandomVector(config.NumOfPrevNeurons + 1, -0.25d / t, 0.25d / t);
                lastDeltaWeights[i] = W.Slice(0, config.NumOfPrevNeurons);
                neurons[i] = new Neuron(new Vector(W.Slice(1, config.NumOfPrevNeurons)), W[0], config.ActivationFunction);
            }
        }

        /// <summary>
        /// Вычисление состояния слоя
        /// </summary>
        /// <param name="I">Вектор входных данных</param>
        /// <returns>Вектор выхода слоя</returns>
        public Vector Compute(Vector I)
        {
            double[] o = new double[neurons.Length];
            for (int i = 0; i < Config.NumOfNeurons; i++)
            {
                neurons[i].Inputs = I;
                o[i] = neurons[i].Output;
            }
            return new Vector(o);
        }

        /// <summary>
        /// Вычисления при обратном проходе для скрытого слоя
        /// </summary>
        /// <param name="GrSums">Градиентные суммы предыдущего слоя</param>
        /// <returns>Вектор градиентных сумм этого слоя</returns>
        public Vector ComputeHiddenBackward(Vector GrSums)
        {
            Vector NewGrSums = Vector.Zero(Config.NumOfPrevNeurons + 1);
            // подсчёт градиентных сумм скрытого слоя
            for (int j = 0; j < NewGrSums.Size; j++)
            {
                double sum = 0;
                for (int k = 0; k < neurons.Length; k++)
                {
                    if (j == 0)
                    {
                        sum += neurons[k].Bias * neurons[k].Derivative * GrSums[k];
                    }
                    else
                    {
                        sum += neurons[k].Weights[j - 1] * neurons[k].Derivative * GrSums[k];
                    }
                }
                NewGrSums[j] = sum;
            }
            // обновление смещений и весов
            for (int i = 0; i < Config.NumOfNeurons; ++i)
            {
                double deltabias = Config.Momentum * lastDeltaWeights[i][0] + Config.LearningRate * neurons[i].Derivative * GrSums[i];
                lastDeltaWeights[i][0] = deltabias;
                neurons[i].Bias += deltabias;
                for (int n = 1; n <= Config.NumOfPrevNeurons; ++n)
                {
                    double deltaw = Config.Momentum * lastDeltaWeights[i][n] + Config.LearningRate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * GrSums[i];
                    lastDeltaWeights[i][n] = deltaw;
                    neurons[i].Weights[n - 1] += deltaw;
                }
            }
            return NewGrSums;
        }

        /// <summary>
        /// Вычисления при обратном проходе для выходного слоя
        /// </summary>
        /// <param name="errors">Вектор ошибок сети</param>
        /// <returns>Вектор градиентных сумм этого слоя</returns>
        public Vector ComputeOutputBackward(Vector errors)
        {
            Vector GrSums = Vector.Zero(Config.NumOfPrevNeurons + 1);
            //вычисление градиентных сумм выходного слоя
            for (int j = 0; j < GrSums.Size; j++)
            {
                double sum = 0;
                for (int k = 0; k < neurons.Length; ++k)
                {
                    if (j == 0)
                    {
                        sum += neurons[k].Bias * errors[k];
                    }
                    else
                    {

                        sum += neurons[k].Weights[j - 1] * errors[k];
                    }
                }
                GrSums[j] = sum;
            }
            // обновление смещений и весов
            for (int i = 0; i < Config.NumOfNeurons; ++i)
            {
                double deltabias = Config.Momentum * lastDeltaWeights[i][0] + Config.LearningRate * errors[i];
                lastDeltaWeights[i][0] = deltabias;
                neurons[0].Bias += deltabias;
                for (int n = 1; n <= Config.NumOfPrevNeurons; ++n)
                {
                    double deltaw = Config.Momentum * lastDeltaWeights[i][n] + Config.LearningRate * neurons[i].Inputs[n - 1] * errors[i];
                    lastDeltaWeights[i][n] = deltaw;
                    neurons[i].Weights[n - 1] += deltaw;
                }
            }
            return GrSums;
        }

        public void SetWeights(double[][] weigths)
        {
            for (int i = 0; i < Config.NumOfNeurons; i++)
            {
                neurons[i].Bias = weigths[i][0];
                neurons[i].Weights = new Vector(new Vector(weigths[i]).Slice(1, Config.NumOfPrevNeurons));
            }
        }

        public double[][] GetWeights()
        {
            double[][] weights = new double[Config.NumOfNeurons][];

            for (int i = 0; i < Config.NumOfNeurons; i++)
            {
                weights[i] = new double[Config.NumOfPrevNeurons + 1];
                weights[i][0] = neurons[i].Bias;
                for (int j = 1; j < weights[i].Length; j++)
                    weights[i][j] = neurons[i].Weights[j - 1];
            }

            return weights;
        }
    }
}
