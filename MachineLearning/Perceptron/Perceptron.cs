using Newtonsoft.Json;
using System.Diagnostics;
using System.IO;
using MachineLearning.LinearAlgebra;
using static System.Console;

namespace MachineLearning.Perceptron
{
    /// <summary>
    /// Перцептрон
    /// </summary>
    class Perceptron
    {
        /// <summary>
        /// Слои перцептрона
        /// </summary>
        Layer[] layers;

        /// <summary>
        /// Индекс выходного слоя
        /// </summary>
        int outputLayerIndex;

        /// <summary>
        /// Конструктор.
        /// На вход следует подавать данные без учёта входного слоя, 
        /// так как он не является вычислительным
        /// </summary>
        /// <param name="configs">Конфигурация слоёв</param>
        public Perceptron(params LayerConfig[] configs)
        {
            if (configs != null)
            {
                layers = new Layer[configs.Length];
                outputLayerIndex = layers.Length - 1;
                for (int i = 0; i <= outputLayerIndex; i++)
                    layers[i] = new Layer(configs[i]);
            }
        }

        /// <summary>
        /// Получить предсказание перцептрона
        /// </summary>
        /// <param name="I">Вектор входных данных</param>
        /// <returns>Некий вектор выходных данных</returns>
        public Vector Predict(Vector I)
        {
            Vector Oj = layers[0].Compute(I);
            for (int j = 1; j < layers.Length; j++)
            {
                Oj = layers[j].Compute(Oj);
            }
            return Oj;
        }

        /// <summary>
        /// Обучение перцептрона методом обратного распространения ошибки
        /// </summary>
        /// <param name="trainset">Обучающая выборка</param>
        /// <param name="threshold">Порог функции стоимости, по умолчанию 0.001</param>
        public void TrainBackProp((Vector X, Vector Y)[] trainset, double threshold = 0.001d)
        {
            var w = Stopwatch.StartNew();
            Vector MSEs = Vector.Zero(trainset.Length);
            double epochCost = 0;
            int epoch = 0;
            do
            {
                for (int i = 0; i < trainset.Length; i++)
                {
                    Vector O = Predict(trainset[i].X);
                    Vector E = trainset[i].Y - O;
                    MSEs[i] = GetMSE(E);
                    Vector GSums = layers[outputLayerIndex].ComputeOutputBackward(E);
                    for (int j = outputLayerIndex - 1; j >= 0; j--)
                    {
                        GSums = layers[j].ComputeHiddenBackward(GSums);
                    }
                }
                epoch++;
                epochCost = GetCost(MSEs);
#if DEBUG
                WriteLine($"{epochCost} - epoch {epoch}");
#endif
            } while (epochCost > threshold);
            w.Stop();
#if DEBUG
            WriteLine($"{w.ElapsedMilliseconds / 1000d} seconds");
#endif
        }

        /// <summary>
        /// Загрузить модель из JSON файла
        /// </summary>
        /// <param name="path">Путь к файлу</param>
        public void LoadModel(string path)
        {
            var settings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
                TypeNameHandling = TypeNameHandling.Auto
            };
            var model = new[]
            {
                new
                {
                    config = new LayerConfig(),
                    weights = new double[1][]
                }
            };
            using (StreamReader r = new StreamReader(path))
            {
                string json = r.ReadToEnd();
                model = JsonConvert.DeserializeAnonymousType(json, model, settings);
            }
            if (model != null)
            {
                layers = new Layer[model.Length];
                outputLayerIndex = layers.Length - 1;
                for (int i = 0; i < layers.Length; i++)
                {
                    layers[i] = new Layer(model[i].config);
                    layers[i].SetWeights(model[i].weights);
                }
            }
        }
        
        /// <summary>
        /// Сохранить модель в JSON файл
        /// </summary>
        /// <param name="path">Путь сохранения</param>
        public void SaveModel(string path)
        {
            var settings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
                TypeNameHandling = TypeNameHandling.Auto
            };

            var model = new object[layers.Length];
            for (int i = 0; i < model.Length; i++)
            {
                model[i] = new
                {
                    config = layers[i].Config,
                    weights = layers[i].GetWeights()
                };
            }

            using (StreamWriter file = File.CreateText($"{path}model{System.DateTime.Now.Ticks}.json"))
            {
                JsonSerializer serializer = new JsonSerializer
                {
                    Formatting = Formatting.Indented,
                    TypeNameHandling = TypeNameHandling.Auto
                };
                serializer.Serialize(file, model);
            }
        }

        /// <summary>
        /// Ошибка одной итерации обучения
        /// </summary>
        /// <param name="errors">Вектор ошибок итерации обучения</param>
        /// <returns></returns>
        double GetMSE(Vector errors) => (errors * errors) * 0.5d;

        /// <summary>
        /// Ошибка эпохи
        /// </summary>
        /// <param name="mses">Вектор ошибок итераций</param>
        /// <returns></returns>
        double GetCost(Vector mses) => (mses * Vector.E(mses.Size)) / mses.Size;
    }
}
