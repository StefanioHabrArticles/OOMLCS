using System;
using System.Linq;
using MachineLearning.Functions;
using MachineLearning.LinearAlgebra;
using MachineLearning.Perceptron;

namespace OOPerceptron
{
    class Program
    {
        string ModelFile { get; set; }

        static void Main(string[] args)
        {
            var p = new Program();
            p.Example1();
            p.Example2();
        }

        void Example1()
        {
            // аппроксиматор функции
            var approximator = new Perceptron
            (
                new LayerConfig
                {
                    NumOfNeurons = 100,
                    NumOfPrevNeurons = 1,
                    LearningRate = 0.005d,
                    Momentum = 0.005d,
                    ActivationFunction = new Tanh(1, 1)
                },
                new LayerConfig
                {
                    NumOfNeurons = 1,
                    NumOfPrevNeurons = 100,
                    LearningRate = 0.01d,
                    Momentum = 0.005d,
                    ActivationFunction = new LeakyReLU(0.1d)
                }
            );

            // значения функции
            var y = new double[] 
            {
                20,
                346.12814956,
                440.811685833281,
                506.386394877906,
                553.932275490213,
                588.456079653438,
                613.525471204964,
                631.72958494378,
                644.948484588186,
                654.547375835747,
                661.517601472129,
                666.579024106174,
                670.254371278109,
                672.923221087921,
                674.861203807411,
                676.268468093033,
                677.290351698946,
                678.032391495382,
                678.571222979033,
                678.962494942032
            };
            // обучающая выборка
            var trainset = new (Vector x, Vector y)[20];
            for (int i = 0; i < trainset.Length; i++)
                trainset[i] = (new Vector(i + 1), new Vector(y[i]));

            // среднее
            var avg = trainset.Select(t => t.y[0]).Average();
            // дисперсия
            var std = (new Vector(trainset.Select(t => t.y[0]).ToArray()) - Vector.VecByConst(Vector.E(trainset.Length), avg)).Length / Math.Sqrt(trainset.Length);
            // нормализация данных
            trainset = trainset.Select(t => (t.x, new Vector((t.y[0] - avg) / std))).ToArray();

            // обучение
            approximator.TrainBackProp(trainset, 0.0009);

            // использование
            for (int i = 0; i < trainset.Length; i++)
                Console.WriteLine($"f({i + 1}) = {approximator.Predict(new Vector(i + 1))[0] * std + avg}");

            // сохранение модели
            ModelFile = approximator.SaveModel(@"C:\Some\path");
        }

        void Example2()
        {
            // загрузка модели из файла
            var approximator = new Perceptron(null);
            approximator.LoadModel($@"C:\Some\path\{ModelFile}");
            Console.WriteLine(approximator.Predict(new Vector(3.4567d)));
        }
    }
}
