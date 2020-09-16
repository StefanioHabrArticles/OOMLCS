# OOMLCS
Object-oriented machine learning with C#

## Примеры

### Аппроксимация функции

__[Модель, полученная в результате обучения](https://gist.github.com/Stepami/6ad7f040d82b635ba150c8d0ef58ec52)__

В этом примере мы [аппроксимируем функцию](https://en.wikipedia.org/wiki/Function_approximation), заданную следующими значениями:
|x|f(x)|
|---|---|
|1|20| 
|2|346.12814956|
|3|440.811685833281|
|4|506.386394877906|
|5|553.932275490213|
|6|588.456079653438|
|7|613.525471204964|
|8|631.72958494378|
|9|644.948484588186|
|10|654.547375835747|
|11|661.517601472129|
|12|666.579024106174|
|13|670.254371278109|
|14|672.923221087921|
|15|674.861203807411|
|16|676.268468093033|
|17|677.290351698946|
|18|678.032391495382|
|19|678.571222979033|
|20|678.962494942032|

[Известно, что перцептрон с одним скрытым слоем может аппроксимировать любую непрерывную функцию многих переменных с любой точностью.](https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D0%BE%D1%80%D0%B5%D0%BC%D0%B0_%D0%A6%D1%8B%D0%B1%D0%B5%D0%BD%D0%BA%D0%BE "Теорема Цыбенко")

Создадим перцептрон:
```cs
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
```
Подготовим обучающие данные:
```cs
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
```
Обучим и проверим:
```cs
// обратное распространение ошибки
approximator.TrainBackProp(trainset, 0.0009);
// получение предскзания
for (int i = 0; i < trainset.Length; i++)
    Console.WriteLine($"f({i + 1}) = {approximator.Predict(new Vector(i + 1))[0] * std + avg}");
```
Загрузка и сохранение модели
```cs
// сохранение
string modelFile = approximator.SaveModel(@"C:\Some\path");
// загрузка
var approximator = new Perceptron(null);
approximator.LoadModel($@"C:\Some\path\{ModelFile}");
Console.WriteLine(approximator.Predict(new Vector(3.4567d)));
```
