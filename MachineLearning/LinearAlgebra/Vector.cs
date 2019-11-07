using System.Linq;
using static System.Math;

namespace MachineLearning.LinearAlgebra
{
    /// <summary>
    /// Вектор из линейной алгебры
    /// </summary>
    class Vector
    {
        /// <summary>
        /// Рандомайзер
        /// </summary>
        private static readonly System.Random rnd = new System.Random();

        /// <summary>
        /// Конструктор по координатам
        /// </summary>
        /// <param name="c">Набор координат</param>
        public Vector(params double[] c)
        {
            Coordinates = c;
            Size = Coordinates.Length;
            Length = Sqrt(Coordinates.Sum(x => x * x));
        }

        /// <summary>
        /// Индексатор
        /// </summary>
        /// <param name="index">Индекс координаты</param>
        /// <returns>Координата вектора по индексу</returns>
        public double this[int index]
        {
            get => Coordinates[index];
            set
            {
                Coordinates[index] = value;
                Length = Sqrt(Coordinates.Sum(x => x * x));
            }
        }

        /// <summary>
        /// Длина вектора
        /// </summary>
        public double Length { get; set; }

        /// <summary>
        /// Координаты вектора
        /// </summary>
        private double[] Coordinates { get; set; }

        /// <summary>
        /// Размерность вектора
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Нормализация вектора
        /// </summary>
        public void Normalize()
        {
            if (!IsZero(this))
                for (int i = 0; i < Size; i++)
                    Coordinates[i] /= Length;
        }

        /// <summary>
        /// Срез координат вектора
        /// </summary>
        /// <param name="from">С этого индекса</param>
        /// <param name="to">По этот индекс</param>
        /// <returns>Возвращает массив координат либо null, если индексы неккоректны</returns>
        public double[] Slice(int from, int to)
        {
            if (from < 0 || from > to || to > Size)
                return null;
            int size = to - from + 1;
            double[] c = new double[size];
            for (int i = from; i <= to; i++)
                c[i - from] = Coordinates[i];
            return c;
        }

        /// <summary>
        /// Умножение координат вектора на константу
        /// </summary>
        /// <param name="v">Вектор</param>
        /// <param name="c">Константа</param>
        /// <returns>Вектор с новыми координатами</returns>
        public static Vector VecByConst(Vector v, double c)
        {
            double[] new_c = new double[v.Size];
            System.Array.Copy(v.Coordinates, new_c, v.Size);

            for (int i = 0; i < v.Size; i++)
                new_c[i] *= c;

            return new Vector(new_c);
        }
        
        /// <summary>
        /// Случайное число с плавающей точкой в заданном диапазоне
        /// </summary>
        /// <param name="a">Нижняя граница</param>
        /// <param name="b">Верхняя граница</param>
        /// <returns>Случайное число</returns>
        private static double GetRandomNumber(double a, double b) => rnd.NextDouble() * (b - a) + a;

        /// <summary>
        /// Получение нулевого вектора
        /// </summary>
        /// <param name="size">Размерность</param>
        /// <returns>Нулевой вектор</returns>
        public static Vector Zero(int size)
        {
            double[] c = new double[size];
            for (int i = 0; i < size; i++)
                c[i] = 0d;
            return new Vector(c);
        }

        /// <summary>
        /// Получение вектора, состоящего из единиц
        /// </summary>
        /// <param name="size">Размерность</param>
        /// <returns>Вектор из единиц</returns>
        public static Vector E(int size)
        {
            double[] c = new double[size];
            for (int i = 0; i < size; i++)
                c[i] = 1d;
            return new Vector(c);
        }

        /// <summary>
        /// Получение вектора со случайными координатами в заданном диапазоне
        /// </summary>
        /// <param name="size">Размерность</param>
        /// <param name="min">Нижняя граница</param>
        /// <param name="max">Верхняя граница</param>
        /// <returns>Вектор со случайными координатами</returns>
        public static Vector RandomVector(int size, double min, double max)
        {
            double[] c = new double[size];
            for (int i = 0; i < size; i++)
                c[i] = GetRandomNumber(min, max);
            return new Vector(c);
        }

        /// <summary>
        /// Нулевой ли вектор?
        /// </summary>
        /// <param name="v">Вектор</param>
        /// <returns>Является ли вектор нулевым</returns>
        public static bool IsZero(Vector v)
        {
            for (int i = 0; i < v.Size; i++)
                if (v.Coordinates[i] != 0)
                    return false;
            return true;
        }

        /// <summary>
        /// Преобразование в строку
        /// </summary>
        /// <returns>Строковое представление вектора</returns>
        public override string ToString()
        {
            string output = string.Empty;
            for (int i = 0; i < Size; i++)
                output += $" {Coordinates[i]} ;";
            output = output.Remove(0, 1);
            output = output.Remove(output.Length - 2, 2);
            return $"{{{output}}}";
        }

        /// <summary>
        /// Сложение векторов
        /// </summary>
        /// <param name="v1">Первый вектор</param>
        /// <param name="v2">Второй вектор</param>
        /// <returns>Сумма двух векторов или null, если у них разная размерность</returns>
        public static Vector operator +(Vector v1, Vector v2)
        {
            if (v1.Size != v2.Size)
                return null;

            double[] new_c = new double[v1.Size];
            for (int i = 0; i < v1.Size; i++)
                new_c[i] = v1.Coordinates[i] + v2.Coordinates[i];

            return new Vector(new_c);
        }

        /// <summary>
        /// Вычитание векторов
        /// </summary>
        /// <param name="v1">Первый вектор</param>
        /// <param name="v2">Второй вектор</param>
        /// <returns>Сумма двух векторов или null, если у них разная размерность</returns>
        public static Vector operator -(Vector v1, Vector v2)
        {
            if (v1.Size != v2.Size)
                return null;

            double[] new_c = new double[v1.Size];
            for (int i = 0; i < v1.Size; i++)
                new_c[i] = v1.Coordinates[i] - v2.Coordinates[i];

            return new Vector(new_c);
        }

        /// <summary>
        /// Скалярное произведение векторов
        /// </summary>
        /// <param name="v1">Первый вектор</param>
        /// <param name="v2">Второй вектор</param>
        /// <returns>ЛК координат двух векторов или double.NaN, если у них разная размерность</returns>
        public static double operator *(Vector v1, Vector v2)
        {
            if (v1.Size != v2.Size)
                return double.NaN;

            double product = 0.0d;
            for (int i = 0; i < v1.Size; i++)
                product += v1.Coordinates[i] * v2.Coordinates[i];

            return product;
        }
    }
}
