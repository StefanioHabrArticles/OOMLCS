namespace MachineLearning
{
    /// <summary>
    /// Интерфейс функции активации
    /// </summary>
    public interface IActivationFunction
    {
        /// <summary>
        /// Функция активации
        /// </summary>
        /// <param name="x">Аргумент</param>
        /// <returns>Значение функции в точке</returns>
        double Activate(double x);
        
        /// <summary>
        /// Производная функции активации
        /// </summary>
        /// <param name="x">Аргумент</param>
        /// <returns>Значение производной в точке</returns>
        double Derivate(double x);
    }
}
