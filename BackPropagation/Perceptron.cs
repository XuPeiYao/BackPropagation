using BackPropagation.Extensions;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public class Perceptron : IPerceptron {
        /// <summary>
        /// 權重集合
        /// </summary>
        public double[] Weights { get;private set; }

        /// <summary>
        /// 閥值
        /// </summary>
        public double Threshold { get; set; } = 0;

        /// <summary>
        /// 激發函數
        /// </summary>
        [JsonIgnore]
        public Func<double, double> Activation { get; set; }
            = Default_Activation;

        /// <summary>
        /// 激發函數的一階微分
        /// </summary>
        [JsonIgnore]
        public Func<double, double> DiffActivation { get; set; } 
            = Default_DiffActivation;

        /// <summary>
        /// 預設激發函數
        /// </summary>
        public static Func<double, double> Default_Activation { get; set; }
            = ActivationFunction.LogisticSigmoid;

        /// <summary>
        /// 預設激發函數的一階微分
        /// </summary>
        public static Func<double, double> Default_DiffActivation { get; set; }
            = ActivationFunction.Diff_LogisticSigmoid;


        /// <summary>
        /// 取得或設定指定索引之權值
        /// </summary>
        /// <param name="Index">索引</param>
        /// <returns>權值</returns>
        public double this[int Index] {
            get {
                return Weights[Index];
            }
            set {
                Weights[Index] = value;
            }
        }

        /// <summary>
        /// 目前感知器權值數量，即輸入值長度
        /// </summary>
        [JsonIgnore]
        public int Length => Weights.Length;


        /// <summary>
        /// 初始化感知器物件
        /// </summary>
        /// <param name="InputCount">輸入值長度</param>
        [JsonConstructor]
        public Perceptron(int InputCount) : this(InputCount, -1, 1) { }

        /// <summary>
        /// 初始化感知器物件
        /// </summary>
        /// <param name="InputCount">輸入值長度</param>
        /// <param name="RandomMinWeight">權值亂數最小值</param>
        /// <param name="RandomMaxWeight">權值亂數最大值</param>
        public Perceptron(int InputCount, double RandomMinWeight, double RandomMaxWeight) {
            if (InputCount == 0) return;

            Random rand = new Random(DateTime.Now.Millisecond);
            Weights = Enumerable.Range(0, InputCount).Select(x => rand.NextDouble(RandomMinWeight, RandomMaxWeight)).ToArray();
            Threshold = rand.NextDouble(-1, 1);
        }

       
        /// <summary>
        /// 計算輸入值產生的結果
        /// </summary>
        /// <param name="Input">輸入值</param>
        /// <returns>運算結果</returns>
        public double Compute(params double[] Input) {
            return Compute(Input.ToList());
        }

        /// <summary>
        /// 計算輸入值產生的結果
        /// </summary>
        /// <param name="Input">輸入值</param>
        /// <returns>運算結果</returns>
        public double Compute(List<double> Input) {
            return Activation(Input.Select((x, i) => x * Weights[i]).Sum() - Threshold);
        }
    }
}
