using BackPropagation.Extensions;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public class Perceptron : IPerceptron {
        #region 用以學習使用的暫存欄位
        /// <summary>
        /// 最後一次的運算輸出
        /// </summary>
        internal double Result { get; set; }

        /// <summary>
        /// 運算輸出與預期結果的誤差
        /// </summary>
        internal double ResultDelta { get; set; }

        /// <summary>
        /// 閥值修正值
        /// </summary>
        internal double ThresholdDelta { get; set; }

        /// <summary>
        /// 權重修正值集合
        /// </summary>
        internal double[] WeightDelta { get; set; }

        /// <summary>
        /// 最後閥值修正值，作為慣性動量依據
        /// </summary>
        internal double Last_ThresholdDelta { get; set; }

        /// <summary>
        /// 最後權重修正值集合，作為慣性動量依據
        /// </summary>
        internal double[] Last_WeightDelta { get; set; }
        #endregion

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
            = ActivationFunction.LogisticSigmoid;

        /// <summary>
        /// 激發函數的一階微分
        /// </summary>
        [JsonIgnore]
        public Func<double, double> DiffActivation { get; set; } 
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
            WeightDelta = Enumerable.Range(0, InputCount).Select(x => 0.0).ToArray();
            Last_WeightDelta = Enumerable.Range(0, InputCount).Select(x => 0.0).ToArray();
            Threshold = rand.NextDouble(-1, 1);
        }

        /// <summary>
        /// 清除最後一次學習迭代產生的數據
        /// </summary>
        internal void ClearResult() {
            Result = 0;
            ResultDelta = 0;
            ThresholdDelta = 0;
            WeightDelta = new double[WeightDelta.Length];
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
            return Result = Activation(Input.Select((x, i) => x * Weights[i]).Sum() - Threshold);
        }
    }
}
