using BackPropagation.Extensions;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public class PerceptronLayer {
        /// <summary>
        /// 目前神經網路層所含感知器節點集合
        /// </summary>
        public IPerceptron[] Instances { get;private set; }

        /// <summary>
        /// 目前神經網路層所需要的輸入值數量
        /// </summary>
        [JsonIgnore]
        public int InputCount { get; private set; }

        /// <summary>
        /// 目前神經網路層所含感知器數量
        /// </summary>
        [JsonIgnore]
        public int Length => Instances.Length;

        /// <summary>
        /// 取得或指定索引感知器
        /// </summary>
        /// <param name="Index">索引</param>
        /// <returns>感知器</returns>
        public IPerceptron this[int Index] {
            get {
                return Instances[Index];
            }
            set {
                Instances[Index] = value;
            }
        }

        /// <summary>
        /// 初始化神經網路層
        /// </summary>
        /// <param name="InputCount">輸入值長度</param>
        /// <param name="InstancesCount">感知器數量</param>
        /// <param name="RandomMinWeight">感知器權值亂數最小值</param>
        /// <param name="RandomMaxWeight">感知器權值亂數最大值</param>
        public PerceptronLayer(int InputCount,int InstancesCount,double RandomMinWeight = -1,double RandomMaxWeight = 1) {
            this.InputCount = InputCount;
            this.Instances = Enumerable
                            .Range(0, InstancesCount)
                            .Select(x => (IPerceptron)new Perceptron(InputCount,RandomMinWeight,RandomMaxWeight))
                            .ToArray();

            if (InstancesCount == 0) this.Instances = null;
        }

        /// <summary>
        /// 計算輸入值產生的結果
        /// </summary>
        /// <param name="Input">輸入值</param>
        /// <returns>運算結果</returns>
        public double[] Compute(params double[] Input) {
            return Compute(Input.ToList());
        }

        /// <summary>
        /// 計算輸入值產生的結果
        /// </summary>
        /// <param name="Input">輸入值</param>
        /// <returns>運算結果</returns>
        public double[] Compute(List<double> Input) {
            return Instances.Select(x => x.Compute(Input)).ToArray();
        }
    }
}
