using BackPropagation.Extensions;
using JsonNet.PrivateSettersContractResolvers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public class PerceptronNetwork {
        /// <summary>
        /// 神經網路所含層集合
        /// </summary>
        public PerceptronLayer[] Layers { get; private set; }

        /// <summary>
        /// 取得神經網路所含層的數量
        /// </summary>
        [JsonIgnore]
        public int Length => Layers.Length;


        /// <summary>
        /// 取得神經網路輸入值長度
        /// </summary>
        [JsonIgnore]
        public int InputCount => Layers.First().InputCount;

        /// <summary>
        /// 取得指定索引之神經網路層
        /// </summary>
        /// <param name="Index">索引</param>
        /// <returns>神經網路層</returns>
        public PerceptronLayer this[int Index] {
            get {
                return Layers[Index];
            }
        }

        /// <summary>
        /// 初始化神經網路
        /// </summary>
        /// <param name="InputCount">輸入值長度</param>
        /// <param name="RandomMinWeight">感知器權值亂數最小值</param>
        /// <param name="RandomMaxWeight">感知器權值亂數最大值</param>
        /// <param name="MiddleLayerInstanceCount">除輸入層外所有層所含節點數(最後一層表示輸出值數量)</param>
        public PerceptronNetwork(int InputCount, double RandomMinWeight, double RandomMaxWeight, params int[] MiddleLayerInstanceCount) {
            if (MiddleLayerInstanceCount == null) return;

            List<PerceptronLayer> LayersList = new List<PerceptronLayer>();
            for (int i = 0; i < MiddleLayerInstanceCount.Length; i++) {
                if (i == 0) {
                    LayersList.Add(new PerceptronLayer(InputCount, MiddleLayerInstanceCount[i], RandomMinWeight, RandomMaxWeight));
                } else {
                    LayersList.Add(new PerceptronLayer(LayersList[i - 1].Instances.Length, MiddleLayerInstanceCount[i], RandomMinWeight, RandomMaxWeight));
                }
            }
            Layers = LayersList.ToArray();
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
            double[] result = Input.ToArray();
            for (int i = 0; i < Layers.Length; i++) {
                result = Layers[i].Compute(result);
            }
            return result;
        }

        /// <summary>
        /// 計算輸入值產生的結果
        /// </summary>
        /// <param name="Input">學習資料</param>
        /// <returns>運算結果</returns>
        public double[] Compute(LearnData Input) {
            return Compute(Input.Input);
        }

        internal LearnPerceptron[][] LearnCompute(params double[] Input) {
            List<LearnPerceptron[]> result = new List<LearnPerceptron[]>();
            double[] input = Input.ToArray();
            for (int i = 0; i < Layers.Length; i++) {
                input = Layers[i].Compute(input);
                
                result.Add(input.Select((x,i2)=>
                    new LearnPerceptron(i == 0 ? InputCount : Layers[i].InputCount) {
                        Result = input[i2]
                    }
                ).ToArray());
            }
            return result.ToArray();
        }

        internal LearnPerceptron[][] GetLearnPerceptrons() {
            return Layers.Select((x, i) =>
                x.Instances.Select((y, i2) => new LearnPerceptron(y.Length)).ToArray()
            ).ToArray();
        }

        internal class LearnPerceptron {
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
            internal double[] WeightsDelta { get; set; }

            /// <summary>
            /// 最後閥值修正值，作為慣性動量依據
            /// </summary>
            internal double Last_ThresholdDelta { get; set; }

            /// <summary>
            /// 最後權重修正值集合，作為慣性動量依據
            /// </summary>
            internal double[] Last_WeightsDelta { get; set; }
            #endregion

            internal LearnPerceptron(int InputCount) {
                WeightsDelta = Enumerable.Range(0, InputCount).Select(x => 0.0).ToArray();
                Last_WeightsDelta = Enumerable.Range(0, InputCount).Select(x => 0.0).ToArray();
            }
        }

        /// <summary>
        /// 反傳導學習方法
        /// </summary>
        /// <param name="Data">學習資料集和</param>
        /// <param name="Rate">學習速率</param>
        /// <param name="Momentum">慣性動量</param>
        /// <param name="Difference">目標誤差值</param>
        /// <param name="Iterations">迭代次數限制</param>
        public void Train(LearnData[] Data, double Rate, double Momentum = 0, double Difference = 0, int Iterations = -1) {
            for (int i = 0; Iterations == -1 || i < Iterations; i++) {
                //學習用暫存資料
                LearnPerceptron[][] LearnNodes = GetLearnPerceptrons();

                double Error = 0;
                foreach (LearnData Item in Data) {
                    Error += BackPropagate(LearnNodes,Item, Rate);
                }
                
                #region 修正
                for (int layer = Length - 1; layer > -1; layer--) {
                    for (int instance = 0; instance < this[layer].Length; instance++) {
                        //修正閥值
                        this[layer][instance].Threshold += LearnNodes[layer][instance].ThresholdDelta + LearnNodes[layer][instance].Last_ThresholdDelta * Momentum;// / Math.Sqrt(Data.Length);
                        for (int weight = 0; weight < this[layer][instance].Length; weight++) {
                            this[layer][instance][weight] += LearnNodes[layer][instance].WeightsDelta[weight] + LearnNodes[layer][instance].Last_WeightsDelta[weight] * Momentum;// / Math.Sqrt(Data.Length);
                        }
                    }
                }
                #endregion

                #region 備份修正值
                for(int layer = Length - 1;layer > -1;layer--) {
                    for(int instance = 0; instance < this[layer].Length; instance++) {
                        LearnNodes[layer][instance].Last_ThresholdDelta = LearnNodes[layer][instance].ThresholdDelta;
                        LearnNodes[layer][instance].Last_WeightsDelta = LearnNodes[layer][instance].WeightsDelta;
                    }
                }
                #endregion

                Console.WriteLine($"迭代:{i}\t{Error}");
                if (Error <= Difference) {
                    break;
                }
            }
        }

        /// <summary>
        /// 反傳導學習子方法
        /// </summary>
        /// <param name="Data">學習資料</param>
        /// <param name="Rate">學習速率</param>
        /// <returns></returns>
        private double BackPropagate(LearnPerceptron[][] LearnNodes, LearnData Data, double Rate) {
            LearnPerceptron[][] ComputeResult = LearnCompute(Data.Input);

            double Error = 0;

            #region 感知器節點運算結果誤差
            for (int layer = Length - 1; layer > -1; layer--) {
                for (int instance = 0; instance < this[layer].Length; instance++) {
                    if (layer == Length - 1) {//輸出層
                        LearnNodes[layer][instance].ResultDelta =
                            (Data.Output[instance] - ComputeResult[layer][instance].Result) *
                            this[layer][instance].DiffActivation(ComputeResult[layer][instance].Result);
                        Error += Math.Abs(Data.Output[instance] - ComputeResult[layer][instance].Result);
                    } else {//隱藏層
                        //下層偏差值加權總合
                        double sumDelta = 0;// ComputeResult[layer][instance].Result
                        for (int nextInstance = 0; nextInstance < Layers[layer + 1].Length; nextInstance++) {
                            sumDelta += //下一層的節點的誤差加權
                                LearnNodes[layer + 1][nextInstance].ResultDelta *
                                this[layer + 1][nextInstance][instance];
                        }
                        LearnNodes[layer][instance].ResultDelta = sumDelta *
                            this[layer][instance].DiffActivation(ComputeResult[layer][instance].Result);
                    }
                }
            }
            #endregion

            #region 計算感知器節點的權重與閥值修正值
            for (int layer = Length - 1; layer > -1; layer--) {
                for (int instance = 0; instance < this[layer].Length; instance++) {
                    //修正閥值
                    LearnNodes[layer][instance].ThresholdDelta += -Rate * LearnNodes[layer][instance].ResultDelta;
                    for (int weight = 0; weight < this[layer][instance].Length; weight++) {
                        if (layer == 0) {//輸入層
                                         //本節點偏差與輸入值
                            for (int upInstance = 0; upInstance < Data.Input.Length; upInstance++) {
                                LearnNodes[layer][instance].WeightsDelta[upInstance] += Rate * LearnNodes[layer][instance].ResultDelta * Data.Input[upInstance];
                            }
                        } else {
                            //本節點偏差與上層節點給的輸入
                            for (int upInstance = 0; upInstance < this[layer - 1].Length; upInstance++) {
                                LearnNodes[layer][instance].WeightsDelta[upInstance] += Rate * LearnNodes[layer][instance].ResultDelta * ComputeResult[layer - 1][upInstance].Result;
                            }
                        }
                    }
                }
            }
            #endregion

            return Error;
        }

        /// <summary>
        /// 將目前神經網路轉換為JSON格式用以儲存
        /// </summary>
        /// <returns>JSON資料</returns>
        public JObject ToJObject() => JObject.FromObject(this);

        /// <summary>
        /// 由JSON格式資料中重建神經網路
        /// </summary>
        /// <param name="Json">JSON資料</param>
        /// <returns>神經網路實體</returns>
        public static PerceptronNetwork Load(JObject Json) {
            var settings = new JsonSerializerSettings {
                ContractResolver = new PrivateSetterContractResolver()
            };
            return JsonConvert.DeserializeObject<PerceptronNetwork>(Json.ToString(), settings);
        }

    }
}
