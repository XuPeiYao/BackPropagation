using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    public class LearnData {
        public double[] Input { get; set; }
        public double[] Output { get; set; }

        /// <summary>
        /// 資料集標準化
        /// </summary>
        /// <param name="Data">原始資料集</param>
        /// <param name="InputMin">輸入標準最小值</param>
        /// <param name="InputMax">輸入標準最大值</param>
        /// <param name="OutputMin">輸出標準最小值</param>
        /// <param name="OutputMax">輸出標準最大值</param>
        public static void Standardize(LearnData[] Data,int InputMin = 0,int InputMax = 1,int OutputMin = 0,int OutputMax = 1) {
            var DataInputMax = Enumerable.Range(0, Data.First().Input.Length).Select(
                i => Data.Select(x=>x.Input).Select(x=>x[i]).Max()
            ).ToArray();
            var DataInputMin = Enumerable.Range(0, Data.First().Input.Length).Select(
                i => Data.Select(x => x.Input).Select(x => x[i]).Min()
            ).ToArray();
            var DataOutputMax = Enumerable.Range(0, Data.First().Output.Length).Select(
                i => Data.Select(x => x.Output).Select(x => x[i]).Max()
            ).ToArray();
            var DataOutputMin = Enumerable.Range(0, Data.First().Output.Length).Select(
                i => Data.Select(x => x.Output).Select(x => x[i]).Min()
            ).ToArray();

            for(int i = 0; i < Data.Length; i++) {
                Data[i].Input = Data[i].Input.Select((x, i2) =>
                    ConvertValue((x - DataInputMin[i2]) / (DataInputMax[i2] - DataInputMin[i2]), InputMin, InputMax)
                ).ToArray();
                Data[i].Output = Data[i].Output.Select((x, i2) =>
                    ConvertValue((x - DataOutputMin[i2]) / (DataOutputMax[i2] - DataOutputMin[i2]), OutputMin, OutputMax)
                ).ToArray();
            }
        }

        private static double ConvertValue(double Value,int Min,int Max) => Value * (Max - Min) + Min;
    }
}
