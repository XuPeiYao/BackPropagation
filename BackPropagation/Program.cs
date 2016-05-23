using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation {
    class Program {
        static void Main(string[] args) {
            //建構一個輸入層數值數量為2 感知器權重亂數範圍為0~1，預設激發函數為LogisticSigmoid
            //隱藏層節點數量為4 輸出層數量為1的神經網路
            PerceptronNetwork network = new PerceptronNetwork(2, 0, 1, new int[] { 4,1 });
            LearnData[] data = new LearnData[] {//學習用資料
                new LearnData() { Input = new double[] {0,0 },Output = new double[] { 0 }},
                new LearnData() { Input = new double[] {1,1 },Output = new double[] { 0 }},
                new LearnData() { Input = new double[] {1,0 },Output = new double[] { 1 }},
                new LearnData() { Input = new double[] {0,1 },Output = new double[] { 1 }},
            };
            
            //呼叫學習函數，速率0.8，目標誤差值0.05
            network.Train(data,0.8,0.05);

            //測試
            foreach (var item in data) {
                Console.WriteLine($"測試({string.Join(",", item.Input)}) => {string.Join(",", network.Compute(item.Input))}");
            }

            //儲存神經網路結果
            StreamWriter writer = new StreamWriter("output.json");
            writer.Write(network.ToJObject());//匯出JSON，也可使用Load方法反序列化
            writer.Close();

            Console.ReadKey();
        }
    }
}
